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
//! ⚠️ **`--ignored` is a blanket, and one of the tests it selects is not a smoke.**
//! [`long_clip_coherence_under_the_bounded_window`] is the sc-15127/sc-15571 research sweep: 85 min at
//! its default single seed, and the seeds it needs to return a verdict at all put it over four hours.
//! The real-weight lane therefore `--skip`s it and dispatches it as its own job (sc-17276); add
//! `--skip long_clip_coherence_under_the_bounded_window` when you want the ~20-minute smoke set.
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
use mlx_gen_krea_realtime::{
    decode_latents_to_video, decode_tiling, KreaRealtimeConfig, KvCacheQuant, MODEL_ID,
};

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
/// [`long_clip_coherence_under_the_bounded_window`] (sc-15127/sc-15585), which measured real drift but
/// did not resolve whether the bounded KV window causes it.
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
    fn tiny_safetensors(path: &Path, fill: u8) {
        let mut header = serde_json::to_vec(&serde_json::json!({
            "weight": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}
        }))
        .unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut bytes = Vec::with_capacity(8 + header.len() + 8);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend(std::iter::repeat_n(fill, 8));
        std::fs::write(path, bytes).unwrap();
    }
    // The provider now seals the complete physical receipt before construction, so this
    // weights-free validation uses a structurally valid tiny q4 inventory rather than an empty dir.
    let dir_tmp = tempfile::tempdir().unwrap();
    let dir = dir_tmp.path().to_path_buf();
    std::fs::write(
        dir.join("config.json"),
        br#"{"quantization":{"bits":4,"group_size":64}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    tiny_safetensors(&dir.join("dit.safetensors"), 1);
    tiny_safetensors(&dir.join("t5_encoder.safetensors"), 2);
    tiny_safetensors(&dir.join("vae.safetensors"), 3);
    let spec = LoadSpec::new(WeightsSource::Dir(dir));
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
/// The finding this encodes: v2v@0 is identity **in the latents** — the decisive form of that is
/// [`v2v_strength_zero_is_latent_identity`], which drives the AR loop on encoded latents with no VAE
/// between the source and the answer and measures the residual directly. What is left in `C` is that
/// residual carried through the decode, not the `.sample()` draw: `A'` measures the draw at ~0.01/255,
/// two orders of magnitude under `C`, so sampling cannot be the explanation. See the MEASURED block on
/// the assertions for the numbers and for what `C` is gated against.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn v2v_strength_zero_preserves_the_source() {
    use mlx_gen_wan::{preprocess_i2v_image, WanVae};
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::random;

    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);

    // Pin the decode budget across BOTH the product render and the control below (sc-17276).
    // `decode_tiling` is free-aware — `free x 0.85`, `free = MLX limit - resident` — so with no pin the
    // plan is a function of whatever happened to be free at each call, and the two calls are made at
    // very different residencies: the product decodes inside `run` with the DiT and TE resident, the
    // control decodes afterwards with them dropped. A host big enough to fit a single-pass decode also
    // takes the `None` branch for both, which silently changes what `A` measures. That free variable is
    // literally the second cause the `C` assertion names, and it must not be one of this test's inputs.
    // Same pin, same reason, as `long_clip_coherence_under_the_bounded_window`; 20 GiB is the operating
    // point `decode_tiling`'s own docs derive at 832x480.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");

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
    // Both decodes are done, so the pin has served its purpose. Dropped HERE rather than at the end of
    // the test so the assertions below cannot leak a 20 GiB budget into whatever runs next in this
    // process — these tests share one process under `--test-threads 1`, and libtest carries on to the
    // next test after a panic. A panic ABOVE this line still leaks, as it does at the other four pin
    // sites in this file; today that is inert because this test sorts last of the six the lane runs.
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
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
    // MEASURED (832x480, 33 frames, Q4, budget pinned to 20 GiB ⇒ spatial 256/64, no temporal tiling):
    // A = 2.98, A' = 0.01, B = 2.94, **C = 0.37**.
    //
    // So: v2v at strength 0 IS identity — the output sits ~0.4/255 from a plain VAE round-trip of its
    // own source, and `B` tracks `A` to well under a tenth of `A`, i.e. the whole distance from the
    // source to the v2v@0 clip is the VAE's own round-trip and nothing else moved the picture.
    //
    // WHERE THE REMAINING ~0.4 COMES FROM, and why it is not gated against `A'`: it is the AR loop's
    // numerical residual. [`v2v_strength_zero_is_latent_identity`] measures that residual directly in
    // the latents — it gates it at 2% of the source scale, and on the same Q4 snapshot both this
    // machine and CI run 30787887176 measure it at 0.91% (mean |d| 0.01044 against a source mean |x|
    // of 1.15338). Carried through the decode, that is the ~0.4/255 seen here. It is NOT the
    // `.sample()` draw: `A'` measures that draw at ~0.01/255, ~40x smaller.
    //
    // WHY THIS IS AN ABSOLUTE BUDGET AND NOT `C < A * 0.1` (sc-17276). That relative form was
    // calibrated by sc-8446 (1deefff6, #287) when `A` was ~27.6 — the tiled decode of the day, which
    // sc-15325 (51d65a1a, #292) then fixed. Post-fix a tiled decode reaches single-pass quality (the
    // spatial-only plan this bucket selects is 0.31/255 against single-pass, `decode_tiling`'s docs),
    // so `A` collapsed ~9x to ~3.0 while `C`, an AR-loop residual, did not scale with it at all;
    // 0.1*A became 0.30 against a 0.37-0.38 measurement and the gate has been latently red ever since.
    // `C` is not a fraction of the VAE's error, so it must not be gated as one.
    //
    // AND WHY THE PIN IS STILL LOAD-BEARING even though it did not cause that red (unpinned this
    // measures C = 0.38, pinned 0.37 — the old gate failed at both). It is the same sc-15325 fix that
    // makes it necessary: a control/product decode-plan mismatch used to be worth ~26/255 and was
    // caught by any sane gate, and post-fix it is worth ~0.3/255 — INSIDE this budget. The gate can no
    // longer see that confound, so the pin removes it by construction instead.
    //
    // THE BUDGET'S HEADROOM, both sides, MEASURED — a budget nobody has driven to red is a budget
    // nobody knows the width of. Below: 0.37. Above: the same test body re-run against a
    // strength-0.25 render of the same source, everything else held (sc-17276), measures A = 2.98,
    // B = 13.81, **C = 13.77** — 37x the strength-0 value and 13.8x this budget, and it trips this
    // assertion. So the budget sits between a measured pass at 0.37 and a measured fail at 13.77
    // rather than between a measurement and a guess.
    const V2V0_IDENTITY_BUDGET: f64 = 1.0;
    assert!(
        a > 1.0,
        "the VAE round-trip control is a near no-op (A={a:.2}) — C cannot be interpreted against it"
    );
    assert!(
        c < V2V0_IDENTITY_BUDGET,
        "v2v@0 is {c:.2}/255 from a VAE round-trip of its own source, over the \
         {V2V0_IDENTITY_BUDGET:.2}/255 identity budget — the strength=0 denoise is not behaving as an \
         identity (A={a:.2}, A'={a_prime:.2}). The decode is not a candidate explanation the way it \
         used to be: both clips above went through one pinned plan"
    );
    // NOT an independent measurement — `|B - A| <= C` always, so this can only fire where `C` already
    // could. It is the tighter bound on the DIRECTIONAL part of `C`, which is the part a pipeline that
    // renders instead of preserving produces: the strength-0.25 arm above moves B to 13.81 against A's
    // 2.98. Cheap, and it fails ~3x sooner than the absolute budget when the deviation is one-way.
    assert!(
        (b - a).abs() < a * 0.1,
        "source->v2v@0 is {b:.2}/255 against a {a:.2}/255 VAE round-trip of the same source — a \
         strength-0 render must be its source's round-trip and nothing more, so these two must agree"
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

/// **Measured KV-cache residency at the production geometry, per storage tier.** The self-attention
/// KV cache holds post-RoPE **activations**, so the *weight* tier does not shrink it: a Q4 DiT still
/// carries a bf16 cache, which is the whole point of measuring it next to the Q4 weights. What *does*
/// shrink it is [`KreaArConfig::kv_cache_quant`](mlx_gen_krea_realtime::KreaArConfig::kv_cache_quant)
/// (sc-17807), so this drives the real DiT chunk-by-chunk **once per tier** — the shipped bf16 cache
/// and Q8 — over the same bounded window the pipeline uses. sc-17894 trims the cache to the actual
/// next read before measuring: at the shipped geometry that is one 4,680-token chunk, not the
/// 9,360-token attention window (which includes the next chunk's own K/V).
///
/// The gate is the one that makes the published per-token table evidence rather than arithmetic:
/// measured `retained_bytes` must equal
/// [`kv_bytes_per_token`](mlx_gen_krea_realtime::KreaRealtimeConfig::kv_bytes_per_token) × retained
/// tokens **exactly**, on real weights, for both tiers. A table that drifted from the allocator, or a
/// tier that silently stored dense, fails here.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn kv_cache_residency_at_the_production_geometry() {
    use mlx_gen_krea_realtime::{
        load_krea_realtime_transformer_with_quant, CausalKreaTransformer, CausalKvCache,
    };
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
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);
    // The bound is structural: cached history is the attention window minus the next chunk, plus a
    // disjoint always-on sink prefix (zero in this production row).
    let max_tokens = cfg
        .ar
        .max_attention_size()
        .saturating_sub(cfg.ar.block_size())
        + cfg.ar.sink_tokens();
    let mut peak_by_tier: Vec<(Option<KvCacheQuant>, usize, usize)> = Vec::new();

    for tier in [None, Some(KvCacheQuant::Q8)] {
        let mut tier_cfg = cfg.clone();
        tier_cfg.ar.kv_cache_quant = tier;
        let per_token = tier_cfg
            .kv_bytes_per_token()
            .expect("the tier must be expressible at this backbone's geometry");
        // Built from the config rather than `transformer.new_cache()` so both tiers run against the
        // one loaded DiT (the transformer bakes its tier at construction).
        let mut cache = CausalKvCache::new(
            tier_cfg.wan.num_layers,
            tier_cfg.ar.max_attention_size(),
            tier_cfg.ar.sink_tokens(),
            tier,
        );
        let label = match tier {
            None => "bf16".to_string(),
            Some(q) => format!("q{}/g{}", q.bits, q.group_size),
        };
        println!("  --- KV tier {label} ({per_token} bytes/token) ---");

        let mut start = 0usize;
        let mut peak_retained_bytes = 0usize;
        // The ALLOCATOR's peak, not the cache's own accounting. `retained_bytes` proves the packed
        // cache is smaller; only this proves the packed path is smaller *in practice* — see the
        // assertion after the loop for the mechanism it guards.
        mlx_rs::memory::reset_peak_memory();
        for c in 0..chunks {
            let chunk = Array::zeros::<f32>(&[
                cfg.wan.in_dim as i32,
                fpb,
                latent_h as i32,
                latent_w as i32,
            ])
            .expect("chunk");
            let velocity = transformer
                .forward_chunk(&chunk, 500.0, &cross_kv, start, &mut cache)
                .expect("causal chunk forward");
            start += (fpb as usize) * cfg.ar.frame_seq_length;
            // Supply the ACTUAL next chunk length before measuring. This is the sc-17894 mechanism:
            // `window_prev` evicts exactly the tokens that read cannot consume, and doing it here
            // also proves the packed and dense caches expose the same retained shape.
            cache
                .window_prev(cfg.ar.block_size())
                .expect("trim to the next read");
            // MLX is LAZY: without this `eval` the whole chunk forward is an unexecuted graph, the
            // cache arrays are unmaterialized, and `get_active_memory()` reads ~0 — the measurement
            // would be a shape calculation dressed up as a memory reading. Forcing the cache (and the
            // velocity that depends on the same graph) makes the reported bytes and the allocator's
            // active figure both real, and makes the run take actual GPU time.
            //
            // `eval_retained` materializes the cache **as stored** and `retained_bytes` reports that
            // same representation, so a quantized cache (sc-17807) is measured at its packed cost.
            // Evaluating `layer_kv` and summing its `nbytes` instead would dequantize first and
            // report every tier at the bf16 size — blind to the thing the knob changes.
            mlx_rs::transforms::eval([&velocity]).expect("materialize the chunk forward");
            cache.eval_retained().expect("materialize the KV cache");
            let bytes = cache.retained_bytes();
            peak_retained_bytes = peak_retained_bytes.max(bytes);
            println!(
                "  chunk {c:>2}: committed {} tok, next-read retained {} tok, KV {:.2} GiB, MLX active {:.2} GiB",
                cache.stored_tokens(),
                cache.retained_tokens(),
                gib(bytes),
                gib(mlx_rs::memory::get_active_memory()),
            );
            // The published per-token cost, checked against the allocator on real weights every
            // chunk — including the chunks before eviction starts and the ones after it.
            assert_eq!(
                bytes,
                cache.retained_tokens() * per_token,
                "{label}: {} retained tokens should cost {per_token} bytes each",
                cache.retained_tokens()
            );
        }

        println!(
            "  KV-cache residency at {w}x{h} ({} tok/frame, window {} frames, KV {label}): \
             {:.2} GiB (MLX active after PARTIAL staging -- text embed + cross-attn k/v only, ~15% \
             of the DiT: {:.2} GiB)",
            cfg.ar.frame_seq_length,
            cfg.ar.streaming_local_attn_frames(),
            gib(peak_retained_bytes),
            gib(partially_staged),
        );
        assert!(
            peak_retained_bytes > 0,
            "{label}: the KV cache never retained anything — the measurement is inert"
        );
        assert!(
            cache.retained_tokens() <= max_tokens,
            "{label}: retained {} tokens > the actual next read's {max_tokens} — eviction is not \
             trimming the cache",
            cache.retained_tokens()
        );
        let mlx_peak = mlx_rs::memory::get_peak_memory();
        println!(
            "  {label}: MLX active PEAK across the chunk loop {:.2} GiB",
            gib(mlx_peak)
        );
        peak_by_tier.push((tier, peak_retained_bytes, mlx_peak));
        mlx_rs::memory::clear_cache();
    }

    let (dense_kv, dense_peak) = (peak_by_tier[0].1, peak_by_tier[0].2);
    let (q8_kv, q8_peak) = (peak_by_tier[1].1, peak_by_tier[1].2);
    println!(
        "  Q8 vs bf16 — retained KV {:.2}/{:.2} GiB, MLX active peak {:.2}/{:.2} GiB",
        gib(q8_kv),
        gib(dense_kv),
        gib(q8_peak),
        gib(dense_peak),
    );
    let old_dense_window = cfg.ar.max_attention_size() * cfg.kv_bytes_per_token().unwrap();
    assert_eq!(
        dense_kv * 2,
        old_dense_window,
        "the shipped 6-frame window should retain only the 3-frame cached history the next read uses"
    );
    assert!(
        ((q8_kv as f64 / old_dense_window as f64) - 0.265625).abs() < 1e-9,
        "Q8 plus next-read eviction should compose to 0.265625x of the old bf16 window"
    );

    // **The assertion the prose needed.** `causal.rs` argues that the dequantized read window is an
    // anonymous graph intermediate MLX frees as each layer's attention completes, so it does not
    // land in peak. That argument is sound against the pinned MLX (the tape detaches each node after
    // it evaluates, and `denoise_chunk_inner` drops the window `Vec` before anything evals) — but it
    // is a *premise*, and if it ever fails the quantized path holds all `num_layers` dequantized
    // windows at once and peaks at ~1.53x the dense cache, i.e. WORSE than not quantizing. Nothing
    // else in the tree would notice. This does.
    assert!(
        q8_peak < dense_peak,
        "the Q8 cache peaked at {:.2} GiB against bf16's {:.2} — quantizing did not lower the \
         allocator's peak, which means the dequantized read window is no longer being freed per \
         layer and the packed path now costs MORE than the dense one",
        gib(q8_peak),
        gib(dense_peak),
    );
    // ...and the saving is of the size the per-token table predicts, not a rounding accident. The
    // retained window is identical between tiers, so the whole KV delta should reach the peak.
    let kv_saving = (dense_kv - q8_kv) as f64;
    let peak_saving = dense_peak.saturating_sub(q8_peak) as f64;
    assert!(
        peak_saving > kv_saving * 0.8,
        "the KV cache shrank by {:.2} GiB but the allocator's peak only fell by {:.2} — most of \
         the saving is being spent somewhere else",
        gib(dense_kv - q8_kv),
        gib(dense_peak.saturating_sub(q8_peak)),
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
//   * a **within-regime three-dose response**: rows A/D/F use the same bounded local-attention path
//     at windows 6/15/30 (13/10/5 rolls). The decision statistic is a paired-seed slope, not an
//     endpoint ranking.
//   * a **within-regime zero-eviction rate finding** (row Z): the shipped window on a clip short
//     enough that it never rolls. Its short-segment slope is retained, but the recorded value is
//     higher than row A's long-clip rate and over-predicts row A when extrapolated, so it does not
//     establish a rate floor.
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
///   ≤ 5/255;
/// * **from above** — every plausible AR failure shape (linear wash-out ramp, plateauing wash-out,
///   localized subject collapse, hue rotation) scores above it, several of them by 2–6×.
///
/// **Both sides are synthetic, and that is a real limit.** An earlier revision of this comment also
/// claimed the within-regime zero-eviction row `Z` "bounds this content's own rate". It does not, and
/// the sweep now computes the comparison instead of asserting it ([`S18Sweep::rate_floor_clause`],
/// reported in the verdict): row Z's post segment is only 12 output frames, its measured slope
/// (29.63/100f) is *higher* than the shipped row's over 156 frames (15.56/100f), and extrapolating it
/// across the long clip predicts ~46/255 — roughly 1.9× more drift than row A actually shows. A
/// 12-frame OLS slope is dominated by ordinary frame-to-frame motion and does not extrapolate. Row Z
/// also cannot be made longer: the shipped window evicts as soon as a clip passes 6 latent frames, so
/// **no longer zero-eviction row exists at the shipped window**. There is therefore no measured
/// same-content floor, and "past the budget" means "past an absolute number bracketed by synthetic
/// controls", not "past a measured same-content baseline".
///
/// Those numbers are 640×384. sc-17324 measured row Z at **832×480** — the shipping bucket — for the
/// first time and it comes out the same way, harder: Z runs at 38.29/100f against row A's 18.82,
/// A/Z = 0.49. So the missing floor is not an artifact of the smaller bucket. See
/// [`MEASURED_832_SC17324`] and [`the_sc17324_832_sweep_reports_its_rate_comparison`].
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

/// Which [`DESCRIPTOR_NAMES`] component produced [`worst_drift`]. Recorded per cell so "what mode is
/// this clip failing in?" is answerable from the table instead of from prose.
fn worst_component(d: &[ComponentDrift; N_DESC]) -> &'static str {
    let mut best = (0usize, -1.0f64);
    for (i, c) in d.iter().enumerate() {
        let v = c.trend.abs().max(c.gated_excursion().abs());
        if v > best.1 {
            best = (i, v);
        }
    }
    DESCRIPTOR_NAMES[best.0]
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
    latent_frames: usize,
    /// The verdict statistic: worst trend or worst z-gated excursion (see [`worst_drift`]).
    drift: f64,
    trend: f64,
    excursion: f64,
    /// Worst per-100-frame slope — the only length-comparable statistic, used for row Z.
    slope: f64,
    /// Which descriptor component produced `drift` (see [`worst_component`]).
    component: &'static str,
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
    /// AR-loop wall time in milliseconds — the throughput half of sc-17807's feasibility question.
    /// The story warned that dequantizing per read "may spend back the win in bandwidth"; without a
    /// recorded time the artifacts cannot answer that, only the live log can.
    ar_wall_ms: u64,
}

/// One row of the S18 sweep. Rows share a prompt, seed and geometry, but **not** a clip: changing
/// `local_attn_size` changes the attention mask and therefore the sampling trajectory. Rows D, F and Z
/// are within-regime controls (same bounded local-attention path); row E is an out-of-regime reference.
struct DriftRun {
    row: char,
    label: &'static str,
    window: i64,
    sink: usize,
    /// Latent frames for this row. Row Z is deliberately short — short enough that the shipped window
    /// never evicts — so only its *slope* is comparable with the long rows.
    latent_frames: Option<usize>,
}

/// Number of AR chunks that append past a bounded read window. A negative window is the separate
/// global-attention regime and never evicts.
fn eviction_rolls(latent_frames: usize, frames_per_block: usize, window: i64) -> usize {
    if window < 0 {
        return 0;
    }
    (0..latent_frames.div_ceil(frames_per_block))
        .filter(|chunk| frames_per_block * (chunk + 1) > window as usize)
        .count()
}

/// **sc-15127 (S18) — does a long batch clip drift as the bounded KV window rolls, and does a
/// first-chunk sink anchor fix it?**
///
/// Generates the clip once per (row, seed) — varying only the KV read window, the sink anchor and the
/// seed — scores each with [`score_clip`], and prints the table the verdict rests on. Each row's MLX
/// peak is reported too, so the sink's KV cost is measured rather than asserted: a sink is permanently
/// resident, and the bounded window exists precisely because KV is expensive on this host.
///
/// **Rows.** `A` shipped (window 6, sink 0) · `B` sink 1 · `C` sink 3 · `D` wide window 15 · `F`
/// wider bounded window 30 (together A/D/F form the within-regime dose ladder) · `E` the checkpoint's
/// global window (an out-of-regime *reference*, not a floor, and **not attribution evidence** —
/// different attention mask, `n = 1`, no variance estimate) · `Z` the within-regime zero-eviction
/// *rate* row (shipped window, clip short enough never to roll).
///
/// Row Z was intended as a rate floor and **turned out not to be one** — see
/// [`S18Sweep::rate_floor_clause`] for the measured reason. It is still run, still recorded and still
/// reported in the verdict, because "the intended floor does not work, and here is the number that
/// shows it" is the honest form of that result.
///
/// **Driving it.** `KREA_S18_ROWS` (default `ABCDFEZ`) selects rows; `KREA_S18_SEEDS` (comma-separated,
/// default `7`) selects seeds. Each (row, seed) prints one `S18CELL` TSV line, so a long sweep can be
/// run in pieces and re-aggregated without holding a five-hour process open.
///
/// ⚠️ **Geometry — the global reference is the expensive row.** Row E runs the checkpoint's *global*
/// window, whose KV is `latent_frames × frame_seq_length` tokens at **800 KiB/token** (`2 (K and V) ×
/// 40 layers × 5120 dim × 2 bytes` = 819,200 bytes —
/// [`KreaRealtimeConfig::kv_bytes_per_token`](mlx_gen_krea_realtime::KreaRealtimeConfig::kv_bytes_per_token),
/// measured against the allocator by [`kv_cache_residency_at_the_production_geometry`]). At the
/// 832×480 reference bucket a 45-latent-frame clip is 70,200 tokens ≈ **53.6 GiB** of KV before
/// activations — exactly the ~27 GB-of-KV problem
/// [`mac_ar_config`](mlx_gen_krea_realtime::mac_ar_config) exists to dodge, so its cost is a finding
/// rather than a harness bug.
///
/// **Corrected in sc-17807.** This doc priced the cache at 546 kB per DiT token and the clip at
/// 38 GiB — both 1.5× low, which made row E look like it ought to fit a 128 GiB host with room to
/// spare. It did not: the
/// recorded run measured a 63.32 GiB MLX peak and sc-15571 recorded it swapping a host's boot volume
/// full. The corrected figure is what
/// [`the_kv_per_token_figure_in_crate_prose_is_the_computed_one`] gates.
///
/// **Row E is UNMEASURABLE on current infrastructure — decided in sc-17324. Do not keep retrying it.**
///
/// The older claim here was that row E *SIGKILLs* a 128 GiB host at 832×480 (sc-15127: jetsam at
/// step 39/75). That is too strong: CI run 30787887176 (2026-08-03, `nax-macos`, 128 GiB) ran it to
/// completion at 832×480 — `S18CELL E 7 832x480 45 0 55.7356 ...`, a **63.32 GiB** MLX active peak,
/// ~17.5 min. But it fit that box by roughly 0.3 GiB, and sc-15571 records it driving a host into
/// enough swap to fill the boot volume even at 640×384. On `nax-macos-2` (~101 GiB, where the
/// `rw-krea` lane actually runs) it fits at neither bucket.
///
/// So row E is neither impossible nor fine — it is a row with no reproducible home, which is worth
/// naming once rather than rediscovering. `scripts/ci/s18_memory_preflight.py` refuses it by default
/// on both hosts and `krea_s18_rows` no longer includes it. **Nothing downstream depends on it:**
/// [`S18Sweep::verdict`] already declines to use row E as attribution evidence (out of regime,
/// different attention mask, n = 1, no variance estimate), so a sweep without it loses a reference,
/// not a conclusion.
///
/// Row **F** is a different case with a real answer: 46,800 tokens ≈ 49 GiB at 832×480 took
/// `nax-macos-2` down twice (runs 30948568453 and 31051413212, both inside row F seed 7), but at
/// 640×384 it needs ~34 GiB and fits — the bucket the recorded sc-15127 sweep already used for it.
/// The bounded rows run at both buckets, and **both buckets are recorded** — see
/// [`the_recorded_s18_sweep_is_what_the_docs_claim`].
///
/// ⚠️ **Row F is a whole-host risk on a *shared* machine, not only on a CI runner (sc-17807).** The
/// preflight's budget subtracts a fixed 60 GiB of non-MLX overhead, calibrated on a dedicated
/// runner. A dev Mac with several agent worktrees building concurrently is not that box, and its
/// real overhead is not knowable in advance. Running rows A/D/F unattended on one — after checking
/// `uptime`, which reports load and says nothing about memory — OOM-crashed it, at row A's
/// AR→VAE-decode transition where the footprint peaks (the AR peak plus the pinned 20 GiB decode
/// budget). Check free memory (`vm_stat`, free + inactive), require roughly 2× the predicted peak,
/// and re-check before every cell rather than once at the start.
///
/// **The sc-17807 KV-tier A/B therefore ran rows A and D only**, and that costs the comparison
/// nothing structural: [`S18Sweep::kv_tier_comparison`] compares whatever rows are present in both
/// arms, and [`S18Sweep::validate_window_dose_ladder`] returns early without F. What it does cost is
/// the widest *dose* — the tier's saving grows with the window, so row F is where quantizing helps
/// most, and it is exactly the row the A/B has no local measurement for. With sc-17894's exact-read
/// retention its q8 prediction is ~17.1 GiB of KV against 32.1 bf16, i.e. the row that killed
/// `nax-macos-2` becoming affordable on it. That stays a **prediction** until someone runs it on a
/// host that can hold it.
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
    let want_rows = std::env::var("KREA_S18_ROWS").unwrap_or_else(|_| "ABCDFEZ".to_string());
    let (latent_h, latent_w) = (h / 8, w / 8);
    // sc-17807 — the KV cache storage tier. Unset is the shipped bf16 cache, so an unchanged
    // dispatch measures exactly what it measured before. Setting it re-runs the same rows against a
    // quantized cache, which is a SEPARATE arm: it is recorded on its own and compared against the
    // bf16 baseline by `s18_kv_tier_ab_from_accumulated_cells`, never pooled with it.
    //
    // Parsed strictly, and NOT through `env_opt_usize`: that swallows a malformed value into `None`,
    // which here means "silently measure the bf16 arm while the operator believes they dispatched
    // the quantized one" — an hours-long run whose evidence is mislabelled and whose A/B compares a
    // tier against itself. Empty is the one accepted non-numeric form, because that is how a shell
    // spells "unset this for one invocation" (`env KREA_S18_KV_BITS= ...`).
    let kv_quant = parse_kv_tier_env().unwrap_or_else(|e| panic!("{e}"));
    let kv_tier = kv_tier_label(kv_quant);

    let base = KreaRealtimeConfig::krea_realtime_14b();
    let fpb = base.ar.num_frames_per_block; // 3 latent frames per AR chunk
    let shipped_window = base.ar.streaming_local_attn_frames() as i64; // 6 latent frames
    let wide_window = shipped_window * 2 + 3; // 15 latent frames — middle dose (10 rolls at 45f)
    let wider_bounded_window = shipped_window * 5; // 30 latent frames — high dose (5 rolls at 45f)
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
            row: 'F',
            label: "F wider bound (window 30, sink 0)",
            window: wider_bounded_window,
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
            cfg.ar.kv_cache_quant = kv_quant;
            cfg.ar.frame_seq_length =
                (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
            cfg.ar.seq_length = lat * cfg.ar.frame_seq_length;

            let chunks = lat.div_ceil(fpb);
            // Chunk `c` commits latent frames up to `fpb*(c+1)`; it evicts once that exceeds the
            // window. A global window (`local_attn_size < 0`) never evicts.
            let rolls = eviction_rolls(lat, fpb, r.window);

            println!(
                "=== {} seed {seed} | {lat} latent frames, {chunks} chunks, {rolls} evicting, \
                 baseline {row_pre_len} output frames, window {} tok, KV {kv_tier} ({} B/token)",
                r.label,
                cfg.ar.max_attention_size(),
                cfg.kv_bytes_per_token()
                    .expect("the KV tier must fit this backbone")
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
                 {:.4}\t{:.4}\t{:.4}\t{}\t{kv_tier}\t{}",
                r.row,
                worst_drift(&d),
                worst_trend(&d),
                worst_excursion(&d),
                worst_slope(&d),
                ar_peak,
                clip_mean,
                head_motion,
                tail_motion,
                worst_component(&d),
                ar_wall.as_millis()
            );
            results.push(S18Row {
                label: format!("{} s{seed}", r.label),
                latent_frames: lat,
                drift: worst_drift(&d),
                trend: worst_trend(&d),
                excursion: worst_excursion(&d),
                slope: worst_slope(&d),
                component: worst_component(&d),
                rolls,
                peak: ar_peak,
                clip_mean,
                head_motion,
                tail_motion,
                row: r.row,
                seed,
                ar_wall_ms: ar_wall.as_millis() as u64,
            });
        }
    }
    std::env::remove_var("WAN_VAE_BUDGET_GIB");

    println!("=== sc-15127 S18 summary ({w}x{h}) ==========================================");
    println!(
        "  row                                       rolls   drift   trend   excur  slope/100f  \
         peak GiB  clip%  tail motion  component"
    );
    for r in &results {
        println!(
            "  {:<41} {:>5}  {:>6.2}  {:>6.2}  {:>6.2}  {:>10.3}  {:>8.2}  {:>5.2}  {:>8.2}  {}",
            r.label,
            r.rolls,
            r.drift,
            r.trend,
            r.excursion,
            r.slope,
            gib(r.peak),
            r.clip_mean,
            r.tail_motion,
            r.component
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
    // Row Z's contribution: the within-regime, zero-eviction *rate*, in the only length-comparable
    // statistic there is. This is REPORTED, not gated — see `S18Sweep::rate_floor_clause` for why the
    // measured Z slope turned out not to bound row A's.
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
            "  within-regime rate: zero-eviction row Z {z:.3}/100f vs shipped row A {a:.3}/100f \
             (A/Z {:.2}x) — a ratio at or above 1 would make Z a floor; below 1 it is not one",
            a / z.max(1e-6)
        );
    }

    let sweep = S18Sweep {
        bucket: format!("{w}x{h}"),
        kv: kv_tier,
        cells: results.iter().map(S18Cell::from_measured).collect(),
    };
    println!("  {}", sweep.summary());
    // A verdict needs the shipped row AND the within-regime dose ladder it is attributed against.
    // Row A alone is *resolvable* — `validate_window_dose_ladder` returns early when D or F is
    // missing — so gating on `A` alone would let a one-row dispatch emit a verdict whose attribution
    // clause has no ladder behind it. Piecewise runs are the normal way to drive this sweep now
    // (sc-17655), so the partial branch is the common path, not the exotic one: measure the rows in
    // whatever pieces the runner can afford, then re-aggregate with
    // [`s18_verdict_from_accumulated_cells`].
    // Structural checks run on EVERY dispatch, complete or not — they are about whether the rows
    // measured are what they claim to be, not about whether there are enough of them. A partial
    // dispatch that mis-parameterises row E or Z, or runs too short a clip, must fail here rather
    // than at re-aggregation hours later.
    if let Err(e) = sweep.structural_checks() {
        panic!("{e}");
    }
    let complete = ['A', 'D', 'F'].iter().all(|row| want_rows.contains(*row));
    if complete {
        match sweep.verdict() {
            Ok(v) => println!("  VERDICT: {v}"),
            Err(e) => panic!("{e}"),
        }
    } else {
        println!(
            "  (partial sweep: rows `{want_rows}` are not the whole A/D/F dose ladder, so no \
             verdict is computed — re-aggregate the S18CELL lines with \
             `KREA_S18_CELLS=<file> cargo test -p mlx-gen-krea-realtime --test \
             generate_smoke s18_verdict_from_accumulated_cells -- --exact --ignored --nocapture`)"
        );
    }
}

/// **Re-aggregate a piecewise S18 sweep and apply the verdict rule to it.**
///
/// [`long_clip_coherence_under_the_bounded_window`] measures every (row, seed) it is asked for and
/// prints one `S18CELL` TSV line each, and its docs have always said those lines exist "so a long
/// sweep can be run in pieces and re-aggregated without holding a five-hour process open". Nothing
/// exposed that re-aggregation until sc-17655: the verdict was computed only inside the measuring
/// process, so the pieces could be measured but never resolved, and the only way to get a verdict
/// was the whole 4.3-hour sweep in one go — which on `nax-macos-2` head-of-line-blocks `rw-audio`,
/// `rw-llm` and `rw-chroma` for half a night (the sc-16981 failure, and the reason sc-17324's own
/// powered run got cancelled).
///
/// This is that entry point. It reads accumulated `S18CELL` lines, rebuilds the [`S18Sweep`], and
/// applies **the same [`S18Sweep::verdict`]** the live run would have — no second copy of the rule.
///
/// ```text
/// cat run-A.tsv run-B.tsv ... > all-cells.tsv
/// KREA_S18_CELLS=$PWD/all-cells.tsv cargo test -p mlx-gen-krea-realtime --test generate_smoke \
///   s18_verdict_from_accumulated_cells -- --exact --ignored --nocapture
/// ```
///
/// `-p` is not optional: two workspace crates carry a `generate_smoke` test target, and without it
/// the scail2 binary also runs, matches nothing under `--exact`, and exits 0 — the "0 tests, still
/// green" shape this file warns about elsewhere. Use an ABSOLUTE path for the file, too: `cargo
/// test` runs the binary with the CRATE root as its working directory, not the workspace root.
///
/// Input is the artifact the sweep job uploads verbatim — `VERDICT:` lines and blanks are ignored,
/// so `s18-cells.tsv` files concatenate without editing. `KREA_S18_BUCKET` selects the geometry when
/// the input mixes buckets; with one bucket present it is inferred. Duplicate (row, seed) cells are
/// NOT silently dropped: a re-measured cell reaches `validate_window_dose_ladder`, which rejects the
/// ladder rather than quietly averaging two runs of the same configuration.
///
/// `#[ignore]` because it is an operator entry point rather than a gate — it asserts nothing about
/// *which* verdict comes out, only that the rule can be applied to evidence that arrived in pieces.
/// The rule's own outcomes stay gated, without weights, by
/// [`the_s18_verdict_rule_distinguishes_its_outcomes`].
#[test]
#[ignore = "operator entry point: re-aggregates S18CELL evidence named by KREA_S18_CELLS"]
fn s18_verdict_from_accumulated_cells() {
    let path = std::env::var("KREA_S18_CELLS").expect(
        "set KREA_S18_CELLS to a file of accumulated S18CELL lines (concatenated `s18-cells.tsv` \
         artifacts are the expected input)",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the accumulated cells at `{path}`: {e}"));
    let want_bucket = std::env::var("KREA_S18_BUCKET").ok();
    let sweep = s18_sweep_from_accumulated(&raw, want_bucket.as_deref())
        .unwrap_or_else(|e| panic!("`{path}`: {e}"));

    let mut rows: Vec<char> = sweep.cells.iter().map(|c| c.row).collect();
    rows.sort_unstable();
    rows.dedup();
    println!(
        "re-aggregated {} cells at {} from `{path}` — rows {}",
        sweep.cells.len(),
        sweep.bucket,
        rows.iter().collect::<String>()
    );
    println!("  {}", sweep.summary());
    // The structural guards apply to re-assembled evidence exactly as they do to a live sweep — a
    // mis-parameterised row E or Z is no less wrong for having been measured in a separate job.
    if let Err(e) = sweep.structural_checks() {
        panic!("{e}");
    }
    match sweep.verdict() {
        Ok(v) => println!("  VERDICT: {v}"),
        Err(e) => panic!("{e}"),
    }
}

/// **The re-aggregation ingest must refuse duplicated evidence, not pool it.**
///
/// This is the gate for [`s18_sweep_from_accumulated`], which is otherwise reachable only through an
/// `#[ignore]`d entry point and would ship uncovered.
///
/// The duplicate case is the one that matters and it is not hypothetical: the documented workflow is
/// `cat piece-1.tsv piece-2.tsv > all.tsv`, every piece downloads under the same `s18-cells.tsv`
/// name, so including one twice is a slip rather than an act of malice. It must not degrade
/// *quietly*, because pooling a duplicate does not add noise — it removes it. `spread` is
/// `2*SD/sqrt(n)` over the sample SD, so the same three cells listed twice keep the mean and shrink
/// the interval by ~3.2x, which is the wrong direction: re-pasting evidence would buy confidence.
/// The assertion below pins exactly that, by showing the pooled sweep would have crossed from
/// unresolvable to resolved.
#[test]
fn accumulated_s18_evidence_rejects_duplicated_cells() {
    // A row A whose three seeds straddle the budget widely enough that 2*SEM exceeds the margin:
    // unresolvable at n = 3, and resolvable if the same cells are counted twice.
    let cell = |seed: u64, drift: f64| {
        format!("S18CELL\tA\t{seed}\t832x480\t45\t13\t{drift:.4}\t{drift:.4}\t0.0000\t10.0000\t18719499004\t2.0000\t14.0000\t14.0000\topp-B-Y")
    };
    // mean 20.0 against the 8.0 budget, so margin 12.0, and sample SD 12.0. At n = 3 the spread is
    // 2*12/sqrt(3) = 13.86 > 12 (unresolvable); listed twice, n = 6 and SD 10.73 give 8.76 < 12
    // (resolved). The duplicate does not add a single new measurement, and buys the answer.
    let once = [cell(7, 8.0), cell(11, 20.0), cell(23, 32.0)].join("\n");

    let sweep = s18_sweep_from_accumulated(&once, None).expect("three distinct cells parse");
    assert_eq!(sweep.cells.len(), 3, "one cell per (row, seed)");
    assert_eq!(sweep.bucket, "832x480", "bucket inferred from the evidence");
    let single = sweep.verdict();

    // Pooling the duplicate would CHANGE THE ANSWER — this is the whole reason the guard exists,
    // and asserting it keeps the guard from being weakened into a no-op later.
    let doubled_cells = {
        let evidence = parse_s18_evidence(&[once.clone(), once.clone()].join("\n")).expect("parse");
        evidence
            .iter()
            .map(|c| S18Cell {
                row: c.row,
                seed: c.seed,
                latent_frames: c.latent_frames,
                rolls: c.rolls,
                reported_drift: c.drift,
                trend: c.trend,
                excursion: c.excursion,
                slope: c.slope,
                peak_bytes: c.peak_bytes,
                clip_mean: c.clip_mean,
                head_motion: c.head_motion,
                tail_motion: c.tail_motion,
                ar_wall_ms: c.ar_wall_ms.unwrap_or(0),
                component: "",
            })
            .collect::<Vec<_>>()
    };
    let pooled = S18Sweep {
        bucket: "832x480".to_string(),
        kv: KV_TIER_BF16.to_string(),
        cells: doubled_cells,
    };
    // The claim is specifically that the duplicate defeats the POWER gate — not that it produces a
    // clean verdict (this fixture has no sink rows, so the pooled sweep stops on that instead).
    // Crossing from "cannot resolve which side of the budget" to "resolved" is the damage.
    let single_err = single.expect_err("n = 3 must be underpowered for this fixture");
    assert!(
        single_err.contains("UNDERPOWERED"),
        "fixture must start underpowered or the assertion below proves nothing: {single_err}"
    );
    let pooled_verdict = pooled.verdict();
    let still_underpowered = pooled_verdict
        .as_ref()
        .err()
        .is_some_and(|e| e.contains("UNDERPOWERED"));
    assert!(
        !still_underpowered,
        "fixture must be one where pooling DEFEATS the power gate, or the assertion below proves \
         nothing: {pooled_verdict:?}"
    );

    // ...and the ingest must not let that happen.
    let twice = [once.clone(), once.clone()].join("\n");
    let err = s18_sweep_from_accumulated(&twice, None)
        .expect_err("the same file concatenated twice must be rejected");
    assert!(
        err.contains("duplicate (row, seed)") && err.contains("A/seed 7"),
        "the rejection must name the offending cells: {err}"
    );

    // Duplicate detection is per bucket: the same (row, seed) at two geometries is two measurements.
    let cross_bucket = format!(
        "{}\n{}",
        cell(7, 2.0),
        cell(7, 2.0).replace("832x480", "640x384")
    );
    assert!(
        s18_sweep_from_accumulated(&cross_bucket, Some("832x480")).is_ok(),
        "the same seed at another bucket is not a duplicate"
    );
    let mixed = s18_sweep_from_accumulated(&cross_bucket, None)
        .expect_err("mixed buckets with no selection must be refused");
    assert!(mixed.contains("mixes buckets"), "got: {mixed}");

    // Artifact noise must survive: `VERDICT:` lines, blanks and indented cells.
    let noisy = format!(
        "  VERDICT: drift is real (whatever)\n\n  {}\n",
        once.replace('\n', "\n  ")
    );
    assert_eq!(
        s18_sweep_from_accumulated(&noisy, None)
            .expect("artifact noise is filtered, indented cells are kept")
            .cells
            .len(),
        3,
        "indented S18CELL lines must not be silently dropped"
    );
    assert!(
        s18_sweep_from_accumulated("  VERDICT: nothing here\n", None)
            .expect_err("a file with no cells is not evidence")
            .contains("no S18CELL lines"),
    );
}

/// **Structural checks must not depend on the sweep being verdict-complete.**
///
/// sc-17655 made partial dispatches the normal way to drive this sweep, and these three checks used
/// to live inside [`S18Sweep::verdict`] — reachable only when rows A/D/F were all present. A piece
/// that mis-parameterises row E or Z has to fail on its own.
#[test]
fn s18_structural_checks_run_without_a_complete_ladder() {
    let cell = |row: char, seed: u64, rolls: usize, latent: usize| S18Cell {
        row,
        seed,
        latent_frames: latent,
        rolls,
        reported_drift: 20.0,
        trend: 20.0,
        excursion: 0.0,
        slope: 10.0,
        peak_bytes: 1,
        clip_mean: 2.0,
        head_motion: 14.0,
        tail_motion: 14.0,
        ar_wall_ms: 0,
        component: "",
    };
    let sweep = |cells: Vec<S18Cell>| S18Sweep {
        bucket: "832x480".to_string(),
        kv: KV_TIER_BF16.to_string(),
        cells,
    };

    // A row E that evicted is not a global reference — caught with no A, D or F in sight.
    let e_only = sweep(vec![cell('E', 7, 3, 45)]);
    assert!(
        e_only.validate_window_dose_ladder().is_ok(),
        "no ladder here"
    );
    assert!(e_only
        .structural_checks()
        .expect_err("a rolled row E must be rejected")
        .contains("not a reference"),);
    // A row Z that evicted is not a zero-eviction floor.
    assert!(sweep(vec![cell('Z', 7, 1, 6)])
        .structural_checks()
        .expect_err("an evicting row Z must be rejected")
        .contains("zero-eviction floor"));
    // Too short a clip is rejected on a row-A-only piece.
    assert!(sweep(vec![cell('A', 7, 2, 9)])
        .structural_checks()
        .expect_err("a short clip must be rejected")
        .contains("not a long clip"));
    // Well-formed pieces pass.
    assert!(sweep(vec![cell('A', 7, 13, 45), cell('Z', 11, 0, 6)])
        .structural_checks()
        .is_ok());
}

/// Rebuild an [`S18Sweep`] from accumulated `S18CELL` lines.
///
/// Split out of [`s18_verdict_from_accumulated_cells`] so the ingest is testable with no weights,
/// no environment and no files — see [`accumulated_s18_evidence_rejects_duplicated_cells`]. The
/// entry point itself is `#[ignore]`d, so without this split none of the parsing, bucket selection
/// or duplicate rejection below would be covered by anything.
fn s18_sweep_from_accumulated(src: &str, want_bucket: Option<&str>) -> Result<S18Sweep, String> {
    // The uploaded artifact interleaves `  VERDICT:` lines with the cells, and concatenating
    // several leaves blank lines — `parse_s18_evidence` rejects both, so filter to cells first.
    // `trim_start` because a hand-assembled file is the expected input and indented cells are the
    // obvious way to lose one silently.
    let cells_only: String = src
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("S18CELL"))
        .collect::<Vec<_>>()
        .join("\n");
    if cells_only.is_empty() {
        return Err("contains no S18CELL lines — it is not S18 sweep evidence".to_string());
    }
    let evidence = parse_s18_evidence(&cells_only)?;

    let mut buckets: Vec<String> = evidence.iter().map(|c| c.bucket.clone()).collect();
    buckets.sort();
    buckets.dedup();
    let bucket = match want_bucket {
        Some(want) => {
            if !buckets.iter().any(|b| b == want) {
                return Err(format!(
                    "bucket `{want}` is not present (buckets: {buckets:?})"
                ));
            }
            want.to_string()
        }
        None => {
            if buckets.len() != 1 {
                return Err(format!(
                    "mixes buckets {buckets:?} — set KREA_S18_BUCKET to choose one, because a \
                     verdict is per geometry and pooling them would compare different clips"
                ));
            }
            buckets[0].clone()
        }
    };

    let selected: Vec<&S18Evidence> = evidence.iter().filter(|c| c.bucket == bucket).collect();

    // sc-17807 — the KV storage tier is part of what a cell measured, so a file that mixes tiers is
    // refused rather than pooled. Pooling would look like "more seeds" while actually comparing two
    // different models: quantizing the cache perturbs the very coherence this sweep measures, which
    // is why the knob ships opt-in and why its arm is recorded separately from the bf16 baseline.
    let mut tiers: Vec<String> = selected
        .iter()
        .map(|c| c.kv.clone().unwrap_or_else(|| KV_TIER_BF16.to_string()))
        .collect();
    tiers.sort();
    tiers.dedup();
    let kv = match tiers.as_slice() {
        [only] => only.clone(),
        _ => {
            return Err(format!(
                "mixes KV cache tiers {tiers:?} at {bucket} — a verdict is per tier, and pooling \
                 them would compare two different models while looking like added seeds. Split the \
                 evidence and re-aggregate each tier on its own."
            ))
        }
    };

    // Duplicate (row, seed) cells are REJECTED here rather than left to the verdict rule.
    // `validate_window_dose_ladder` does reject them — but only once rows A, D and F are all
    // present; it returns early otherwise, and a piecewise sweep routinely holds one or two rows,
    // which is precisely when this function is used. Pooling is not a harmless double count:
    // `spread` is 2*SD/sqrt(n) over the SAMPLE SD, so including the same evidence twice shrinks
    // the interval by ~3.2x at n = 3 and can flip an UNDERPOWERED sweep into a confident verdict.
    // Concatenating one `s18-cells.tsv` twice is a single `cat` away, so the ingest owns this.
    let mut seen: Vec<(char, u64)> = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();
    for cell in &selected {
        let key = (cell.row, cell.seed);
        if seen.contains(&key) {
            duplicates.push(format!("{}/seed {}", cell.row, cell.seed));
        } else {
            seen.push(key);
        }
    }
    if !duplicates.is_empty() {
        duplicates.sort();
        duplicates.dedup();
        return Err(format!(
            "duplicate (row, seed) cells at {bucket}: {}. The same configuration appears more than \
             once — most likely a piece was included twice. Pooling duplicates SHRINKS the \
             between-seed spread and manufactures confidence, so the input must be de-duplicated \
             rather than accepted.",
            duplicates.join(", ")
        ));
    }

    let cells: Vec<S18Cell> = selected
        .iter()
        .map(|c| S18Cell {
            row: c.row,
            seed: c.seed,
            latent_frames: c.latent_frames,
            rolls: c.rolls,
            reported_drift: c.drift,
            trend: c.trend,
            excursion: c.excursion,
            slope: c.slope,
            peak_bytes: c.peak_bytes,
            clip_mean: c.clip_mean,
            head_motion: c.head_motion,
            tail_motion: c.tail_motion,
            // `None` for a pre-sc-17807 line, which has no time in it. Reported as "not recorded"
            // rather than as a zero that would read like an instantaneous run.
            ar_wall_ms: c.ar_wall_ms.unwrap_or(0),
            // `S18Cell` holds `&'static str` so the recorded tables can be `const`. Rather than
            // leak the parsed string, bind it to the matching [`DESCRIPTOR_NAMES`] entry; anything
            // else (including the historical rows that predate the field) becomes "". Nothing in
            // `verdict`/`summary` reads this — only the recorded-evidence tests do — so an unknown
            // descriptor is worth neither a leak nor a hard error here.
            component: c
                .component
                .as_deref()
                .and_then(|name| DESCRIPTOR_NAMES.iter().find(|known| **known == name))
                .copied()
                .unwrap_or(""),
        })
        .collect();

    Ok(S18Sweep { bucket, kv, cells })
}

/// One measured (row, seed) cell of the S18 sweep.
#[derive(Clone, Copy, Debug)]
struct S18Cell {
    /// `A` shipped · `B` sink 1 · `C` sink 3 · `D` wide15 · `F` wide30 · `E` global reference ·
    /// `Z` zero-roll.
    row: char,
    seed: u64,
    latent_frames: usize,
    rolls: usize,
    /// The `S18CELL` verdict value as printed by the original run.
    reported_drift: f64,
    /// Worst absolute one-way component trend, 0..255.
    trend: f64,
    /// Worst absolute z-gated component excursion, 0..255.
    excursion: f64,
    /// Worst absolute component slope over the post-roll segment, per 100 output frames. Recorded
    /// because it is the **only** statistic comparable across rows of different clip length, which is
    /// the entire point of row Z — see [`S18Sweep::rate_floor_clause`]. Same component as
    /// [`trend`](Self::trend) (they differ by the constant post-segment length), so the segment length
    /// is recoverable as `100 * trend / slope`.
    slope: f64,
    /// MLX active peak bytes for this cell's AR loop.
    peak_bytes: usize,
    /// Mean percentage of pixels at or above 250 in the brightest channel.
    clip_mean: f64,
    /// Mean absolute RGB frame-to-frame delta over the first third, in 0..255 units.
    head_motion: f64,
    /// Mean absolute RGB frame-to-frame delta over the last third of the clip, in 0..255 units.
    ///
    /// This is the sweep's freeze check. It is retained in the recorded cells so the crate prose's
    /// count and rounded range are derived from the raw `S18CELL` evidence rather than hand-copied.
    tail_motion: f64,
    /// AR-loop wall time in milliseconds. Recorded because sc-17807's feasibility question has a
    /// throughput half — dequantizing the read window per layer could have spent the memory win back
    /// in bandwidth, and a cell with no time in it cannot say either way. `0` for the historical
    /// cells, which predate the field.
    ar_wall_ms: u64,
    /// Which [`DESCRIPTOR_NAMES`] component actually produced [`drift`](Self::drift) for this cell.
    ///
    /// Recorded so the reader can check *which channel* scored a row rather than taking the earlier
    /// "saturation" framing of the mode on trust. It is not the same thing as the corroborating
    /// `report_artifacts` saturation statistic, and this field is what makes that checkable.
    component: &'static str,
}

impl S18Cell {
    /// Retain every verdict and freeze-evidence field emitted by the live real-weight sweep.
    fn from_measured(row: &S18Row) -> Self {
        Self {
            row: row.row,
            seed: row.seed,
            latent_frames: row.latent_frames,
            rolls: row.rolls,
            reported_drift: row.drift,
            trend: row.trend,
            excursion: row.excursion,
            slope: row.slope,
            peak_bytes: row.peak,
            clip_mean: row.clip_mean,
            head_motion: row.head_motion,
            tail_motion: row.tail_motion,
            ar_wall_ms: row.ar_wall_ms,
            component: row.component,
        }
    }

    /// The gated statistic: trend AND excursion must be inside budget, i.e. their max must be.
    fn drift(&self) -> f64 {
        self.trend.abs().max(self.excursion.abs())
    }

    /// Output frames in this cell's post-roll segment, recovered from the trend/slope pair.
    fn post_len(&self) -> Option<f64> {
        (self.slope.abs() > 1e-9).then(|| 100.0 * self.trend.abs() / self.slope.abs())
    }
}

#[test]
fn an_s18_cell_retains_tail_motion_from_the_live_sweep() {
    let measured = S18Row {
        label: "test".into(),
        latent_frames: 45,
        drift: 1.0,
        trend: 2.0,
        excursion: 3.0,
        slope: 4.0,
        component: "luma-mean",
        rolls: 5,
        peak: 6,
        clip_mean: 7.0,
        head_motion: 8.0,
        tail_motion: 12.3456,
        ar_wall_ms: 0,
        row: 'A',
        seed: 9,
    };
    let cell = S18Cell::from_measured(&measured);
    assert_eq!(
        cell.tail_motion, measured.tail_motion,
        "the live S18 result-to-record conversion must not discard the freeze measurement"
    );
}

#[test]
fn the_s18_bounded_dose_ladder_spans_thirteen_ten_and_five_rolls() {
    let (latent_frames, frames_per_block) = (45, 3);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, 6), 13);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, 15), 10);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, 30), 5);
    assert_eq!(eviction_rolls(latent_frames, frames_per_block, -1), 0);
}

/// Outcome of the within-regime A/D/F window dose-response. The decision statistic is the mean,
/// across matched seeds, of the OLS slope of drift against roll count at windows 6/15/30.
///
/// **Symmetric by construction.** An earlier form fired whenever `D + combined_spread >= A`, i.e.
/// whenever the wider window failed to *beat* the shipped one, and then asserted an unqualified
/// falsification ("eviction is not the mechanism"). That is a non-inferiority test wearing a
/// falsification's clothes: noisier data makes it *easier* to satisfy, which is the same inversion the
/// max−min → 2·SEM change fixed elsewhere in this file. It is also inconsistent with the sink
/// comparison, which correctly applies a *resolvability* standard to exactly this shape of question.
///
/// So: neither direction may be asserted unless the fitted slope clears its between-seed 2·SEM, and
/// when it does not, the outcome names the effect size the enlarged roll span **can** exclude.
enum WindowAttribution {
    /// One or more of A/D/F was not measured — the three-dose response was not run.
    Unmeasured,
    /// The fitted slope is inside its 2·SEM. Nothing may be concluded in either direction.
    Unresolved {
        f: f64,
        f_rolls: usize,
        slope: f64,
        unc: f64,
        n: usize,
    },
    /// The fitted slope is negative beyond the predeclared 2·SEM heuristic. This rejects a positive
    /// linear/monotone roll-count contribution over the measured 5–13-roll range; it does not exclude
    /// every possible cache-window mechanism.
    NoPositiveLinearDoseResponse {
        f: f64,
        f_rolls: usize,
        slope: f64,
        unc: f64,
        n: usize,
    },
    /// The fitted slope is positive beyond the predeclared 2·SEM heuristic: more evictions predict
    /// more drift over the measured bounded-window dose range.
    PositiveLinearDoseResponse {
        f: f64,
        f_rolls: usize,
        slope: f64,
        unc: f64,
        n: usize,
    },
}

/// A measured S18 sweep at one geometry bucket, and the decision rule applied to it. Split out of the
/// real-weight driver so the **rule** is gated in CI rather than only exercised on the gated GPU run.
#[derive(Debug)]
struct S18Sweep {
    bucket: String,
    /// The **KV cache storage tier** this sweep ran at (sc-17807) — `"bf16"` for the shipped cache,
    /// `"q8/g64"` and friends for a quantized one. Carried on the sweep rather than on each cell
    /// because one dispatch runs one tier, and pooling two tiers would compare different models
    /// while looking like added seeds. `s18_sweep_from_accumulated` refuses a mixed file for exactly
    /// that reason.
    kv: String,
    cells: Vec<S18Cell>,
}

/// The label a bf16-KV sweep carries — the shipped default, and what every cell recorded before
/// sc-17807 implicitly is.
const KV_TIER_BF16: &str = "bf16";

/// Smallest drift difference the sc-17807 KV-tier A/B will call a regression, in 0..255 units.
///
/// Statistical resolvability is necessary but not sufficient here. The paired statistic
/// ([`S18Sweep::paired_deltas`]) is powerful precisely because it cancels content variance — which
/// means three paired deltas can land with a sample SD near zero, and then *any* consistent shift
/// clears its own 2·SEM. A 0.1/255 "regression" resolved that way is an artifact of `n = 3`, not a
/// finding, so the rule also requires the effect to be big enough to act on.
///
/// One eighth of [`DRIFT_BUDGET`] — one unit on the 0..255 scale the descriptors live on, i.e. the
/// coarsest resolution at which a difference in these statistics means anything at all.
///
/// **Declared with the tension visible**, because a floor chosen after seeing the data is worthless:
/// the sc-17807 calibration cell (row A, seed 7, 640×384) measured q8 at +0.89/255 against bf16, so
/// this floor sits *above* the one observation that existed when it was written. A resolvable
/// difference under it is REPORTED as "resolvable but IMMATERIAL" with its magnitude, never
/// silently folded into agreement.
const KV_TIER_MATERIAL_FLOOR: f64 = DRIFT_BUDGET / 8.0;

/// The `S18CELL` / summary label for a KV storage tier: `bf16`, or `q<bits>/g<group>`.
fn kv_tier_label(quant: Option<KvCacheQuant>) -> String {
    match quant {
        None => KV_TIER_BF16.to_string(),
        Some(q) => format!("q{}/g{}", q.bits, q.group_size),
    }
}

/// Read the sc-17807 KV storage tier from `KREA_S18_KV_BITS` / `KREA_S18_KV_GROUP_SIZE`.
///
/// **Strict on purpose.** Unset (or empty — how a shell spells "unset for this invocation") is the
/// shipped bf16 cache; anything else must parse to a tier MLX can express. The tempting
/// `.parse().ok()` form swallows a typo into `None`, which here does not mean "no tier" but "run the
/// bf16 arm for two hours while the log, the recorded cells and the A/B all say q8" — a mislabelled
/// arm compared against itself. Split out so the parse is gated without weights.
fn parse_kv_tier_env() -> std::result::Result<Option<KvCacheQuant>, String> {
    fn field(name: &str, default: i32) -> std::result::Result<Option<i32>, String> {
        match std::env::var(name) {
            Err(_) => Ok(None),
            Ok(raw) if raw.trim().is_empty() => Ok(None),
            Ok(raw) => raw
                .trim()
                .parse::<i32>()
                .map(Some)
                .map_err(|_| format!("{name}=`{raw}` is not an integer (default {default})")),
        }
    }
    let group_size = field("KREA_S18_KV_GROUP_SIZE", 64)?.unwrap_or(64);
    let Some(bits) = field("KREA_S18_KV_BITS", 0)? else {
        // A group size with no width selects nothing and is almost certainly a half-set dispatch.
        if std::env::var("KREA_S18_KV_GROUP_SIZE").is_ok_and(|v| !v.trim().is_empty()) {
            return Err(
                "KREA_S18_KV_GROUP_SIZE is set but KREA_S18_KV_BITS is not — that selects the bf16 \
                 cache, which is not what a half-set quantization dispatch means"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    let quant = KvCacheQuant { bits, group_size };
    quant.validate().map_err(|e| format!("{e}"))?;
    Ok(Some(quant))
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

    /// The between-seed uncertainty heuristic on a row's **mean**: twice the standard error.
    /// This is the sweep's predeclared resolvability rule, not a Student-t 95% confidence interval
    /// (with three seeds, that would need a materially larger multiplier). `None` if the row has fewer
    /// than two seeds — in which case there is **no variance estimate at all**, which the verdict must
    /// say out loud.
    ///
    /// Deliberately not the max−min range: that *grows* with the number of seeds, so a rule gated on
    /// it would get harder to satisfy the more evidence you collected. This shrinks as 1/√n, which is
    /// what makes "add seeds until the comparison resolves" a real option.
    fn spread(&self, row: char) -> Option<f64> {
        let v = self.of(row);
        (v.len() >= 2).then(|| 2.0 * std_f64(&v) / (v.len() as f64).sqrt())
    }

    /// Mean of a row's worst per-100-frame slope — the length-comparable rate statistic.
    fn mean_slope(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| c.slope.abs())
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// Mean post-roll segment length of a row, in output frames.
    fn mean_post_len(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .filter_map(|c| c.post_len())
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// The **within-regime rate** comparison the [`DRIFT_BUDGET`] doc used to assert and never compute:
    /// row Z is the same attention regime with *zero* evictions, so if a per-frame rate floor exists
    /// for this content, Z is where it lives, and [`S18Cell::slope`] is the only statistic comparable
    /// across Z's short clip and A's long one.
    ///
    /// It is **reported, not gated**, because on the recorded data it does not separate: see the
    /// returned text. A short-segment OLS slope is dominated by ordinary frame-to-frame motion, so Z's
    /// slope is *larger* than A's, and extrapolating it over A's post segment over-predicts A's own
    /// measured drift. Row Z cannot be lengthened either — the shipped 6-latent-frame window evicts as
    /// soon as the clip passes 6 latent frames, so a longer zero-eviction row does not exist at the
    /// shipped window. The consequence is stated rather than hidden: this sweep has **no measured
    /// same-content rate floor**, and [`DRIFT_BUDGET`] is bracketed by the synthetic motion/jitter and
    /// failure-shape controls alone.
    fn rate_floor_clause(&self) -> String {
        let (Some(a_slope), Some(z_slope)) = (self.mean_slope('A'), self.mean_slope('Z')) else {
            return String::new();
        };
        let a_post = self.mean_post_len('A').unwrap_or(0.0);
        let z_post = self.mean_post_len('Z').unwrap_or(0.0);
        let a_trend = self.mean_slope('A').unwrap_or(0.0) * a_post / 100.0;
        let predicted = z_slope * a_post / 100.0;
        if z_slope < a_slope {
            format!(
                " The within-regime zero-eviction row Z does NOT establish a rate floor here: over \
                 its {z_post:.0}-output-frame post segment it runs at {z_slope:.2}/100f against the \
                 shipped row's {a_slope:.2}/100f over {a_post:.0} — Z is lower, so the shipped row's \
                 rate does exceed the zero-eviction rate."
            )
        } else {
            format!(
                " The within-regime zero-eviction row Z does NOT establish a rate floor here, and \
                 the sweep says so rather than omitting it: over its {z_post:.0}-output-frame post \
                 segment Z runs at {z_slope:.2}/100f, which is HIGHER than the shipped row's \
                 {a_slope:.2}/100f over {a_post:.0} frames. Extrapolated across the long clip that \
                 rate predicts {predicted:.1}/255 of drift — {:.1}x what row A actually measures \
                 ({a_trend:.1}/255) — so the extrapolation over-predicts and a {z_post:.0}-frame OLS \
                 slope (dominated by ordinary frame-to-frame motion) is not extrapolable. Row Z \
                 cannot be lengthened: the shipped window evicts as soon as the clip passes 6 latent \
                 frames, so no longer zero-eviction row exists at the shipped window. This sweep \
                 therefore has NO measured same-content rate floor, and the budget is bracketed by \
                 the synthetic motion/jitter and failure-shape controls alone.",
                predicted / a_trend.max(1e-6)
            )
        }
    }

    /// Per-seed OLS slopes of drift against roll count over the A/D/F ladder. Matching by seed prevents
    /// ordinary seed-to-seed content variation from masquerading as a dose response.
    fn window_dose_slopes(&self) -> Vec<f64> {
        let mut seeds: Vec<u64> = self
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.seed)
            .collect();
        seeds.sort_unstable();
        seeds.dedup();
        seeds
            .into_iter()
            .filter_map(|seed| {
                let points: Vec<(f64, f64)> = ['A', 'D', 'F']
                    .iter()
                    .filter_map(|&row| {
                        self.cells
                            .iter()
                            .find(|c| c.row == row && c.seed == seed)
                            .map(|c| (c.rolls as f64, c.drift()))
                    })
                    .collect();
                if points.len() != 3 {
                    return None;
                }
                let x_mean = points.iter().map(|p| p.0).sum::<f64>() / 3.0;
                let y_mean = points.iter().map(|p| p.1).sum::<f64>() / 3.0;
                let denom = points.iter().map(|p| (p.0 - x_mean).powi(2)).sum::<f64>();
                (denom > 0.0).then(|| {
                    points
                        .iter()
                        .map(|p| (p.0 - x_mean) * (p.1 - y_mean))
                        .sum::<f64>()
                        / denom
                })
            })
            .collect()
    }

    /// The slope is a paired-seed statistic over the exact three-dose experiment: 45 latent frames
    /// with A/D/F at 13/10/5 rolls. Refuse malformed evidence instead of silently accepting duplicate
    /// seeds, a shorter F clip, a different window width, a non-finite statistic, or fitting duplicate
    /// roll counts.
    fn validate_window_dose_ladder(&self) -> std::result::Result<(), String> {
        if ['A', 'D', 'F'].iter().any(|&row| self.of(row).is_empty()) {
            return Ok(());
        }
        let seeds = |row| {
            let mut values: Vec<u64> = self
                .cells
                .iter()
                .filter(|c| c.row == row)
                .map(|c| c.seed)
                .collect();
            values.sort_unstable();
            values
        };
        let (a_seeds, d_seeds, f_seeds) = (seeds('A'), seeds('D'), seeds('F'));
        let has_duplicate = |values: &[u64]| values.windows(2).any(|pair| pair[0] == pair[1]);
        if has_duplicate(&a_seeds) || has_duplicate(&d_seeds) || has_duplicate(&f_seeds) {
            return Err(format!(
                "the A/D/F dose ladder contains duplicate seed cells: \
                 A={a_seeds:?}, D={d_seeds:?}, F={f_seeds:?}"
            ));
        }
        if a_seeds != d_seeds || a_seeds != f_seeds {
            return Err(format!(
                "the A/D/F dose ladder does not contain the same matched seeds: \
                 A={a_seeds:?}, D={d_seeds:?}, F={f_seeds:?}"
            ));
        }
        for cell in self
            .cells
            .iter()
            .filter(|c| matches!(c.row, 'A' | 'D' | 'F'))
        {
            let expected_rolls = match cell.row {
                'A' => 13,
                'D' => 10,
                'F' => 5,
                _ => unreachable!(),
            };
            if cell.latent_frames != 45 || cell.rolls != expected_rolls {
                return Err(format!(
                    "the A/D/F dose ladder must be the exact 45-latent-frame, 13/10/5-roll \
                     experiment: row {} seed {} has {} latent frames and {} rolls",
                    cell.row, cell.seed, cell.latent_frames, cell.rolls
                ));
            }
            if !cell.reported_drift.is_finite()
                || !cell.trend.is_finite()
                || !cell.excursion.is_finite()
            {
                return Err(format!(
                    "the A/D/F dose ladder contains a non-finite drift statistic: row {} seed {}",
                    cell.row, cell.seed
                ));
            }
        }
        let slopes = self.window_dose_slopes();
        if slopes.len() != a_seeds.len() || slopes.iter().any(|slope| !slope.is_finite()) {
            return Err(format!(
                "the A/D/F dose ladder did not produce one finite paired slope per seed: \
                 {} slopes for {} matched seeds",
                slopes.len(),
                a_seeds.len()
            ));
        }
        Ok(())
    }

    /// Classify the within-regime window dose-response. See [`WindowAttribution`] for why this is
    /// two-sided.
    fn window_attribution(&self) -> WindowAttribution {
        let (Some(_d), Some(f)) = (self.mean('D'), self.mean('F')) else {
            return WindowAttribution::Unmeasured;
        };
        let f_rolls = self
            .cells
            .iter()
            .find(|c| c.row == 'F')
            .map(|c| c.rolls)
            .unwrap_or(0);
        let slopes = self.window_dose_slopes();
        let n = slopes.len();
        let slope = if slopes.is_empty() {
            0.0
        } else {
            mean_f64(&slopes)
        };
        let unc = if n >= 2 {
            2.0 * std_f64(&slopes) / (n as f64).sqrt()
        } else {
            f64::INFINITY
        };
        if slope.abs() <= unc {
            WindowAttribution::Unresolved {
                f,
                f_rolls,
                slope,
                unc,
                n,
            }
        } else if slope < 0.0 {
            WindowAttribution::NoPositiveLinearDoseResponse {
                f,
                f_rolls,
                slope,
                unc,
                n,
            }
        } else {
            WindowAttribution::PositiveLinearDoseResponse {
                f,
                f_rolls,
                slope,
                unc,
                n,
            }
        }
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
            "{} KV {} — A shipped {} | B sink1 {} | C sink3 {} | D wide15 {} | F wide30 {} | E global-ref {} | budget \
             {DRIFT_BUDGET:.2}/255 (absolute)",
            self.bucket,
            self.kv,
            f('A'),
            f('B'),
            f('C'),
            f('D'),
            f('F'),
            f('E'),
        )
    }

    /// Per-seed `arm − baseline` drift deltas for a row, matched by seed — `None` when the two arms
    /// share fewer than two seeds on this row.
    ///
    /// Pairing is what makes this A/B powerful enough to be worth running. Both arms drive the same
    /// seeds, so the same init noise produces the same content; the unpaired difference of means
    /// carries the full seed-to-seed content variance, which at the recorded 640×384 scatter cannot
    /// resolve anything smaller than ~6.7/255 on row A — 84% of the whole [`DRIFT_BUDGET`]. The
    /// paired form cancels it, and it is the same matched-seed reasoning
    /// [`S18Sweep::window_dose_slopes`] already applies to the dose ladder.
    fn paired_deltas(&self, baseline: &S18Sweep, row: char) -> Option<Vec<f64>> {
        let mut deltas = Vec::new();
        for cell in self.cells.iter().filter(|c| c.row == row) {
            if let Some(base) = baseline
                .cells
                .iter()
                .find(|b| b.row == row && b.seed == cell.seed)
            {
                deltas.push(cell.drift() - base.drift());
            }
        }
        (deltas.len() >= 2).then_some(deltas)
    }

    /// Mean AR-loop wall time (ms) for a row, or `None` when no cell recorded one (the pre-sc-17807
    /// cells store `0`, which would read as an instantaneous run rather than as absent).
    fn mean_wall_ms(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row && c.ar_wall_ms > 0)
            .map(|c| c.ar_wall_ms as f64)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// Mean MLX active peak (bytes) for a row, or `None` if it was not measured.
    fn mean_peak(&self, row: char) -> Option<f64> {
        let v: Vec<f64> = self
            .cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| c.peak_bytes as f64)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// **The sc-17807 A/B: is a quantized KV cache worse than the bf16 baseline?**
    ///
    /// `self` is the quantized arm, `baseline` the bf16 one at the same geometry. Three things are
    /// gated, and all three can fail:
    ///
    ///   1. **Drift**, per row, on the same gated statistic [`S18Sweep::verdict`] uses. The primary
    ///      form is the **matched-seed paired delta** (`arm(seed) − base(seed)`), following this
    ///      file's own dose-ladder precedent: both arms run the same seeds, so the same init noise
    ///      produces the same content, and pairing cancels the content variance that otherwise
    ///      dominates. It matters — against the recorded 640×384 scatter the unpaired form cannot
    ///      resolve anything smaller than ~6.7/255 on row A, i.e. 84% of the whole
    ///      [`DRIFT_BUDGET`]. The unpaired difference-of-means (`sqrt(a² + b²)` on the two arms'
    ///      2·SEM) is the fallback when the seeds do not match.
    ///   2. **Memory**, per row. A quantized cache whose measured peak is not *below* the baseline's
    ///      has bought nothing, and the entire story is the saving — so this is an `Err`, not a
    ///      footnote. There is a real mechanism for it to fail (a dequantized read window that stops
    ///      being freed per layer would make the packed path *worse* at peak than the dense one).
    ///   3. **Rankability.** If no row resolves in either direction, the answer is "this A/B did not
    ///      resolve", not "not resolvably worse" — the headline is what a reader takes away, and it
    ///      must not assert a conclusion the seeds cannot support. The minimum detectable effect is
    ///      reported so an underpowered result reads as underpowered.
    ///
    /// **What this rule is NOT.** It is not a drift pass/fail: [`S18Sweep::verdict`] is applied to
    /// the arm separately (that is item 3's "using the existing verdict rule"), but at these
    /// geometries row A already scores far outside [`DRIFT_BUDGET`] in *both* arms, so `verdict`
    /// takes its drift-attribution branch and returns `Ok` for either. It answers "does the recorded
    /// sweep still say what the docs say", not "is q8 acceptable". This comparison is what answers
    /// the second question, and only relative to bf16.
    fn kv_tier_comparison(&self, baseline: &S18Sweep) -> std::result::Result<String, String> {
        if self.bucket != baseline.bucket {
            return Err(format!(
                "cannot compare KV tiers across geometries ({} vs {}) — the clips differ",
                self.bucket, baseline.bucket
            ));
        }
        if self.kv == baseline.kv {
            return Err(format!(
                "both arms ran the same KV tier ({}) — that is a repeat, not an A/B",
                self.kv
            ));
        }
        if baseline.kv != KV_TIER_BF16 {
            return Err(format!(
                "the baseline arm is `{}`, not the shipped `{KV_TIER_BF16}` cache — the comparison \
                 is defined against what ships",
                baseline.kv
            ));
        }

        let mut lines: Vec<String> = Vec::new();
        let mut worse: Vec<String> = Vec::new();
        let (mut compared, mut rankable) = (0usize, 0usize);
        for row in ['A', 'B', 'C', 'D', 'F'] {
            let (Some(arm), Some(base)) = (self.mean(row), baseline.mean(row)) else {
                continue;
            };
            compared += 1;
            // Matched-seed paired deltas when the two arms ran the same seeds — the powerful form,
            // and the one this file already uses for the dose ladder. `None` when the seeds do not
            // line up, in which case the unpaired difference of means stands in.
            let paired = self.paired_deltas(baseline, row);
            let (delta, unc, form) = match &paired {
                Some(d) if d.len() >= 2 => (
                    mean_f64(d),
                    Some(2.0 * std_f64(d) / (d.len() as f64).sqrt()),
                    format!("paired, n={}", d.len()),
                ),
                _ => (
                    arm - base,
                    match (self.spread(row), baseline.spread(row)) {
                        (Some(a), Some(b)) => Some((a * a + b * b).sqrt()),
                        _ => None,
                    },
                    "unpaired".to_string(),
                ),
            };
            let verdict = match unc {
                None => "UNRANKABLE (an arm has one seed — no variance estimate, so this row is \
                     ABSENT from the comparison rather than agreeing with it)"
                    .to_string(),
                Some(u) => {
                    rankable += 1;
                    // TWO conditions, reported separately. Statistical resolvability alone is not
                    // enough: three paired deltas can have a sample SD near zero by chance, and a
                    // 0.1/255 "regression" resolved that way is an artifact of a small sample, not a
                    // finding. Materiality alone is not enough either — that is just eyeballing.
                    let resolved = delta.abs() > u;
                    let material = delta.abs() >= KV_TIER_MATERIAL_FLOOR;
                    match (resolved, material, delta > 0.0) {
                        (false, _, _) => format!(
                            "unresolved ({form}; |Δ| {:.2} <= 2*SEM {u:.2}, so this A/B can only \
                             exclude a regression larger than {:.2}/255)",
                            delta.abs(),
                            u.max(KV_TIER_MATERIAL_FLOOR)
                        ),
                        (true, false, _) => format!(
                            "resolvable but IMMATERIAL ({form}; Δ {delta:+.2} beyond 2*SEM \
                             {u:.2}, but under the {KV_TIER_MATERIAL_FLOOR:.2}/255 floor)"
                        ),
                        (true, true, true) => {
                            worse.push(format!("{row} +{delta:.2}/255 ({form}, 2*SEM {u:.2})"));
                            format!("WORSE by {delta:.2} beyond 2*SEM {u:.2} ({form})")
                        }
                        (true, true, false) => {
                            format!("better by {:.2} beyond 2*SEM {u:.2} ({form})", -delta)
                        }
                    }
                }
            };
            lines.push(format!(
                "row {row}: {} {arm:.2} vs {KV_TIER_BF16} {base:.2} — {verdict}",
                self.kv
            ));
        }
        if compared == 0 {
            return Err(format!(
                "no bounded row is present in both arms at {} — there is nothing to compare",
                self.bucket
            ));
        }
        // The memory half — GATED, not reported. A quantized cache that does not lower the measured
        // peak has bought nothing, and "the KV dominates AR video memory" is the whole story.
        let mut mem: Vec<String> = Vec::new();
        let mut no_saving: Vec<String> = Vec::new();
        for row in ['A', 'B', 'C', 'D', 'F'] {
            if let (Some(arm), Some(base)) = (self.mean_peak(row), baseline.mean_peak(row)) {
                mem.push(format!(
                    "{row} {:.2}->{:.2} GiB ({:.2}x)",
                    base / 1024f64.powi(3),
                    arm / 1024f64.powi(3),
                    arm / base
                ));
                if arm >= base {
                    no_saving.push(format!("{row} {:.2}x", arm / base));
                }
            }
        }
        // The throughput half — reported, because the story's feasibility question explicitly asks
        // whether dequantizing per read spends the memory win back in bandwidth.
        let mut wall: Vec<String> = Vec::new();
        for row in ['A', 'B', 'C', 'D', 'F'] {
            if let (Some(arm), Some(base)) = (self.mean_wall_ms(row), baseline.mean_wall_ms(row)) {
                wall.push(format!(
                    "{row} {:.0}->{:.0} s ({:.2}x)",
                    base / 1000.0,
                    arm / 1000.0,
                    arm / base
                ));
            }
        }
        if !no_saving.is_empty() {
            return Err(format!(
                "KV tier `{}` did NOT lower the measured MLX active peak at {} on {} — the tier \
                 exists to buy memory, so a coherence result without a saving beside it is not a \
                 pass. Peaks: {}.\n  {}",
                self.kv,
                self.bucket,
                no_saving.join(", "),
                mem.join(", "),
                lines.join("\n  ")
            ));
        }

        if !worse.is_empty() {
            return Err(format!(
                "KV tier `{}` is RESOLVABLY WORSE than the shipped bf16 cache at {} on {}. A \
                 cheaper cache that costs long-clip coherence is not a win, whatever it saves \
                 ({}).\n  {}",
                self.kv,
                self.bucket,
                worse.join(", "),
                if mem.is_empty() {
                    "no peaks recorded".to_string()
                } else {
                    mem.join(", ")
                },
                lines.join("\n  ")
            ));
        }
        // Rankability decides the HEADLINE, which is the line a reader takes away. A comparison in
        // which nothing resolved has not shown agreement; it has shown that these seeds cannot tell.
        let headline = if rankable == 0 {
            format!(
                "NO ROW WAS RANKABLE — KV tier `{}` vs the shipped bf16 cache at {} is UNDECIDED \
                 over {compared} bounded row(s) (an arm has one seed, so there is no variance \
                 estimate). This is not agreement. Add seeds (KREA_S18_SEEDS) to both arms",
                self.kv, self.bucket
            )
        } else {
            format!(
                "KV tier `{}` is not resolvably worse than the shipped bf16 cache at {} over \
                 {rankable} rankable of {compared} bounded row(s)",
                self.kv, self.bucket
            )
        };
        Ok(format!(
            "{headline}. MLX active peak: {}. AR wall: {}.\n  {}",
            if mem.is_empty() {
                "not recorded".to_string()
            } else {
                mem.join(", ")
            },
            if wall.is_empty() {
                "not recorded".to_string()
            } else {
                wall.join(", ")
            },
            lines.join("\n  ")
        ))
    }

    /// Checks that a measurement is **structurally** what it claims to be, independent of whether
    /// there is enough of it to conclude anything.
    ///
    /// These three were inside [`S18Sweep::verdict`], which meant they only ran on a sweep complete
    /// enough to reach a verdict. Since sc-17655 the sweep is dispatched in row-sized pieces and a
    /// piece is *usually* not verdict-complete, so a mis-parameterised row would have sailed through
    /// green with a "partial sweep" line and only surfaced at re-aggregation — after the GPU time was
    /// already spent. They are cheap and they are about the rows themselves, so both the live sweep
    /// and the re-aggregation entry point now run them unconditionally.
    ///
    /// The shipped-row check is skipped when row A is absent, which is legitimate for a piece; row A
    /// being *required* is [`S18Sweep::verdict`]'s concern, not this one's.
    fn structural_checks(&self) -> std::result::Result<(), String> {
        if let Some(shipped_rolls) = self
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.rolls)
            .max()
        {
            if shipped_rolls < 10 {
                return Err(format!(
                    "the shipped row only rolled the window {shipped_rolls} times — that is not a \
                     long clip, so it cannot answer the long-clip question. Raise \
                     KREA_S18_LATENT_FRAMES."
                ));
            }
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
        Ok(())
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
        self.structural_checks()?;
        self.validate_window_dose_ladder()?;
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
            let worst_bounded = ['A', 'B', 'C', 'D', 'F']
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
            for (row, dose) in [('D', self.mean('D')), ('F', self.mean('F'))] {
                if let Some(d) = dose {
                    if d > DRIFT_BUDGET {
                        return Err(format!(
                            "the shipped window scored {a:.2}/255 (inside the {DRIFT_BUDGET:.2} \
                             budget) but wider bounded row {row} scored {d:.2} — the within-regime \
                             dose-response disagrees, so this run does not support a no-drift verdict"
                        ));
                    }
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
            // (rows A/D/F, the same local-attention path at wider windows and fewer rolls) is what
            // decides it, and it is judged SYMMETRICALLY — see `WindowAttribution`.
            //
            // sc-15585 enlarges the bounded ladder from A/D (13 -> 10 rolls) to A/D/F
            // (13 -> 10 -> 5 rolls). The decision statistic is the matched-seed OLS slope over all
            // three doses, not a two-row endpoint difference.
            let head = format!(
                "drift is real ({a:.2}/255 over {shipped_rolls} rolls against a {DRIFT_BUDGET:.2} \
                 budget)"
            );
            let rate = self.rate_floor_clause();
            let attribution = self.window_attribution();
            let d_rolls = self
                .cells
                .iter()
                .find(|c| c.row == 'D')
                .map(|c| c.rolls)
                .unwrap_or(0);
            let attrib_clause = match attribution {
                WindowAttribution::Unmeasured => String::new(),
                WindowAttribution::NoPositiveLinearDoseResponse {
                    f,
                    f_rolls,
                    slope,
                    unc,
                    n,
                } => {
                    let sinks = ['B', 'C']
                        .iter()
                        .filter_map(|&r| self.mean(r).map(|m| format!("{r} {m:.2}")))
                        .collect::<Vec<_>>()
                        .join(", ");
                    // Row E is deliberately NOT cited here: it is out of regime (a different
                    // attention mask), n=1, and has no variance estimate. It is a reference, and a
                    // reference does not get to carry an attribution claim.
                    //
                    // This arm returns early, so it is the one place that interpolates `{rate}`
                    // itself — mid-paragraph, because its own sentences continue past it. Every
                    // other arm gets the rate clause appended once below (sc-17841); keep this
                    // `{rate}` when editing the wording here.
                    return Ok(format!(
                        "{head}, but it does NOT support a positive linear bounded-window dose \
                         response: the three-dose within-regime fit runs the wrong way — A/D/F span \
                         {shipped_rolls}/{d_rolls}/{f_rolls} rolls across {n} matched seeds and fit \
                         {slope:+.3}/255 per roll, clear of the predeclared 2*SEM heuristic \
                         {unc:.3}. Row F scores {f:.2}/255 against shipped A's {a:.2}; fewer \
                         evictions predict less coherence, excluding a positive linear/monotone \
                         roll-count contribution over the measured 5-13-roll range. This does not \
                         exclude a non-linear or otherwise different cache-window mechanism, and a \
                         first-chunk sink anchor is not indicated.{rate} No sink is wired. Sink \
                         rows for the record: {sinks}. The drift itself needs its own investigation."
                    ));
                }
                WindowAttribution::Unresolved {
                    f,
                    f_rolls,
                    slope,
                    unc,
                    n,
                } => format!(
                    ", the attribution to the bounded KV window is NOT resolvable at this sample \
                     size — the three-dose A/D/F fit over {n} matched seeds and \
                     {shipped_rolls}/{d_rolls}/{f_rolls} rolls is {slope:+.3}/255 per roll, inside the \
                     predeclared 2*SEM heuristic of {unc:.3}, so NEITHER direction may be asserted. \
                     Row F scores \
                     {f:.2}/255 against shipped A's {a:.2}. Across the enlarged {}-roll span, this \
                     design's practical 2*SEM magnitude floor is {:.2}/255 \
                     ({:.0}% of the shipped row's drift); anything smaller remains below the \
                     practical floor.",
                    shipped_rolls.saturating_sub(f_rolls),
                    (slope.abs() + unc) * shipped_rolls.saturating_sub(f_rolls) as f64,
                    100.0 * (slope.abs() + unc) * shipped_rolls.saturating_sub(f_rolls) as f64
                        / a.max(1e-6)
                ),
                WindowAttribution::PositiveLinearDoseResponse {
                    f,
                    f_rolls,
                    slope,
                    unc,
                    n,
                } => format!(
                    ", and it supports a positive bounded-window dose response: the three-dose A/D/F \
                     fit over {n} matched seeds and {shipped_rolls}/{d_rolls}/{f_rolls} rolls is \
                     {slope:+.3}/255 per roll, clear of the predeclared 2*SEM heuristic {unc:.3}. Row \
                     F scores {f:.2}/255 against shipped A's {a:.2}; more evictions predict more drift \
                     across the measured bounded-window dose range. This is evidence for a linear \
                     roll-count contribution in that range, not proof that every drift mechanism is \
                     cache-window driven."
                ),
            };
            // The rate floor is an A-vs-Z comparison — "does the shipped window's drift rate exceed
            // the zero-eviction rate on the same content" — and is INDEPENDENT of the A/D/F dose
            // ladder. It used to be interpolated inside the attribution arms only, so a sweep whose
            // attribution is `Unmeasured` discarded the measured row-Z evidence along with the empty
            // attribution clause and never printed it. That is the shape new sweeps now take at the
            // shipping bucket: row F needs ~49 GiB at 832x480 against `nax-macos-2`'s ~101 GiB
            // (the committed `MEASURED_832` record predates that host and still carries one), so a
            // fresh 832x480 sweep there has no row F at all, and sc-17324's three row-Z cells of
            // real GPU time went unreported (sc-17841). Appended once here so no attribution outcome
            // can drop it again.
            let attrib_and_rate = if rate.is_empty() {
                attrib_clause
            } else if attrib_clause.is_empty() {
                // `head` ends on `)`, not a full stop, and the rate clause is a sentence.
                format!(".{rate}")
            } else {
                format!("{attrib_clause}{rate}")
            };
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
                    "{head}{attrib_and_rate} And the sink anchor's effect on it is NOT resolvable at \
                     this sample size either: the best sink row ({row}) is {best_sink:.2}/255 \
                     against a {threshold:.2} repair threshold, a gap of {:.2} inside a combined \
                     between-seed scatter of {combined:.2}. No sink is wired — permanently-resident \
                     KV must not be bought on an unresolved comparison.",
                    (threshold - best_sink).abs()
                ));
            }
            if best_sink >= threshold {
                return Ok(format!(
                    "{head}{attrib_and_rate} And a first-chunk sink anchor does NOT repair it: the \
                     best sink row ({row}) only reached {best_sink:.2}/255 against a {threshold:.2} \
                     repair threshold. No sink is wired; sc-15127 needs a different anchor."
                ));
            }
            Ok(format!(
                "{head}{attrib_and_rate} And a first-chunk sink anchor repairs it to \
                 {best_sink:.2}/255 (row {row})."
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
    let sweep = |means: [f64; 6], spread: f64, n: usize| {
        let mut cells = Vec::new();
        for (i, row) in ['A', 'B', 'C', 'D', 'F', 'E'].iter().enumerate() {
            for k in 0..n {
                let off = if n < 2 {
                    0.0
                } else {
                    spread * (k as f64 / (n - 1) as f64 - 0.5)
                };
                // Dose rows have independent per-seed noise. Reusing the same offset on every row
                // would cancel it from each matched-seed slope and fabricate zero uncertainty.
                let row_noise = if *row == 'F' { -off } else { off };
                cells.push(S18Cell {
                    row: *row,
                    seed: k as u64,
                    latent_frames: 45,
                    rolls: match *row {
                        'D' => 10,
                        'F' => 5,
                        'E' => 0,
                        _ => 13,
                    },
                    reported_drift: means[i] + row_noise,
                    trend: means[i] + row_noise,
                    excursion: 0.0,
                    // A 100-output-frame post segment, so slope == trend numerically. Sweeps built by
                    // this closure have no row Z, so the rate-floor clause is inert in them; cases 2g
                    // and 2h below add one explicitly, and the RECORDED 640x384 data has one.
                    slope: means[i] + row_noise,
                    peak_bytes: 1,
                    clip_mean: 0.0,
                    head_motion: 2.0,
                    tail_motion: 2.0,
                    ar_wall_ms: 0,
                    component: "luma-mean",
                });
            }
        }
        S18Sweep {
            bucket: "test".into(),
            kv: KV_TIER_BF16.to_string(),
            cells,
        }
    };

    // Add the zero-eviction row Z that `sweep` does not build, so the A-vs-Z rate-floor comparison
    // has something to compare. Shaped like the recorded 640x384 row Z: a 6-latent-frame clip — the
    // shipped window, so it never evicts and `structural_checks`' `rolls == 0` holds — whose
    // `100 * trend / slope` post segment is ~12 output frames against row A's 100. Row A's mean slope
    // is its mean drift in these sweeps (slope == trend in the closure above), so the caller picks
    // which side of row A row Z lands on purely by choosing slopes.
    let with_z = |mut s: S18Sweep, slopes: [f64; 3]| {
        for (k, slope) in slopes.into_iter().enumerate() {
            s.cells.push(S18Cell {
                row: 'Z',
                seed: k as u64,
                latent_frames: 6,
                rolls: 0,
                reported_drift: 7.2,
                trend: 7.2,
                excursion: 0.0,
                slope,
                peak_bytes: 1,
                clip_mean: 0.0,
                head_motion: 2.0,
                tail_motion: 2.0,
                ar_wall_ms: 0,
                component: "luma-mean",
            });
        }
        s
    };

    // 1. Coherent: every bounded row sits far under the budget, with a resolvable margin.
    let v = sweep([2.0, 2.1, 1.9, 2.2, 2.0, 1.8], 0.4, 3)
        .verdict()
        .expect("a flat, replicated sweep must yield a verdict");
    assert!(v.starts_with("coherent"), "got: {v}");

    // 2. Drift, repaired by the anchor: the shipped row is far past the budget and the sinks pull it
    //    most of the way back.
    let v = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a repaired-drift sweep must yield a verdict");
    assert!(v.starts_with("drift is real"), "got: {v}");

    // 2b. Drift with a significant NEGATIVE linear dose response: fewer rolls are worse. This rejects
    //     a positive linear/monotone roll-count contribution over the measured range, but must not
    //     claim that every possible cache-window mechanism is impossible or reach for a sink.
    let v = sweep([40.0, 8.0, 6.0, 42.0, 45.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unattributed drift is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        v.contains("does NOT support a positive linear bounded-window dose response"),
        "got: {v}"
    );
    assert!(
        v.contains("does not exclude a non-linear"),
        "the conclusion must retain the model-form limitation: {v}"
    );
    assert!(!v.contains("repairs it to"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");
    assert!(v.contains("three-dose within-regime fit"), "got: {v}");
    // ...and it must NOT lean on row E. Row E is out of regime (a different attention mask), n=1 and
    // has no variance estimate; citing it as attribution evidence is the same inference this rule
    // refuses everywhere else, with the sign flipped.
    assert!(
        !v.contains("global"),
        "the out-of-regime, n=1 global reference row must not appear in the attribution sentence: {v}"
    );
    // ...and with a row Z present this arm must still carry the rate clause. It is the ONE arm that
    // returns early, so it interpolates `{rate}` itself rather than taking the shared append below
    // it; without this case nothing in the suite would notice that copy being deleted as redundant.
    let v = with_z(
        sweep([40.0, 8.0, 6.0, 42.0, 45.0, 2.0], 2.0, 3),
        [60.0, 61.0, 62.0],
    )
    .verdict()
    .expect("a negative dose response with a measured Z is still a conclusion");
    assert!(
        v.contains("sink anchor is not indicated. The within-regime zero-eviction row Z"),
        "the early-returning attribution arm must emit the rate clause in place, before its own \
         remaining sentences: {v}"
    );

    // 2c. A shallow three-dose slope inside its between-seed scatter. The symmetric rule must refuse,
    //     in both directions, and report the practical 2*SEM magnitude floor without calling it a
    //     confidence interval.
    let v = sweep([40.0, 8.0, 6.0, 40.5, 41.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unresolvable attribution is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        v.contains("attribution to the bounded KV window is NOT resolvable"),
        "got: {v}"
    );
    assert!(!v.contains("NOT attributable"), "got: {v}");
    assert!(v.contains("NEITHER direction may be asserted"), "got: {v}");
    assert!(v.contains("practical 2*SEM magnitude floor"), "got: {v}");

    // 2d. **The reviewer's own probe.** Row D is genuinely ~40% BETTER than row A, but the scatter
    //     swamps it. The old rule returned an unqualified "the dose-response runs the wrong way /
    //     eviction is not the mechanism" on exactly this data. It must not.
    let probe = |row: char, vals: [f64; 3]| {
        vals.into_iter().enumerate().map(move |(k, t)| S18Cell {
            row,
            seed: k as u64,
            latent_frames: 45,
            rolls: match row {
                'D' => 10,
                'F' => 5,
                _ => 13,
            },
            reported_drift: t,
            trend: t,
            excursion: 0.0,
            slope: t,
            peak_bytes: 1,
            clip_mean: 0.0,
            head_motion: 2.0,
            tail_motion: 2.0,
            ar_wall_ms: 0,
            component: "luma-mean",
        })
    };
    let mut cells: Vec<S18Cell> = probe('A', [20.0, 40.0, 22.0]).collect();
    cells.extend(probe('D', [10.0, 26.0, 13.0]));
    cells.extend(probe('F', [30.0, 5.0, 10.0]));
    cells.extend(probe('B', [18.0, 30.0, 20.0]));
    cells.extend(probe('C', [17.0, 28.0, 19.0]));
    let v = S18Sweep {
        bucket: "reviewer-probe".into(),
        kv: KV_TIER_BF16.to_string(),
        cells,
    }
    .verdict()
    .expect("the probe must still yield a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        !v.contains("NOT attributable"),
        "the reviewer's probe has row D 40% BETTER than row A and must never return a window \
         falsification — got: {v}"
    );
    assert!(
        !v.contains("runs the wrong way"),
        "the reviewer's probe must never return a wrong-way dose-response — got: {v}"
    );
    assert!(
        v.contains("attribution to the bounded KV window is NOT resolvable"),
        "got: {v}"
    );

    // 2e. And the OTHER direction must be assertable when the evidence clears the heuristic: a wider
    //     window materially better than the shipped one supports a positive bounded-window response.
    let v = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an implicating dose-response is still a conclusion");
    assert!(
        v.contains("supports a positive bounded-window dose response"),
        "got: {v}"
    );

    // 2f. The middle D dose must materially participate in the three-point fit. Hold the A/F
    //     endpoints fixed and move only D across the curve: an endpoint-only implementation would
    //     return the same classification for both sweeps and this mutation gate would fail.
    let negative_middle = sweep([40.0, 8.0, 6.0, 0.0, 40.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a negative middle-dose slope is still a conclusion");
    let positive_middle = sweep([40.0, 8.0, 6.0, 100.0, 40.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a positive middle-dose slope is still a conclusion");
    assert!(
        negative_middle.contains("does NOT support a positive linear bounded-window dose response"),
        "moving only D low must make the fitted slope negative: {negative_middle}"
    );
    assert!(
        positive_middle.contains("supports a positive bounded-window dose response"),
        "moving only D high must make the fitted slope positive: {positive_middle}"
    );

    // 2g. **Drift, no row F, but a measured row Z.** The A-vs-Z rate-floor comparison is independent
    //     of the A/D/F dose ladder, so an `Unmeasured` attribution must not take the row-Z evidence
    //     down with it. This is the shape a fresh 832x480 sweep takes on the current host, where row
    //     F does not fit — the committed `MEASURED_832` still carries an F, but new runs will not
    //     (sc-17841).
    let no_f = || {
        let mut s = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
        s.cells.retain(|c| c.row != 'F');
        s
    };
    // Row A's mean slope is 40.00/100f, so a Z above it is the recorded finding: Z is HIGHER,
    // therefore it is not a rate floor.
    let v = with_z(no_f(), [60.0, 61.0, 62.0])
        .verdict()
        .expect("a drift sweep with no F but a measured Z is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(
        v.contains("zero-eviction row Z does NOT establish a rate floor here"),
        "the A-vs-Z rate comparison must survive an Unmeasured window attribution — this is the \
         sc-17841 regression, and without the fix the clause is discarded with the empty \
         attribution clause: {v}"
    );
    assert!(
        v.contains("which is HIGHER than the shipped row's"),
        "row Z above row A must report the HIGHER branch, with its numbers: {v}"
    );
    assert!(
        v.contains("budget). The within-regime"),
        "with no attribution clause the rate sentence must be joined to the head by a full stop, \
         not run on after `budget)`: {v}"
    );
    // ...and the clause must be an addition, not a replacement: the sink conclusion still lands.
    assert!(v.contains("repairs it to"), "got: {v}");
    // The OTHER rate-floor branch — a row Z genuinely below row A — would mean a same-content floor
    // does exist, which is the outcome the recorded sweeps' narrowed wording is contingent on. It has
    // no coverage anywhere else in the suite, and it is reachable from exactly this shape.
    let v = with_z(no_f(), [10.0, 11.0, 12.0])
        .verdict()
        .expect("a drift sweep with a low row Z is still a conclusion");
    assert!(
        v.contains("Z is lower, so the shipped row's rate does exceed the zero-eviction rate"),
        "row Z below row A must report the LOWER branch: {v}"
    );
    // Negative control: the same sweep with NO row Z must not manufacture a rate clause, and the
    // sentence-joining full stop above must not leak into the empty case. The missing stop after
    // `budget)` here is pre-existing wording this change deliberately leaves alone — it is pinned so
    // the join stays conditional on there being a clause to join.
    let v = no_f()
        .verdict()
        .expect("a drift sweep with neither F nor Z is still a conclusion");
    assert!(
        !v.contains("zero-eviction row Z"),
        "an unmeasured row Z must produce no rate clause at all: {v}"
    );
    assert!(
        v.contains("budget) And a first-chunk"),
        "with no rate clause the head must not gain a trailing full stop: {v}"
    );

    // 2h. The rate clause must ALSO still be emitted on the attribution arms that already carried it,
    //     now that it is appended once rather than interpolated per-arm.
    let v = with_z(
        sweep([40.0, 8.0, 6.0, 40.5, 41.0, 2.0], 2.0, 3),
        [60.0, 61.0, 62.0],
    )
    .verdict()
    .expect("an unresolvable attribution with a measured Z is still a conclusion");
    assert!(
        v.contains("attribution to the bounded KV window is NOT resolvable"),
        "got: {v}"
    );
    assert!(
        v.contains("practical floor. The within-regime zero-eviction row Z"),
        "the rate clause must follow the attribution clause without losing its sentence break: {v}"
    );

    // 3. Drift the sink does NOT repair — a real finding, but it must never read as "ship a sink".
    let v = sweep([40.0, 38.0, 39.0, 30.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unrepaired drift is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("does NOT repair it"), "got: {v}");
    assert!(!v.contains("repairs it to"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 3b. Drift where the sink comparison is swamped by seed scatter — must say so, and must still
    //     refuse to wire a sink.
    let v = sweep([40.0, 22.0, 26.0, 30.0, 20.0, 2.0], 12.0, 3)
        .verdict()
        .expect("an unresolvable repair comparison is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("NOT resolvable"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 4. Incoherent evidence: the shipped row looks clean but the WIDER window looks worse. That
    //    ordering is impossible under the drift hypothesis, so the run proves nothing either way.
    let e = sweep([3.0, 3.0, 3.0, 25.0, 40.0, 2.0], 0.4, 3)
        .verdict()
        .expect_err("contradictory rows must not produce a no-drift verdict");
    assert!(e.contains("dose-response"), "got: {e}");

    // 5. UNDERPOWERED, near the budget: one seed per row and a shipped row close to the budget. The
    //    rule must refuse rather than pick a side — this is the guard the review demanded.
    let e = sweep([13.0, 8.0, 9.0, 12.5, 12.0, 2.0], 0.0, 1)
        .verdict()
        .expect_err("a single-seed sweep near the budget must not produce a verdict");
    assert!(e.contains("UNDERPOWERED"), "got: {e}");
    assert!(e.contains("NO variance estimate"), "got: {e}");

    // 5b. ...and the same means WITH enough seeds still refuse while the shipped row's own scatter
    //     swamps its margin against the budget.
    let e = sweep([13.0, 8.0, 9.0, 12.5, 12.0, 2.0], 12.0, 3)
        .verdict()
        .expect_err("a sweep whose seed scatter swamps its margin must not produce a verdict");
    assert!(e.contains("between-seed spread"), "got: {e}");

    // 6. Underpowered but unambiguously clean: one seed, but everything is far under the budget. That
    //    is reportable — as a narrowed claim that says so.
    let v = sweep([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.0, 1)
        .verdict()
        .expect("a single-seed but grossly clean sweep must yield a narrowed verdict");
    assert!(v.starts_with("underpowered but no gross drift"), "got: {v}");
    assert!(v.contains("does NOT rank the configs"), "got: {v}");

    // 7. And the sweep must have been long enough / the reference must really be a reference.
    let mut short = sweep([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.4, 3);
    for c in short.cells.iter_mut().filter(|c| c.row == 'A') {
        c.rolls = 4;
    }
    assert!(short.verdict().unwrap_err().contains("not a long clip"));
    let mut bad_ref = sweep([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.4, 3);
    for c in bad_ref.cells.iter_mut().filter(|c| c.row == 'E') {
        c.rolls = 3;
    }
    assert!(bad_ref.verdict().unwrap_err().contains("not a reference"));
    let mut unmatched = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    unmatched.cells.retain(|c| !(c.row == 'F' && c.seed == 2));
    assert!(
        unmatched
            .verdict()
            .unwrap_err()
            .contains("same matched seeds"),
        "dropping one F seed must invalidate the paired slope"
    );
    let mut wrong_dose = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    for c in wrong_dose.cells.iter_mut().filter(|c| c.row == 'F') {
        c.rolls = 10;
    }
    assert!(
        wrong_dose
            .verdict()
            .unwrap_err()
            .contains("exact 45-latent-frame, 13/10/5-roll"),
        "using the wrong F window must invalidate the three-dose fit"
    );
    let mut short_f = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    for c in short_f.cells.iter_mut().filter(|c| c.row == 'F') {
        c.latent_frames = 36;
        c.rolls = 2;
    }
    assert!(
        short_f
            .verdict()
            .unwrap_err()
            .contains("exact 45-latent-frame, 13/10/5-roll"),
        "a shorter F clip must not masquerade as the planned dose"
    );
    let mut duplicate_seed = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    let duplicate = *duplicate_seed
        .cells
        .iter()
        .find(|c| c.row == 'F' && c.seed == 0)
        .unwrap();
    duplicate_seed.cells.push(duplicate);
    assert!(
        duplicate_seed
            .verdict()
            .unwrap_err()
            .contains("duplicate seed cells"),
        "duplicate evidence must not be silently reduced to the first matching cell"
    );
    let mut non_finite = sweep([40.0, 8.0, 6.0, 30.0, 20.0, 2.0], 2.0, 3);
    non_finite
        .cells
        .iter_mut()
        .find(|c| c.row == 'F' && c.seed == 0)
        .unwrap()
        .trend = f64::INFINITY;
    assert!(
        non_finite
            .verdict()
            .unwrap_err()
            .contains("non-finite drift statistic"),
        "non-finite evidence must never fall through to an attribution verdict"
    );
    // 8. No shipped row at all — refuse, do not index off the end.
    let empty = S18Sweep {
        bucket: "test".into(),
        kv: KV_TIER_BF16.to_string(),
        cells: vec![],
    };
    assert!(empty.verdict().unwrap_err().contains("nothing to conclude"));
}

/// **The two WITHDRAWN sc-15127 claims must not survive in crate *prose*.**
///
/// Every other S18 gate in this file inspects the *generated verdict string*. That is exactly why two
/// stale doc blocks in `src/t2v.rs` — the module header and the `mac_ar_config` ship-decision doc —
/// asserted both withdrawn claims through two full review rounds: no gate reads hand-written prose, so
/// prose is where a retracted conclusion hides. This test closes that hole by reading the doc-bearing
/// sources themselves.
///
/// Doc comments wrap, so the check normalises `//!` / `///` markers and runs of whitespace to single
/// spaces before matching — a banned phrase split across two lines is still a banned phrase.
#[test]
fn the_withdrawn_s18_claims_do_not_survive_in_crate_prose() {
    /// `//!` / `///` markers and line wrapping removed, whitespace collapsed, lowercased.
    fn normalise(src: &str) -> String {
        src.split_whitespace()
            .filter(|w| *w != "//!" && *w != "///")
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    // Both phrases are WITHDRAWN findings, not style nits — see sc-15571's "WITHDRAWN" section.
    //
    //   "not attributable to the bounded ..." — the enlarged A/D/F 13/10/5-roll ladder remains
    //     UNRESOLVED in both buckets, not falsified: its matched-seed slopes (+0.571/255 at 640×384,
    //     −0.278/255 at 832×480) are inside their predeclared 2·SEM heuristics (1.897 and 1.678).
    //   "saturation run-away" — `saturation` wins NO row-A cell in either bucket (`S18Cell::component`
    //     records the winner per cell); the headline mode is colour-cast/tone/structure.
    const BANNED: &[(&str, &str)] = &[
        (
            "not attributable to the bounded",
            "the enlarged A/D/F 13/10/5-roll ladder remains UNRESOLVED in both buckets, not \
             falsified — its matched-seed slopes (+0.571/255 at 640×384, −0.278/255 at 832×480) \
             remain inside their predeclared 2·SEM heuristics (1.897 and 1.678)",
        ),
        (
            "saturation run-away",
            "`saturation` wins NO row-A cell in either bucket (see the per-cell `S18Cell::component` \
             winners) — the headline mode is a colour-cast/tone/structure drift, alongside which a \
             saturation rise is only separately observable",
        ),
    ];

    for (file, src) in [
        ("src/t2v.rs", include_str!("../src/t2v.rs")),
        ("src/lib.rs", include_str!("../src/lib.rs")),
    ] {
        let prose = normalise(src);
        for (phrase, why) in BANNED {
            assert!(
                !prose.contains(&phrase.to_lowercase()),
                "{file} contains the WITHDRAWN sc-15127 claim \"{phrase}\".\n\n\
                 Why it is banned: {why}.\n\n\
                 Do NOT delete this assertion to make the build pass — it exists because both of \
                 these phrases survived two review rounds in hand-written crate prose while every \
                 other S18 gate only inspected the generated verdict string. Rewrite the prose to \
                 match the verdict the recorded sweep actually supports (see \
                 `generate_t2v_from_components`' doc and sc-15571's WITHDRAWN section). If a future \
                 measurement genuinely re-establishes one of these claims, retire the entry here in \
                 the same change that records the data."
            );
        }
    }
}

/// **CI gate for the sc-17807 KV-tier A/B rule.** The quantized arm is measured on a gated host; the
/// rule applied to those numbers must not itself be untested, and it must be able to return every
/// failure it claims to gate. A comparison that can only pass is not a comparison.
///
/// Drives [`S18Sweep::kv_tier_comparison`] over each outcome, with hand-built cells so no weights are
/// needed. The baseline's scatter here is deliberately of the same order as the recorded sweep's
/// (±4-5/255 on row A at 640×384), NOT a fabricated tight one — a demonstration of symmetry on data
/// unlike the real data would prove nothing about the real comparison.
#[test]
fn the_kv_tier_comparison_can_fail_the_quantized_arm() {
    fn cell(row: char, seed: u64, drift: f64, peak: usize) -> S18Cell {
        S18Cell {
            row,
            seed,
            latent_frames: 45,
            rolls: 13,
            reported_drift: drift,
            trend: drift,
            excursion: 0.0,
            slope: 1.0,
            peak_bytes: peak,
            clip_mean: 1.0,
            head_motion: 12.0,
            tail_motion: 8.0,
            ar_wall_ms: 300_000,
            component: "luma-mean",
        }
    }
    let sweep = |kv: &str, cells: Vec<S18Cell>| S18Sweep {
        bucket: "640x384".into(),
        kv: kv.into(),
        cells,
    };
    // Row A at the recorded 640x384 scatter: 23.04 / 31.05 / 28.43, spread (2*SEM) ~4.72. Any
    // unpaired comparison against this cannot resolve less than ~6.7/255.
    const BASE: [f64; 3] = [23.0413, 31.0541, 28.4337];
    let seeds = [7u64, 11, 23];
    let arm_cells = |shift: f64, peak: usize| {
        seeds
            .iter()
            .zip(BASE)
            .map(|(&seed, d)| cell('A', seed, d + shift, peak))
            .collect::<Vec<_>>()
    };
    let bf16 = sweep(KV_TIER_BF16, arm_cells(0.0, 16_000_000_000));

    // 1. A near-identical arm is not a regression, and reports the saving and the wall time.
    let same = sweep("q8/g64", arm_cells(0.1, 12_000_000_000));
    let ok = same
        .kv_tier_comparison(&bf16)
        .expect("+0.10/255 is under the material floor");
    assert!(ok.contains("not resolvably worse"), "{ok}");
    // A constant shift gives the paired deltas zero scatter, so +0.10 clears its own 2*SEM. It must
    // still not be called a regression — and it must be REPORTED rather than swallowed, which is
    // what distinguishes a material floor from simply raising the threshold.
    assert!(ok.contains("resolvable but IMMATERIAL"), "{ok}");
    assert!(ok.contains("0.10"), "the magnitude must survive: {ok}");
    assert!(
        ok.contains("0.75x"),
        "the memory saving must be reported: {ok}"
    );
    assert!(ok.contains("AR wall"), "{ok}");

    // 1b. ...and a difference that is material but NOT statistically resolvable is also not a
    //     regression. Both conditions are required, and this is the one that fails the other way.
    let noisy_baseline = sweep(
        KV_TIER_BF16,
        vec![
            cell('A', 7, 20.0, 16_000_000_000),
            cell('A', 11, 30.0, 16_000_000_000),
            cell('A', 23, 25.0, 16_000_000_000),
        ],
    );
    let scattered = sweep(
        "q8/g64",
        vec![
            cell('A', 7, 27.0, 12_000_000_000),
            cell('A', 11, 26.0, 12_000_000_000),
            cell('A', 23, 32.0, 12_000_000_000),
        ],
    );
    let text = scattered
        .kv_tier_comparison(&noisy_baseline)
        .expect("a large but unresolved delta is not a regression");
    assert!(text.contains("unresolved"), "{text}");
    assert!(text.contains("can only exclude"), "{text}");

    // 2. THE POWER RESULT. A uniform +3.0/255 regression is FAR inside the unpaired 2*SEM (~6.7)
    //    and would pass an unpaired comparison — the paired form catches it, because both arms ran
    //    the same seeds and the per-seed delta is a constant with zero scatter. This is the whole
    //    reason the statistic is paired; if it regresses to unpaired, this assertion goes red.
    let worse = sweep("q8/g64", arm_cells(3.0, 12_000_000_000));
    let err = worse
        .kv_tier_comparison(&bf16)
        .expect_err("a uniform +3.00/255 paired regression must not pass");
    assert!(err.contains("RESOLVABLY WORSE"), "{err}");
    assert!(
        err.contains("paired"),
        "the paired form must be the one that caught it: {err}"
    );
    assert!(err.contains("not a win"), "{err}");

    // 3. Correctly signed: a uniform improvement is reported as better, not as worse.
    let better = sweep("q8/g64", arm_cells(-3.0, 12_000_000_000));
    let text = better
        .kv_tier_comparison(&bf16)
        .expect("better is not worse");
    assert!(text.contains("better by"), "{text}");

    // 4. MEMORY IS GATED, not decorated. An arm with identical drift but a HIGHER peak has bought
    //    nothing, and the tier exists to buy memory — so it must fail.
    let no_saving = sweep("q8/g64", arm_cells(0.0, 17_600_000_000));
    let err = no_saving
        .kv_tier_comparison(&bf16)
        .expect_err("a tier that raises the peak has bought nothing");
    assert!(
        err.contains("did NOT lower the measured MLX active peak"),
        "{err}"
    );
    assert!(err.contains("1.10x"), "{err}");
    // Equal peaks are also not a saving.
    assert!(sweep("q8/g64", arm_cells(0.0, 16_000_000_000))
        .kv_tier_comparison(&bf16)
        .is_err());

    // 5. One seed per row is UNRANKABLE, and the HEADLINE must say so. Reporting "not resolvably
    //    worse" off an arm with no variance estimate asserts a conclusion the data cannot support —
    //    the row detail said UNRANKABLE while the headline said agreement.
    let single = sweep("q8/g64", vec![cell('A', 7, 40.0, 12_000_000_000)]);
    let text = single
        .kv_tier_comparison(&bf16)
        .expect("an unrankable row is an absent comparison, not a failure");
    assert!(text.contains("NO ROW WAS RANKABLE"), "{text}");
    assert!(text.contains("This is not agreement"), "{text}");
    assert!(!text.contains("is not resolvably worse"), "{text}");

    // 6. Unmatched seeds fall back to the unpaired form rather than silently comparing nothing.
    let unmatched = sweep(
        "q8/g64",
        vec![
            cell('A', 101, 23.0, 12_000_000_000),
            cell('A', 102, 31.0, 12_000_000_000),
            cell('A', 103, 28.0, 12_000_000_000),
        ],
    );
    let text = unmatched
        .kv_tier_comparison(&bf16)
        .expect("unmatched seeds still compare, less powerfully");
    assert!(text.contains("unpaired"), "{text}");

    // 7. Malformed comparisons are refused rather than answered.
    assert!(bf16
        .kv_tier_comparison(&bf16)
        .unwrap_err()
        .contains("not an A/B"));
    let other_bucket = S18Sweep {
        bucket: "832x480".into(),
        kv: "q8/g64".into(),
        cells: arm_cells(0.0, 12_000_000_000),
    };
    assert!(other_bucket
        .kv_tier_comparison(&bf16)
        .unwrap_err()
        .contains("across geometries"));
    let no_overlap = sweep("q8/g64", vec![cell('D', 7, 20.0, 12_000_000_000)]);
    assert!(no_overlap
        .kv_tier_comparison(&bf16)
        .unwrap_err()
        .contains("nothing to compare"));
    // A "baseline" that is itself quantized is not the shipped reference.
    let q4 = sweep("q4/g64", arm_cells(0.0, 8_000_000_000));
    assert!(same
        .kv_tier_comparison(&q4)
        .unwrap_err()
        .contains("not the shipped"));
}

/// **Resolve a measured KV-tier A/B — the sc-17807 counterpart of
/// [`s18_verdict_from_accumulated_cells`].**
///
/// Both arms are measured by `long_clip_coherence_under_the_bounded_window` (the quantized one with
/// `KREA_S18_KV_BITS` set), each emitting `S18CELL` lines. This is what turns those two piles of
/// lines into an answer, and it applies **both** rules item 3 asks for:
///
///   * [`S18Sweep::verdict`] — the sweep's existing rule — to the quantized arm on its own, so the
///     arm has to be a structurally valid sweep before it is compared to anything;
///   * [`S18Sweep::kv_tier_comparison`] against the bf16 arm, which is the A/B itself.
///
/// ```text
/// KREA_S18_CELLS=$PWD/cells-bf16.tsv KREA_S18_Q8_CELLS=$PWD/cells-q8.tsv \
///   cargo test -p mlx-gen-krea-realtime --test generate_smoke \
///   s18_kv_tier_ab_from_accumulated_cells -- --exact --ignored --nocapture
/// ```
///
/// **Both arms must come from the same host.** The recorded `MEASURED_640` baseline was measured on
/// `nax-macos`; a second box reproduces its row A seed 7 closely but not bit-exactly (different
/// Metal silicon → a different trajectory, and even a different winning descriptor component). That
/// gap is well inside the between-seed spread, but comparing a local quantized arm against a remote
/// bf16 baseline would confound tier with host, so this takes two evidence files rather than
/// defaulting one side to the recorded table.
///
/// `#[ignore]` because it is an operator entry point over supplied evidence, like its sibling. The
/// rules it applies are gated without weights by
/// [`the_kv_tier_comparison_can_fail_the_quantized_arm`] and
/// [`the_s18_verdict_rule_distinguishes_its_outcomes`].
#[test]
#[ignore = "operator entry point: resolves a KV-tier A/B from KREA_S18_CELLS + KREA_S18_Q8_CELLS"]
fn s18_kv_tier_ab_from_accumulated_cells() {
    let read = |var: &str| -> String {
        let path = std::env::var(var)
            .unwrap_or_else(|_| panic!("set {var} to a file of accumulated S18CELL lines"));
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read `{path}`: {e}"))
    };
    let want_bucket = std::env::var("KREA_S18_BUCKET").ok();
    let baseline = s18_sweep_from_accumulated(&read("KREA_S18_CELLS"), want_bucket.as_deref())
        .unwrap_or_else(|e| panic!("KREA_S18_CELLS: {e}"));
    let arm = s18_sweep_from_accumulated(&read("KREA_S18_Q8_CELLS"), want_bucket.as_deref())
        .unwrap_or_else(|e| panic!("KREA_S18_Q8_CELLS: {e}"));

    println!("baseline: {}", baseline.summary());
    println!("arm:      {}", arm.summary());

    // The arm has to stand on its own before it is compared: same structural guards, same verdict
    // rule the bf16 sweep is held to. (`verdict` answers "does this sweep still say what the docs
    // say", not "is q8 acceptable" — the comparison below is what answers the second question.)
    if let Err(e) = arm.structural_checks() {
        panic!("quantized arm: {e}");
    }
    match arm.verdict() {
        Ok(v) => println!("  ARM VERDICT: {v}"),
        Err(e) => panic!("quantized arm: {e}"),
    }
    match baseline.verdict() {
        Ok(v) => println!("  BASELINE VERDICT: {v}"),
        Err(e) => panic!("baseline arm: {e}"),
    }
    match arm.kv_tier_comparison(&baseline) {
        Ok(v) => println!("  KV TIER A/B: {v}"),
        Err(e) => panic!("{e}"),
    }
}

/// **CI gate on the recorded sc-17807 KV-tier A/B — and it is a RECORDED FAILURE.**
///
/// The measured answer to the story's question ("a cheaper cache that increases drift is not a
/// win, and this sweep is the instrument that can say so") is that **Q8 KV costs coherence**. Both
/// arms were measured on one host — rows A/B/C/D x seeds 7/11/23 at 640x384, 45 latent frames, q4
/// weights — and the comparison returns `Err`:
///
/// | row | sink | per-seed Δ (q8 − bf16) | mean | 2*SEM | |
/// |---|---|---|---|---|---|
/// | A | 0 | +0.89, +7.36, −2.52 | +1.91 | 5.79 | unresolved |
/// | B | 1 | +2.12, +2.15, −0.67 | +1.20 | 1.87 | unresolved |
/// | C | 3 | +2.44, +3.59, +2.34 | **+2.79** | 0.80 | **RESOLVABLY WORSE** |
/// | D | 0 | −0.62, +3.34, +3.87 | +2.20 | 2.83 | unresolved |
///
/// **Read it as one effect, not one bad row.** Every row's mean is positive, 9 of the 12 paired
/// deltas are positive, and row C is simply the row whose paired scatter (2*SEM 0.80) is tight
/// enough to *resolve* something the others cannot — their bounds only exclude regressions above
/// 1.87–5.79/255. The consistent reading is a drift cost of roughly **+2/255** across the bounded
/// rows, resolvable where the power exists. That it lands on the largest-sink row is *consistent
/// with* — but does not establish — the mechanism you would guess: sink KV is quantized once and
/// then attended for the whole clip, so its error never ages out the way a rolling tail's does.
/// Distinguishing "an effect everywhere, resolved only on C" from "an effect concentrated on the
/// sink" needs more seeds on A/B/D, not more prose.
///
/// The memory saving is real and simultaneous: peak 0.86x (A), 0.85x (B), 0.82x (C), 0.76x (D).
/// So this is a **measured trade**, not a free win — which is exactly why
/// [`KreaArConfig::kv_cache_quant`](mlx_gen_krea_realtime::KreaArConfig::kv_cache_quant) defaults
/// to `None` and why nothing in the tree turns it on.
///
/// Provenance: `tests/fixtures/s18_kv_ab_{bf16,q8}_cells.tsv`, the verbatim `S18CELL` lines from
/// twelve gated runs. Row F is absent by decision — see
/// [`long_clip_coherence_under_the_bounded_window`].
///
/// This test asserts the **failure**. If a future change makes the A/B pass, that is either a real
/// improvement or a weakened rule, and either way it must be re-argued here rather than silently
/// flipping green.
#[test]
fn the_recorded_kv_tier_ab_records_a_resolvable_q8_regression() {
    let arm_of = |src: &str, want: &str| {
        let s = s18_sweep_from_accumulated(src, None).expect("the recorded arm must parse");
        assert_eq!(s.kv, want, "the recorded arm carries its tier");
        assert_eq!(s.cells.len(), 12, "4 bounded rows x 3 seeds");
        s
    };
    let bf16 = arm_of(
        include_str!("fixtures/s18_kv_ab_bf16_cells.tsv"),
        KV_TIER_BF16,
    );
    let q8 = arm_of(include_str!("fixtures/s18_kv_ab_q8_cells.tsv"), "q8/g64");

    // Both arms are structurally valid sweeps under the sweep's OWN rule before either is compared.
    for (label, arm) in [("bf16", &bf16), ("q8", &q8)] {
        arm.structural_checks()
            .unwrap_or_else(|e| panic!("{label} arm: {e}"));
        arm.verdict()
            .unwrap_or_else(|e| panic!("{label} arm verdict: {e}"));
    }

    let err = q8
        .kv_tier_comparison(&bf16)
        .expect_err("the recorded A/B is a REGRESSION — see this test's doc before changing it");
    assert!(err.contains("RESOLVABLY WORSE"), "{err}");
    assert!(
        err.contains("C +2.79/255"),
        "row C's effect size is the published one: {err}"
    );
    // The memory saving is real and must stay reported alongside the regression — the finding is a
    // trade, and an `Err` that dropped the saving would misrepresent it as a pure loss.
    for saving in ["0.86x", "0.85x", "0.82x", "0.76x"] {
        assert!(err.contains(saving), "missing peak ratio {saving}: {err}");
    }

    // Every row leans worse. This is what makes "one effect, resolved only where the power is"
    // the honest reading rather than "row C is an outlier" — if a re-measurement ever makes the
    // other rows negative, that reading has to change.
    let mut positive = 0;
    for row in ['A', 'B', 'C', 'D'] {
        let (a, b) = (q8.mean(row).unwrap(), bf16.mean(row).unwrap());
        assert!(a > b, "row {row}: q8 {a:.2} is not above bf16 {b:.2}");
        positive += q8
            .paired_deltas(&bf16, row)
            .expect("matched seeds")
            .iter()
            .filter(|d| **d > 0.0)
            .count();
    }
    assert_eq!(positive, 9, "9 of 12 paired deltas positive");

    // The knob this evidence is about must still be OFF by default. A measured coherence cost that
    // ships enabled would be the worst possible outcome of having measured it.
    assert_eq!(
        KreaRealtimeConfig::krea_realtime_14b().ar.kv_cache_quant,
        None
    );
}

/// **A mis-set KV tier must fail loudly, not silently measure the other arm.**
///
/// The failure this closes costs hours and produces evidence that looks fine: `KREA_S18_KV_BITS=q8`
/// (a plausible typo — the *label* is `q8`, the *value* is `8`) parsed with `.ok()` yields `None`,
/// so the sweep runs the bf16 cache, labels its cells `bf16`, and the A/B then compares the baseline
/// against itself and reports agreement. Every downstream guard in this file would be green.
///
/// Serial by construction: it mutates process env, so it must not interleave with another test that
/// reads it. `RUST_TEST_THREADS=1` is forced repo-wide (`.cargo/config.toml`), which is what makes
/// that safe here.
#[test]
fn a_malformed_kv_tier_env_is_refused_rather_than_silently_bf16() {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            std::env::remove_var("KREA_S18_KV_BITS");
            std::env::remove_var("KREA_S18_KV_GROUP_SIZE");
        }
    }
    let _guard = Guard;

    std::env::remove_var("KREA_S18_KV_BITS");
    std::env::remove_var("KREA_S18_KV_GROUP_SIZE");
    assert_eq!(
        parse_kv_tier_env().unwrap(),
        None,
        "unset is the shipped cache"
    );

    // Empty is how a shell spells "unset for this invocation" — accepted, and it means bf16.
    std::env::set_var("KREA_S18_KV_BITS", "");
    assert_eq!(parse_kv_tier_env().unwrap(), None);

    // The typo that motivates all of this.
    std::env::set_var("KREA_S18_KV_BITS", "q8");
    let err = parse_kv_tier_env().expect_err("`q8` is the label, not the value");
    assert!(err.contains("KREA_S18_KV_BITS"), "{err}");

    // A width MLX cannot express is refused here rather than deep inside the first chunk forward.
    std::env::set_var("KREA_S18_KV_BITS", "7");
    assert!(parse_kv_tier_env()
        .unwrap_err()
        .contains("affine quantization width"));

    // A valid tier resolves, group size included, and labels itself.
    std::env::set_var("KREA_S18_KV_BITS", "8");
    std::env::set_var("KREA_S18_KV_GROUP_SIZE", "32");
    let q = parse_kv_tier_env().unwrap().expect("a valid tier");
    assert_eq!(
        q,
        KvCacheQuant {
            bits: 8,
            group_size: 32
        }
    );
    assert_eq!(kv_tier_label(Some(q)), "q8/g32");
    assert_eq!(kv_tier_label(None), KV_TIER_BF16);

    // A half-set dispatch (group size, no width) is a mistake, not a bf16 request.
    std::env::remove_var("KREA_S18_KV_BITS");
    assert!(parse_kv_tier_env()
        .unwrap_err()
        .contains("KREA_S18_KV_GROUP_SIZE is set"));
}

/// Accumulated evidence from two KV tiers must be refused, not pooled — the sc-17807 analogue of the
/// duplicate-cell and mixed-bucket guards. Pooling would read as "more seeds" while silently
/// averaging two different models, and it is one `cat` away.
#[test]
fn accumulated_s18_evidence_refuses_to_pool_two_kv_tiers() {
    let bf16 = "S18CELL\tA\t7\t640x384\t45\t13\t23.0413\t19.9907\t23.0413\t12.8146\t15625190652\t\
                1.8230\t13.5111\t8.4549\tspatial-sd";
    let q8 = "S18CELL\tA\t11\t640x384\t45\t13\t23.5000\t20.0000\t23.5000\t12.9000\t9900000000\t\
              1.8000\t13.5000\t8.4000\tspatial-sd\tq8/g64";

    // Each tier on its own resolves, and carries the right label.
    assert_eq!(
        s18_sweep_from_accumulated(bf16, None)
            .expect("bf16 alone")
            .kv,
        KV_TIER_BF16,
        "a pre-sc-17807 line has no tier field and is bf16 by construction"
    );
    assert_eq!(
        s18_sweep_from_accumulated(q8, None).expect("q8 alone").kv,
        "q8/g64"
    );

    // Concatenated, they are refused.
    let mixed = format!("{bf16}\n{q8}");
    let err = s18_sweep_from_accumulated(&mixed, None)
        .expect_err("two tiers in one file must not be pooled");
    assert!(err.contains("mixes KV cache tiers"), "{err}");
    assert!(err.contains("two different models"), "{err}");
}

/// **The per-token KV figure quoted in crate prose must be the computed one (sc-17807).**
///
/// The same hole [`the_withdrawn_s18_claims_do_not_survive_in_crate_prose`] closes, for a *number*
/// rather than a claim: a 546 kB per-token cost sat in this file's own S18 doc block and in the S18
/// memory preflight's prose, 1.5× below the real figure, and nothing read it. A per-token cost that is
/// 1.5× low is not cosmetic — it is what makes a row look affordable on a host that it then takes
/// down, which is exactly what sc-17324 spent two runner crashes discovering.
///
/// The ban targets the **assertive** spellings — the withdrawn number quoted as *the* per-token cost,
/// with or without an approximation sign (the exact needles are assembled below). A sentence that names
/// the withdrawn number as withdrawn — as the two correction notes above do — is the point of the
/// correction, not a relapse, so it deliberately stays legal.
///
/// So: the stale figure is banned from the doc-bearing sources, and the figure that replaced it is
/// asserted to be the one [`KreaRealtimeConfig::kv_bytes_per_token`] computes — prose and code cannot
/// drift apart without this going red.
#[test]
fn the_kv_per_token_figure_in_crate_prose_is_the_computed_one() {
    fn normalise(src: &str) -> String {
        src.split_whitespace()
            .filter(|w| *w != "//!" && *w != "///" && *w != "//" && *w != "#")
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let per_token = cfg.kv_bytes_per_token().expect("the shipped bf16 cache");
    assert_eq!(
        per_token, 819_200,
        "800 KiB — 2 x 40 layers x 5120 x 2 bytes"
    );

    for (file, src) in [
        ("tests/generate_smoke.rs", include_str!("generate_smoke.rs")),
        ("src/causal.rs", include_str!("../src/causal.rs")),
        ("src/config.rs", include_str!("../src/config.rs")),
        ("src/t2v.rs", include_str!("../src/t2v.rs")),
        (
            "scripts/ci/s18_memory_preflight.py",
            include_str!("../../../../../scripts/ci/s18_memory_preflight.py"),
        ),
    ] {
        let prose = normalise(src);
        // The needles are ASSEMBLED rather than written as literals: this scan reads its own source
        // file, so writing a banned spelling out here would make the test flag its own assertion. (The
        // same reason `the_withdrawn_s18_claims_do_not_survive_in_crate_prose` scans only `src/`.)
        let n = 546;
        for stale in [
            format!("{n} kb/token"),
            format!("\u{2248}{n}"),
            format!("~{n}"),
        ] {
            assert!(
                !prose.contains(&stale),
                "{file} still quotes the WITHDRAWN `{stale}` KV cost. The derived figure is \
                 {per_token} bytes/token (800 KiB) — see KreaRealtimeConfig::kv_bytes_per_token, \
                 measured against the allocator by kv_cache_residency_at_the_production_geometry. \
                 Do not delete this assertion to make the build pass: a 1.5x-low per-token figure is \
                 what makes an unaffordable sweep row look affordable (sc-17324)."
            );
        }
        // ...and the corrected one is actually present where the cost is discussed, rather than the
        // stale figure having simply been deleted.
        assert!(
            prose.contains("800 kib") || prose.contains("819_200") || prose.contains("819,200"),
            "{file} no longer states the per-token KV cost at all — the correction was a deletion, \
             not a replacement"
        );
    }
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

#[derive(Debug)]
struct S18Evidence {
    row: char,
    seed: u64,
    bucket: String,
    /// The KV storage tier the emitting sweep ran at (sc-17807). `None` for the pre-sc-17807 lines,
    /// which are all bf16 by construction — the knob did not exist when they were measured.
    kv: Option<String>,
    /// AR-loop wall time in milliseconds. `None` for the pre-sc-17807 lines, which predate the field.
    ar_wall_ms: Option<u64>,
    latent_frames: usize,
    rolls: usize,
    drift: f64,
    trend: f64,
    excursion: f64,
    slope: f64,
    peak_bytes: usize,
    clip_mean: f64,
    head_motion: f64,
    tail_motion: f64,
    component: Option<String>,
}

fn parse_s18_evidence(src: &str) -> Result<Vec<S18Evidence>, String> {
    fn number<T: std::str::FromStr>(field: &str, name: &str, line: usize) -> Result<T, String> {
        field
            .parse()
            .map_err(|_| format!("line {line}: invalid {name} `{field}`"))
    }

    src.lines()
        .enumerate()
        .map(|(index, line)| {
            let line_number = index + 1;
            let fields: Vec<&str> = line.split('\t').collect();
            if !(14..=17).contains(&fields.len()) || fields[0] != "S18CELL" {
                return Err(format!(
                    "line {line_number}: expected a 14- to 17-field S18CELL record, got `{line}`"
                ));
            }
            let mut row_chars = fields[1].chars();
            let row = row_chars
                .next()
                .filter(|_| row_chars.next().is_none())
                .ok_or_else(|| format!("line {line_number}: invalid row `{}`", fields[1]))?;
            Ok(S18Evidence {
                row,
                seed: number(fields[2], "seed", line_number)?,
                bucket: fields[3].to_string(),
                kv: fields.get(15).map(|value| (*value).to_string()),
                ar_wall_ms: fields
                    .get(16)
                    .map(|value| number(value, "ar_wall_ms", line_number))
                    .transpose()?,
                latent_frames: number(fields[4], "latent_frames", line_number)?,
                rolls: number(fields[5], "rolls", line_number)?,
                drift: number(fields[6], "drift", line_number)?,
                trend: number(fields[7], "trend", line_number)?,
                excursion: number(fields[8], "excursion", line_number)?,
                slope: number(fields[9], "slope", line_number)?,
                peak_bytes: number(fields[10], "peak_bytes", line_number)?,
                clip_mean: number(fields[11], "clip_mean", line_number)?,
                head_motion: number(fields[12], "head_motion", line_number)?,
                tail_motion: number(fields[13], "tail_motion", line_number)?,
                component: fields.get(14).map(|value| (*value).to_string()),
            })
        })
        .collect()
}

/// Measured cells at **640×384** (the cheaper bucket for the global reference row, and the only one
/// where the sc-15127 sweep recorded it).
///
/// Row E is `n = 1`: its 41.90 GiB MLX peak (44,993,367,088 bytes; 45.0 GB decimal) drove this 128 GiB
/// host into enough swap to fill the boot
/// volume, so the remaining two seeds were abandoned. It is a *reference*, not a control, and nothing
/// the verdict decides turns on it — the attribution is decided by row D, which is within regime and
/// replicated.
///
/// Provenance: the exact source lines are committed in `tests/fixtures/s18_recorded_cells.tsv`.
/// Rows A/B and C seeds 7/11 came from `s18_640.log`
/// (SHA-256 `a9d3d0f4f4163171af2e83148bd75857266c7c97f26e16d391edd762f40d7803`);
/// C seed 23 and rows D/E/Z came from `s18_640c.log`
/// (SHA-256 `c89f4d7da05d3f367318dc03edf73e4f2e9c97509c13afcfaf508840f9e7019c`).
/// Row F came from `sc15585-f640.log`
/// (SHA-256 `802a4028e20059fb8ab75b64b56117fd8419b03e3b62502434bbe314706f8f56`).
/// Historical rows predate the component field; current rows retain `tail_motion` followed by the
/// winning descriptor component. The parser accepts both emitted schemas and binds every present
/// component to the typed table.
const MEASURED_640: &[S18Cell] = &[
    S18Cell {
        row: 'A',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 23.0413,
        trend: 19.9907,
        excursion: 23.0413,
        slope: 12.8146,
        peak_bytes: 15_625_190_652,
        clip_mean: 1.8230,
        head_motion: 13.5111,
        tail_motion: 8.4549,
        ar_wall_ms: 0,
        component: "spatial-sd",
    },
    S18Cell {
        row: 'A',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 31.0541,
        trend: 31.0541,
        excursion: 19.3187,
        slope: 19.9064,
        peak_bytes: 15_918_432_068,
        clip_mean: 2.1838,
        head_motion: 12.8849,
        tail_motion: 9.7033,
        ar_wall_ms: 0,
        component: "luma-mean",
    },
    S18Cell {
        row: 'A',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 28.4337,
        trend: 21.7860,
        excursion: 28.4337,
        slope: 13.9654,
        peak_bytes: 15_918_432_068,
        clip_mean: 2.9023,
        head_motion: 13.0206,
        tail_motion: 8.3340,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 17.4084,
        trend: 11.6090,
        excursion: 17.4084,
        slope: 7.4416,
        peak_bytes: 16_704_871_748,
        clip_mean: 0.5835,
        head_motion: 15.4236,
        tail_motion: 8.7276,
        ar_wall_ms: 0,
        component: "spatial-sd",
    },
    S18Cell {
        row: 'B',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 28.5211,
        trend: 28.5211,
        excursion: 12.3800,
        slope: 18.2828,
        peak_bytes: 16_704_871_748,
        clip_mean: 4.1115,
        head_motion: 15.1563,
        tail_motion: 10.9031,
        ar_wall_ms: 0,
        component: "luma-mean",
    },
    S18Cell {
        row: 'B',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 11.8207,
        trend: 11.8207,
        excursion: 10.3263,
        slope: 7.5774,
        peak_bytes: 16_704_871_748,
        clip_mean: 3.7060,
        head_motion: 15.8878,
        tail_motion: 9.0604,
        ar_wall_ms: 0,
        component: "saturation",
    },
    S18Cell {
        row: 'C',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 17.6473,
        trend: 17.6473,
        excursion: 14.7965,
        slope: 11.3124,
        peak_bytes: 18_277_770_564,
        clip_mean: 0.8194,
        head_motion: 15.7777,
        tail_motion: 9.0903,
        ar_wall_ms: 0,
        component: "contrast",
    },
    S18Cell {
        row: 'C',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 15.7251,
        trend: 15.7251,
        excursion: 0.0,
        slope: 10.0802,
        peak_bytes: 18_277_770_564,
        clip_mean: 4.7272,
        head_motion: 15.4910,
        tail_motion: 8.9464,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 6.2537,
        trend: 6.2537,
        excursion: 5.8270,
        slope: 4.0088,
        peak_bytes: 17_984_529_148,
        clip_mean: 3.9163,
        head_motion: 15.3418,
        tail_motion: 9.8476,
        ar_wall_ms: 0,
        component: "contrast",
    },
    S18Cell {
        row: 'D',
        seed: 7,
        latent_frames: 45,
        rolls: 10,
        reported_drift: 27.5739,
        trend: 27.5739,
        excursion: 21.8348,
        slope: 17.6756,
        peak_bytes: 22_939_277_456,
        clip_mean: 4.9800,
        head_motion: 16.1245,
        tail_motion: 6.8550,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'F',
        seed: 7,
        latent_frames: 45,
        rolls: 5,
        reported_drift: 32.7644,
        trend: 32.7644,
        excursion: 22.6072,
        slope: 21.0028,
        peak_bytes: 36_210_546_832,
        clip_mean: 11.7163,
        head_motion: 16.1245,
        tail_motion: 8.0779,
        ar_wall_ms: 0,
        component: "luma-mean",
    },
    S18Cell {
        row: 'F',
        seed: 11,
        latent_frames: 45,
        rolls: 5,
        reported_drift: 15.9079,
        trend: 6.2232,
        excursion: 15.9079,
        slope: 3.9892,
        peak_bytes: 36_503_788_248,
        clip_mean: 10.6727,
        head_motion: 15.3780,
        tail_motion: 10.0175,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'F',
        seed: 23,
        latent_frames: 45,
        rolls: 5,
        reported_drift: 22.3479,
        trend: 18.2705,
        excursion: 22.3479,
        slope: 11.7119,
        peak_bytes: 36_503_788_248,
        clip_mean: 7.7144,
        head_motion: 15.1025,
        tail_motion: 7.4781,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'D',
        seed: 11,
        latent_frames: 45,
        rolls: 10,
        reported_drift: 33.0361,
        trend: 33.0361,
        excursion: 23.0505,
        slope: 21.1770,
        peak_bytes: 23_232_518_872,
        clip_mean: 5.0395,
        head_motion: 15.3780,
        tail_motion: 6.6431,
        ar_wall_ms: 0,
        component: "luma-mean",
    },
    S18Cell {
        row: 'D',
        seed: 23,
        latent_frames: 45,
        rolls: 10,
        reported_drift: 31.1090,
        trend: 31.1090,
        excursion: 30.5435,
        slope: 19.9416,
        peak_bytes: 23_232_518_872,
        clip_mean: 5.6215,
        head_motion: 15.1025,
        tail_motion: 6.5790,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'E',
        seed: 7,
        latent_frames: 45,
        rolls: 0,
        reported_drift: 34.0619,
        trend: 34.0619,
        excursion: 18.5963,
        slope: 21.8345,
        peak_bytes: 44_993_367_088,
        clip_mean: 11.6145,
        head_motion: 16.1245,
        tail_motion: 9.1457,
        ar_wall_ms: 0,
        component: "luma-mean",
    },
    S18Cell {
        row: 'Z',
        seed: 7,
        latent_frames: 6,
        rolls: 0,
        reported_drift: 1.3928,
        trend: 1.3928,
        excursion: 0.0,
        slope: 11.6069,
        peak_bytes: 13_228_233_860,
        clip_mean: 6.1379,
        head_motion: 8.1185,
        tail_motion: 6.9907,
        ar_wall_ms: 0,
        component: "saturation",
    },
    S18Cell {
        row: 'Z',
        seed: 11,
        latent_frames: 6,
        rolls: 0,
        reported_drift: 3.3254,
        trend: 3.3254,
        excursion: 0.0,
        slope: 27.7120,
        peak_bytes: 13_228_233_860,
        clip_mean: 3.2575,
        head_motion: 8.8693,
        tail_motion: 9.6543,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'Z',
        seed: 23,
        latent_frames: 6,
        rolls: 0,
        reported_drift: 5.9501,
        trend: 5.9501,
        excursion: 0.0,
        slope: 49.5845,
        peak_bytes: 13_228_233_860,
        clip_mean: 2.8602,
        head_motion: 11.2963,
        tail_motion: 17.8587,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
];

/// Measured cells at **832×480** — the crate default and a shipping bucket. Row E is absent by
/// necessity — the sc-15127 sweep believed the global window could not run at this bucket. It can:
/// see the correction on [`long_clip_coherence_under_the_bounded_window`], where CI run 30787887176
/// measured row E here at a 63.32 GiB peak. These constants stay the sc-15127/sc-15585 record, with
/// the provenance below, so that later measurement is NOT retro-fitted into them.
///
/// Provenance: the exact source lines are committed in `tests/fixtures/s18_recorded_cells.tsv`.
/// All rows came from `s18_832.log`
/// (SHA-256 `e48d1d0ffd21b1d74833ee8d6624864132391b14b2f8f19d73bb216540704102`).
/// Row F came from `sc15585-f832.log`
/// (SHA-256 `b437a3d0beb6385f3f4231216380bf6841509069c2daa9d20f4f27ba0a47da75`).
/// Historical rows predate the component field; current rows retain `tail_motion` followed by the
/// winning descriptor component. The parser accepts both emitted schemas and binds every present
/// component to the typed table.
const MEASURED_832: &[S18Cell] = &[
    S18Cell {
        row: 'A',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 47.1323,
        trend: 45.8218,
        excursion: 47.1323,
        slope: 29.3729,
        peak_bytes: 18_719_499_004,
        clip_mean: 2.0986,
        head_motion: 14.0278,
        tail_motion: 14.0410,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'A',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 34.5532,
        trend: 27.7189,
        excursion: 34.5532,
        slope: 17.7685,
        peak_bytes: 19_012_740_420,
        clip_mean: 2.3080,
        head_motion: 15.8402,
        tail_motion: 11.2383,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'A',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 36.0093,
        trend: 29.2038,
        excursion: 36.0093,
        slope: 18.7204,
        peak_bytes: 19_012_740_420,
        clip_mean: 2.4257,
        head_motion: 12.0546,
        tail_motion: 4.9842,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 34.1733,
        trend: 25.6222,
        excursion: 34.1733,
        slope: 16.4245,
        peak_bytes: 20_336_515_800,
        clip_mean: 2.6006,
        head_motion: 15.8427,
        tail_motion: 18.7733,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 31.0055,
        trend: 23.7267,
        excursion: 31.0055,
        slope: 15.2094,
        peak_bytes: 20_336_515_800,
        clip_mean: 7.3309,
        head_motion: 16.7797,
        tail_motion: 15.2673,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 26.8110,
        trend: 18.8356,
        excursion: 26.8110,
        slope: 12.0741,
        peak_bytes: 20_336_515_800,
        clip_mean: 6.4201,
        head_motion: 14.5752,
        tail_motion: 7.9430,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 22.8868,
        trend: 14.6706,
        excursion: 22.8868,
        slope: 9.4042,
        peak_bytes: 23_148_059_352,
        clip_mean: 2.5411,
        head_motion: 16.4200,
        tail_motion: 15.0360,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 19.4879,
        trend: 11.0461,
        excursion: 19.4879,
        slope: 7.0808,
        peak_bytes: 23_148_059_352,
        clip_mean: 5.8078,
        head_motion: 16.8824,
        tail_motion: 15.6104,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 27.9238,
        trend: 21.1164,
        excursion: 27.9238,
        slope: 13.5362,
        peak_bytes: 23_148_059_352,
        clip_mean: 3.5535,
        head_motion: 15.3301,
        tail_motion: 8.2002,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'D',
        seed: 7,
        latent_frames: 45,
        rolls: 10,
        reported_drift: 47.4356,
        trend: 47.4356,
        excursion: 47.0857,
        slope: 30.4074,
        peak_bytes: 31_582_640_856,
        clip_mean: 3.0485,
        head_motion: 16.6669,
        tail_motion: 16.9721,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'D',
        seed: 11,
        latent_frames: 45,
        rolls: 10,
        reported_drift: 39.0004,
        trend: 35.2790,
        excursion: 39.0004,
        slope: 22.6147,
        peak_bytes: 31_582_640_856,
        clip_mean: 5.8909,
        head_motion: 16.7670,
        tail_motion: 10.8825,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'D',
        seed: 23,
        latent_frames: 45,
        rolls: 10,
        reported_drift: 22.8338,
        trend: 22.8338,
        excursion: 22.2395,
        slope: 14.6371,
        peak_bytes: 31_582_640_856,
        clip_mean: 5.0853,
        head_motion: 15.1700,
        tail_motion: 3.1060,
        ar_wall_ms: 0,
        component: "spatial-sd",
    },
    S18Cell {
        row: 'F',
        seed: 7,
        latent_frames: 45,
        rolls: 5,
        reported_drift: 52.3381,
        trend: 52.3381,
        excursion: 49.0238,
        slope: 33.5501,
        peak_bytes: 52_466_899_088,
        clip_mean: 6.6029,
        head_motion: 16.6669,
        tail_motion: 17.0563,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'F',
        seed: 11,
        latent_frames: 45,
        rolls: 5,
        reported_drift: 46.4310,
        trend: 43.6077,
        excursion: 46.4310,
        slope: 27.9537,
        peak_bytes: 52_760_140_504,
        clip_mean: 9.7644,
        head_motion: 16.7670,
        tail_motion: 17.9887,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'F',
        seed: 23,
        latent_frames: 45,
        rolls: 5,
        reported_drift: 23.9173,
        trend: 17.5511,
        excursion: 23.9173,
        slope: 11.2507,
        peak_bytes: 52_760_140_416,
        clip_mean: 7.7918,
        head_motion: 15.1700,
        tail_motion: 9.5642,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
];

/// Measured cells from the **sc-17324 re-measurement at 832×480** — a SEPARATE sweep from the
/// sc-15127/sc-15585 record in [`MEASURED_832`], which is why it lives in its own constant rather
/// than being retro-fitted into that one (see its doc).
///
/// It is the first sweep to measure **row Z at 832×480**. The earlier record only ever had a
/// zero-eviction row at 640×384, so the A-vs-Z rate comparison had never been made at the shipping
/// bucket. It has no rows D or F: row F needs ~46,800 tokens ≈ 49 GiB there and took `nax-macos-2`
/// (~101 GiB) down twice, so the window attribution at this bucket is `Unmeasured` by necessity —
/// which is exactly the shape that used to make [`S18Sweep::verdict`] discard the rate clause
/// (sc-17841). The same sweep effort also re-measured row F at **640×384** (run 31066428149); that
/// cell is deliberately NOT recorded here, because it duplicates a row the sc-15585 record already
/// holds at a different bucket and is a separate question from this one.
///
/// Provenance: the exact source lines are committed in `tests/fixtures/s18_sc17324_832_cells.tsv`
/// (SHA-256 `7d2990063260ccb6662db7011fe17cbf0dfb0ab0d2e6c10ca6c390e0b8945015`), downloaded verbatim
/// from the `krea-s18-sweep-*` artifacts of CI runs 31045621675 (row A), 31052700151 (row B),
/// 31060390246 (row C) and 31062386153 (row Z), all `Real-weight validation` on `nax-macos-2`,
/// 2026-08-05/06, seeds 7/11/23.
const MEASURED_832_SC17324: &[S18Cell] = &[
    S18Cell {
        row: 'A',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 46.0496,
        trend: 39.8551,
        excursion: 46.0496,
        slope: 25.5482,
        peak_bytes: 18_719_499_004,
        clip_mean: 2.3528,
        head_motion: 13.8854,
        tail_motion: 11.8398,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'A',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 29.7962,
        trend: 23.7863,
        excursion: 29.7962,
        slope: 15.2476,
        peak_bytes: 19_012_740_420,
        clip_mean: 4.1180,
        head_motion: 16.6477,
        tail_motion: 10.3963,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'A',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 33.1889,
        trend: 24.4320,
        excursion: 33.1889,
        slope: 15.6615,
        peak_bytes: 19_012_740_420,
        clip_mean: 2.4759,
        head_motion: 13.1568,
        tail_motion: 6.8421,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 35.3929,
        trend: 26.8303,
        excursion: 35.3929,
        slope: 17.1989,
        peak_bytes: 20_043_274_384,
        clip_mean: 2.7394,
        head_motion: 15.8696,
        tail_motion: 17.5720,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 28.4557,
        trend: 15.7773,
        excursion: 28.4557,
        slope: 10.1136,
        peak_bytes: 20_336_515_800,
        clip_mean: 5.9908,
        head_motion: 18.8493,
        tail_motion: 21.3614,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'B',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 24.1367,
        trend: 19.2189,
        excursion: 24.1367,
        slope: 12.3198,
        peak_bytes: 20_336_515_800,
        clip_mean: 3.2686,
        head_motion: 14.5934,
        tail_motion: 8.8424,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 7,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 24.3310,
        trend: 18.5957,
        excursion: 24.3310,
        slope: 11.9203,
        peak_bytes: 22_854_817_936,
        clip_mean: 4.0987,
        head_motion: 16.3794,
        tail_motion: 14.6675,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 11,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 22.1048,
        trend: 10.1373,
        excursion: 22.1048,
        slope: 6.4983,
        peak_bytes: 23_148_059_352,
        clip_mean: 5.3745,
        head_motion: 17.9580,
        tail_motion: 16.1059,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'C',
        seed: 23,
        latent_frames: 45,
        rolls: 13,
        reported_drift: 21.8817,
        trend: 12.1768,
        excursion: 21.8817,
        slope: 7.8056,
        peak_bytes: 23_148_059_352,
        clip_mean: 2.6829,
        head_motion: 15.9028,
        tail_motion: 8.5545,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'Z',
        seed: 7,
        latent_frames: 6,
        rolls: 0,
        reported_drift: 2.5149,
        trend: 2.5149,
        excursion: 0.0000,
        slope: 20.9576,
        peak_bytes: 14_455_722_556,
        clip_mean: 0.6207,
        head_motion: 12.3112,
        tail_motion: 16.1128,
        ar_wall_ms: 0,
        component: "opp-B-Y",
    },
    S18Cell {
        row: 'Z',
        seed: 11,
        latent_frames: 6,
        rolls: 0,
        reported_drift: 8.6831,
        trend: 8.6831,
        excursion: 0.0000,
        slope: 72.3588,
        peak_bytes: 14_748_963_972,
        clip_mean: 3.7369,
        head_motion: 13.8205,
        tail_motion: 19.4322,
        ar_wall_ms: 0,
        component: "spatial-sd",
    },
    S18Cell {
        row: 'Z',
        seed: 23,
        latent_frames: 6,
        rolls: 0,
        reported_drift: 2.5876,
        trend: 2.5876,
        excursion: 0.0000,
        slope: 21.5635,
        peak_bytes: 14_748_963_972,
        clip_mean: 4.7098,
        head_motion: 10.1449,
        tail_motion: 13.8240,
        ar_wall_ms: 0,
        component: "spatial-sd",
    },
];

/// **CI gate on the sc-17841 fix, against real cells rather than synthetic ones.**
///
/// [`the_s18_verdict_rule_distinguishes_its_outcomes`] proves the rule emits the rate clause on an
/// `Unmeasured` attribution using constructed sweeps. This proves it on the measurement that
/// motivated the fix, and pins the finding itself so it stops living in a story description: at the
/// **shipping** bucket, as at 640×384, the zero-eviction row runs FASTER than the shipped row, so
/// there is still no measured same-content rate floor anywhere.
///
/// Before sc-17841 the verdict for exactly these cells contained no rate sentence at all — the
/// clause was computed and then discarded with the empty attribution clause — so three cells of real
/// GPU time produced a number that had to be worked out by hand. This test is what makes that
/// unable to happen again quietly.
#[test]
fn the_sc17324_832_sweep_reports_its_rate_comparison() {
    // The committed raw lines and the typed table must not drift apart, exactly as
    // `the_recorded_s18_sweep_is_what_the_docs_claim` requires of the sc-15127 record.
    let evidence = parse_s18_evidence(include_str!("fixtures/s18_sc17324_832_cells.tsv"))
        .expect("the committed sc-17324 S18CELL evidence must parse");
    assert_eq!(
        evidence.len(),
        MEASURED_832_SC17324.len(),
        "the committed sc-17324 evidence and its table have different cell counts"
    );
    for raw in &evidence {
        assert_eq!(
            raw.bucket, "832x480",
            "this record is the 832x480 re-measurement"
        );
        let cell = MEASURED_832_SC17324
            .iter()
            .find(|c| c.row == raw.row && c.seed == raw.seed)
            .unwrap_or_else(|| {
                panic!(
                    "row {} seed {} is missing from the table",
                    raw.row, raw.seed
                )
            });
        assert_eq!(
            (
                raw.latent_frames,
                raw.rolls,
                raw.peak_bytes,
                raw.drift,
                raw.trend,
                raw.excursion,
                raw.slope,
                raw.clip_mean,
                raw.head_motion,
                raw.tail_motion,
                raw.component.as_deref(),
            ),
            (
                cell.latent_frames,
                cell.rolls,
                cell.peak_bytes,
                cell.reported_drift,
                cell.trend,
                cell.excursion,
                cell.slope,
                cell.clip_mean,
                cell.head_motion,
                cell.tail_motion,
                Some(cell.component),
            ),
            "row {} seed {} differs between the committed lines and the table",
            raw.row,
            raw.seed
        );
        assert!(
            DESCRIPTOR_NAMES.contains(&cell.component),
            "row {} seed {} records component `{}`, which is not a descriptor component",
            raw.row,
            raw.seed,
            cell.component
        );
    }

    let sweep = S18Sweep {
        bucket: "832x480".into(),
        // sc-17324's record predates the sc-17807 KV knob, so it is the shipped bf16 cache.
        kv: KV_TIER_BF16.to_string(),
        cells: MEASURED_832_SC17324.to_vec(),
    };
    println!("{}", sweep.summary());
    // The shape that motivated sc-17841: no D and no F, so the ladder cannot be fitted at all.
    assert!(
        matches!(sweep.window_attribution(), WindowAttribution::Unmeasured),
        "this record has no A/D/F ladder, so its window attribution must be Unmeasured — if that          changed, the sweep is no longer the case sc-17841 was about"
    );

    // The finding, pinned. `mean_slope` is the per-100-output-frame rate, the only statistic
    // comparable across row Z's 12-frame post segment and row A's 156.
    let (a, z) = (
        sweep.mean_slope('A').expect("row A is recorded"),
        sweep.mean_slope('Z').expect("row Z is recorded"),
    );
    assert!(
        (a - 18.8191).abs() < 0.0001 && (z - 38.2933).abs() < 0.0001,
        "the sc-17324 rate statistics drifted: A {a:.4}/100f, Z {z:.4}/100f; expected 18.8191 and          38.2933"
    );
    assert!(
        z > a,
        "row Z now runs SLOWER than row A at 832x480 ({z:.2} vs {a:.2}/100f) — a same-content rate          floor WOULD then exist at the shipping bucket, and both this crate's prose and sc-15571's          narrowed wording would need revisiting"
    );

    // ...and the verdict must actually SAY so. This is the sc-17841 regression gate on real data:
    // before the fix this string contained no rate sentence, because the attribution is Unmeasured.
    let v = sweep
        .verdict()
        .expect("the sc-17324 sweep supports a verdict");
    println!("  VERDICT: {v}");
    assert!(
        v.contains("zero-eviction row Z does NOT establish a rate floor here"),
        "the measured row-Z evidence must reach the verdict rather than being discarded with the          empty attribution clause (sc-17841): {v}"
    );
    assert!(
        v.contains("which is HIGHER than the shipped row's"),
        "row Z is faster than row A here, so the verdict must report the HIGHER branch: {v}"
    );
}

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
            kv: KV_TIER_BF16.to_string(),
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
        // The enlarged three-dose A/D/F ladder still does not resolve the attribution in either
        // bucket. Gate that wording so an unsupported positive or negative causal claim cannot return.
        let v = v.expect("verdict asserted above");
        assert!(
            v.contains("attribution to the bounded KV window is NOT resolvable"),
            "the recorded {bucket} sweep no longer returns an UNRESOLVED attribution — the docs and \
             sc-15571 say it is unresolved, so one of them is now wrong: {v}"
        );
        assert!(
            !v.contains("NOT attributable"),
            "the recorded {bucket} sweep returned a window falsification — that claim was withdrawn \
             in review and must not come back without a slope that clears the predeclared 2*SEM \
             heuristic: {v}"
        );

        // Bind the published exact statistics, not only the broad UNRESOLVED classification. This
        // makes A, D and F all evidence-bearing: changing any dose or silently replacing the
        // three-point OLS with an endpoint contrast changes at least one pinned number.
        let slopes = sweep.window_dose_slopes();
        let slope = mean_f64(&slopes);
        let uncertainty = 2.0 * std_f64(&slopes) / (slopes.len() as f64).sqrt();
        let practical_floor = 8.0 * (slope.abs() + uncertainty);
        let (expected_slope, expected_uncertainty, expected_floor) = match bucket {
            "640x384" => (0.5714, 1.8970, 19.75),
            "832x480" => (-0.2780, 1.6781, 15.65),
            _ => unreachable!(),
        };
        assert!(
            (slope - expected_slope).abs() < 0.0001
                && (uncertainty - expected_uncertainty).abs() < 0.0001
                && (practical_floor - expected_floor).abs() < 0.01,
            "{bucket} exact A/D/F statistics drifted: slope {slope:+.4}, 2*SEM \
             {uncertainty:.4}, practical floor {practical_floor:.2}; expected \
             {expected_slope:+.4}, {expected_uncertainty:.4}, {expected_floor:.2}"
        );
        // Every cell must name the descriptor component that actually scored it, and it must be a
        // real one. This is what makes "which channel failed?" checkable from the table.
        for c in &sweep.cells {
            assert!(
                DESCRIPTOR_NAMES.contains(&c.component),
                "{bucket} row {} seed {} records component `{}`, which is not a descriptor component",
                c.row,
                c.seed,
                c.component
            );
        }
        // ...and specifically: the corroborating `report_artifacts` saturation statistic is NOT the
        // channel that scored row A. Recording the winners is what lets the docs say that instead of
        // implying the two are the same mode.
        let a_components: Vec<&str> = sweep
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.component)
            .collect();
        assert!(
            !a_components.contains(&"saturation"),
            "{bucket} row A is now scored by the `saturation` component — the docs say the \
             corroborating saturation statistic is a DIFFERENT channel from the one that scored row \
             A, so that wording must be revisited: {a_components:?}"
        );
        // Replication is the other half of what the review demanded: a recorded bucket with one seed
        // per row cannot support a ranking claim, so it must not be recorded as if it could.
        for row in ['A', 'B', 'C', 'D', 'F'] {
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

    // The within-regime rate comparison the DRIFT_BUDGET doc used to assert and never compute. It is
    // recorded here as a FINDING, not a floor: row Z's 12-frame slope is higher than row A's over 156
    // frames, so it does not bound row A and the budget rests on the synthetic controls alone. If a
    // future measurement makes Z genuinely lower, this goes red and the doc's narrowed wording — and
    // sc-15571's — must be revisited, because then a same-content floor WOULD exist.
    let sweep = S18Sweep {
        bucket: "640x384".into(),
        kv: KV_TIER_BF16.to_string(),
        cells: MEASURED_640.to_vec(),
    };
    let (z, a) = (
        sweep.mean_slope('Z').expect("row Z is recorded at 640x384"),
        sweep.mean_slope('A').expect("row A is recorded at 640x384"),
    );
    println!("  rate floor: {}", sweep.rate_floor_clause());
    assert!(
        z > a,
        "row Z's zero-eviction rate is now {z:.2}/100f against row A's {a:.2} — Z is LOWER, so it \
         does bound row A's rate after all and the docs' \"no measured same-content rate floor\" \
         wording is stale"
    );
    // Row Z's post segment really is short — 12 output frames against row A's 156 — which is the
    // stated reason its slope does not extrapolate. Pin it so the argument cannot rot.
    let z_post = sweep.mean_post_len('Z').expect("row Z has a slope");
    let a_post = sweep.mean_post_len('A').expect("row A has a slope");
    assert!(
        z_post < a_post / 4.0,
        "row Z's post segment is {z_post:.0} output frames against row A's {a_post:.0} — the docs' \
         \"a 12-frame OLS slope does not extrapolate across 156\" argument no longer describes the \
         data"
    );

    // The committed fixture is the durable copy of the exact S18CELL lines emitted by the preserved
    // real-weight logs. Parse every field and compare it to the hand-shaped tables used by the verdict
    // so neither representation can drift independently.
    let evidence = parse_s18_evidence(include_str!("fixtures/s18_recorded_cells.tsv"))
        .expect("the committed S18CELL evidence must parse");
    let cells: Vec<&S18Cell> = MEASURED_640.iter().chain(MEASURED_832).collect();
    assert_eq!(
        evidence.len(),
        34,
        "the committed S18CELL evidence changed cell count; reconcile it with the source logs"
    );
    assert_eq!(
        evidence.len(),
        cells.len(),
        "the committed S18CELL evidence and MEASURED tables have different cell counts"
    );

    let mut seen = Vec::new();
    for raw in &evidence {
        let table = match raw.bucket.as_str() {
            "640x384" => MEASURED_640,
            "832x480" => MEASURED_832,
            other => panic!("fixture contains an unrecognised S18 bucket `{other}`"),
        };
        let matches: Vec<&S18Cell> = table
            .iter()
            .filter(|cell| cell.row == raw.row && cell.seed == raw.seed)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "{} row {} seed {} must identify exactly one MEASURED cell, found {}",
            raw.bucket,
            raw.row,
            raw.seed,
            matches.len()
        );
        let cell = matches[0];
        assert_eq!(
            (raw.latent_frames, raw.rolls, raw.peak_bytes),
            (cell.latent_frames, cell.rolls, cell.peak_bytes),
            "{} row {} seed {} integer fields differ from the committed S18CELL evidence",
            raw.bucket,
            raw.row,
            raw.seed
        );
        assert_eq!(
            [
                raw.drift,
                raw.trend,
                raw.excursion,
                raw.slope,
                raw.clip_mean,
                raw.head_motion,
                raw.tail_motion,
            ],
            [
                cell.reported_drift,
                cell.trend,
                cell.excursion,
                cell.slope,
                cell.clip_mean,
                cell.head_motion,
                cell.tail_motion,
            ],
            "{} row {} seed {} numeric fields differ from the committed S18CELL evidence",
            raw.bucket,
            raw.row,
            raw.seed
        );
        assert_eq!(
            cell.reported_drift,
            cell.drift(),
            "{} row {} seed {} records a drift that is not max(trend, excursion)",
            raw.bucket,
            raw.row,
            raw.seed
        );
        if raw.row == 'F' {
            assert!(
                raw.component.is_some(),
                "{} row F seed {} must retain the current 15-field S18CELL schema from the exact \
                 sc-15585 log; dropping the winning component weakens the evidence",
                raw.bucket,
                raw.seed
            );
        }
        if let Some(component) = &raw.component {
            assert_eq!(
                component, cell.component,
                "{} row {} seed {} component differs from the committed S18CELL evidence",
                raw.bucket, raw.row, raw.seed
            );
        }
        seen.push((raw.bucket.as_str(), raw.row, raw.seed));
    }
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        evidence.len(),
        "the committed S18CELL evidence contains a duplicate bucket/row/seed tuple"
    );

    assert!(
        evidence
            .iter()
            .all(|cell| cell.tail_motion.is_finite() && cell.tail_motion > 0.0),
        "every recorded S18 cell must retain its positive finite tail-motion freeze measurement"
    );
    let min_tail = evidence
        .iter()
        .map(|cell| cell.tail_motion)
        .fold(f64::INFINITY, f64::min);
    let max_tail = evidence
        .iter()
        .map(|cell| cell.tail_motion)
        .fold(f64::NEG_INFINITY, f64::max);
    let documented = format!(
        "tail motion {min_tail:.1}–{max_tail:.1}/255 per frame across all {} recorded cells",
        evidence.len()
    );
    let prose = include_str!("../src/t2v.rs")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        prose.contains(&documented),
        "src/t2v.rs must derive its freeze-evidence count/range from the recorded sweep: expected \
         `{documented}`"
    );

    let row_e = evidence
        .iter()
        .find(|cell| cell.bucket == "640x384" && cell.row == 'E')
        .expect("the 640x384 global reference row is committed");
    let row_e_gib = row_e.peak_bytes as f64 / 1024_f64.powi(3);
    let row_e_doc = format!("{row_e_gib:.2} GiB MLX peak");
    let recorded_prose = include_str!("generate_smoke.rs")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let row_e_sentence = format!("Row E is `n = 1`: its {row_e_doc}");
    assert!(
        recorded_prose.contains(&row_e_sentence),
        "the recorded-sweep prose must state Row E memory in binary GiB from its recorded byte \
         count: expected `{row_e_sentence}`"
    );
}
