//! Shared fixtures: the snapshot environment, synthetic frames, the memory sampler, the
//! coherence statistics and the artifact report.

use super::*;

// ---------------------------------------------------------------------------------------------
// Environment / fixtures
// ---------------------------------------------------------------------------------------------

pub(super) fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(super) fn env_opt_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Assert the snapshot is present and return its root. The DiT is the file that makes a directory a
/// Krea Realtime snapshot rather than a bare companion staging dir.
pub(super) fn require_snapshot() -> PathBuf {
    // Libtest uses a new thread per case but MLX's allocator cache is process-wide. Release the
    // preceding case's (up to 90 GiB) decode scratch before loading another model, just as the
    // per-decode peak measurements below do. Live arrays are unaffected.
    mlx_rs::memory::clear_cache();
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
pub(super) fn gradient(w: usize, h: usize, phase: usize) -> Image {
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
pub(super) fn smooth_frame(w: usize, h: usize, phase: usize) -> Image {
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
pub(super) fn write_ppm(path: &Path, img: &Image) {
    let mut buf = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    buf.extend_from_slice(&img.pixels);
    std::fs::write(path, buf).expect("write ppm");
}

pub(super) fn dump_frames(frames: &[Image], label: &str) {
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
pub(super) struct MemorySampler {
    pub(super) stop: Arc<AtomicBool>,
    pub(super) max_active: Arc<AtomicUsize>,
    pub(super) max_total: Arc<AtomicUsize>,
    pub(super) handle: Option<std::thread::JoinHandle<()>>,
}

impl MemorySampler {
    pub(super) fn start() -> Self {
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
    pub(super) fn snapshot(&self) -> (usize, usize) {
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
    pub(super) fn finish(mut self) -> (usize, usize) {
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

pub(super) const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

pub(super) fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

// ---------------------------------------------------------------------------------------------
// Coherence
// ---------------------------------------------------------------------------------------------

/// Per-frame statistics used by the coherence assertions.
pub(super) struct FrameStat {
    pub(super) min: u8,
    pub(super) max: u8,
    pub(super) mean: f64,
    /// Mean |Δ| against the previous frame — 0 means the frame is a byte-identical repeat.
    pub(super) delta: f64,
}

/// Mean absolute per-byte difference between two clips of the same geometry, in 0..255 units.
pub(super) fn mean_abs_delta(a: &[Image], b: &[Image]) -> f64 {
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
pub(super) fn mean_abs_delta_columns(a: &[Image], b: &[Image]) -> Vec<f64> {
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

pub(super) fn frame_stats(frames: &[Image]) -> Vec<FrameStat> {
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
pub(super) fn assert_coherent(frames: &[Image], w: usize, h: usize, label: &str) {
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

/// Highlight-clipping statistics, returned rather than printed: `(mean %, worst-frame %)` of pixels
/// whose brightest channel is >= 250. This is what a viewer sees blow out, and it is the metric that
/// separates a starved tile from a healthy one — so a test has to be able to assert on it (sc-15325).
pub(super) fn clip_stats(frames: &[Image]) -> (f64, f64) {
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
pub(super) fn report_artifacts(frames: &[Image], label: &str) {
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
