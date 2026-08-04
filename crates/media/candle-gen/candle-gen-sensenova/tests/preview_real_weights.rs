//! sc-16960 — candle SenseNova-U1 per-step **preview** real-weight validation (epic 16948, Tier 2).
//!
//! Four things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story — with a fifth this story alone has, because SenseNova drives no shared sampler and its
//! frames therefore have no driver-owned numbering to inherit.
//!
//! 1. **The frames actually develop, on both registered ids.**
//!    [`base_preview_frames_evolve_toward_the_final_image`] and
//!    [`fast_preview_frames_evolve_toward_the_final_image`] render through the registered `Generator`
//!    seam with a live sink and measure that each frame is closer to the finished image than the one
//!    before it. Every strip is written out for direct review. `sensenova_u1_8b` runs with **CFG
//!    4.0**, so the CFG lane is the one measured rather than assumed away.
//! 2. **Exactly one frame per outer step, proven against the bespoke loop's OWN counter.**
//!    [`assert_the_strip_converges`] does not merely count frames: the sink and the progress callback
//!    write into **one** event log, and the log must read `Frame(1) Step(1) Frame(2) Step(2) …`. The
//!    loop's own `Progress::Step` sequence is the counter being compared against, and the interleave
//!    additionally pins that each frame is emitted **before** the step it precedes — the contract the
//!    shared drivers and Ideogram's bespoke loop both hold to.
//! 3. **An inert sink is byte-identical.** Each row renders twice on one warmed generator at the same
//!    seed and compares output bytes.
//! 4. **The state at the emission point is what the projector assumes.** The frames come back at the
//!    **token grid** `H/cell × W/cell` — verified from the emitted frames rather than from prose —
//!    which is only possible if the running state really is `[1, 3, H, W]`.
//! 5. **The right first-frame statistic.** SenseNova pools 32×32 = 1024 independent noise samples per
//!    preview pixel, so its first frame is near-flat grey and its *rail-clipped fraction is ~0* where
//!    a VAE family's would be large. Rail-clipping is therefore not the discriminating statistic here
//!    (sc-16959's finding, in a new shape): [`assert_the_strip_converges`] reports the rail-clipped
//!    fraction **and** measures **contrast about the fit's own intercept**, which is what actually
//!    separates a live projection from a mis-scaled one.
//!
//! ```sh
//! SENSENOVA_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--sensenova-u1-8b-mlx\snapshots\<rev>\q8 \
//! SENSENOVA_PREVIEW_FAST_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--sensenova-u1-8b-fast-mlx\snapshots\<rev>\bf16 \
//! SENSENOVA_PREVIEW_ARTIFACT_DIR=E:\out\sc-16960 \
//!   cargo test -p candle-gen-sensenova --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.
//!
//! [`the_projector_is_the_models_own_decode_at_token_resolution`] is the **only non-`#[ignore]`d row
//! in this file** and runs on the committed constants alone — it is the row that must appear in a
//! plain `cargo test` of this file. sc-16954 shipped a red row that hid because the only non-ignored
//! row in its file was excluded by `-- --ignored`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};
use candle_gen_sensenova::preview::project_running_image;

const PROMPT: &str =
    "a lighthouse on a rocky headland at sunset, orange sky, deep blue sea, long shadows";
const SEED: u64 = 16960;

/// The shipped 8B-MoT token cell: `patch_size 16 · merge_size 2`. Read back off the emitted frames
/// rather than trusted — see [`assert_the_strip_converges`].
const CELL: u32 = 32;

/// An input a row cannot run without. Missing means **fail**, not skip.
fn required_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| {
        panic!("{name} must be set — this row validates a shipped route and cannot be skipped")
    });
    let path = PathBuf::from(value);
    assert!(
        path.exists(),
        "{name} points at {} — not found",
        path.display()
    );
    path
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must parse as an integer, got {value:?}"))
        })
        .unwrap_or(fallback)
}

fn artifact_dir() -> PathBuf {
    required_path("SENSENOVA_PREVIEW_ARTIFACT_DIR")
}

/// One entry in the shared render event log: the two callbacks the bespoke loop drives, in the order
/// it drives them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    /// A preview frame: `(current, total)`.
    Frame(u32, u32),
    /// A `Progress::Step`: `(current, total)` — the loop's OWN counter.
    Step(u32, u32),
}

fn save_png(pixels: &[u8], width: u32, height: u32, name: &str) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(&path, pixels, width, height, image::ExtendedColorType::Rgb8)
        .expect("save a PNG");
    eprintln!("  saved {}", path.display());
}

/// Write the frames side by side as one strip, plus each frame individually — the artifact the epic
/// asks to be reviewed directly. One strip **per route**.
fn save_strip(frames: &[PreviewFrame], name: &str) {
    let (w, h) = (
        frames[0].image.width as usize,
        frames[0].image.height as usize,
    );
    let strip_w = w * frames.len();
    let mut strip = vec![0u8; strip_w * h * 3];
    for (i, frame) in frames.iter().enumerate() {
        for y in 0..h {
            let src = &frame.image.pixels[y * w * 3..(y + 1) * w * 3];
            let x0 = (y * strip_w + i * w) * 3;
            strip[x0..x0 + w * 3].copy_from_slice(src);
        }
        save_png(
            &frame.image.pixels,
            frame.image.width,
            frame.image.height,
            &format!("{name}_frame{:02}", frame.current),
        );
    }
    save_png(&strip, strip_w as u32, h as u32, &format!("{name}_strip"));
}

fn downsample_raw(pixels: &[u8], width: u32, height: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let (y0, y1) = (
                y * height / h,
                ((y + 1) * height / h).max(y * height / h + 1),
            );
            let (x0, x1) = (x * width / w, ((x + 1) * width / w).max(x * width / w + 1));
            for channel in 0..3usize {
                let mut sum = 0u32;
                let mut count = 0u32;
                for sy in y0..y1.min(height) {
                    for sx in x0..x1.min(width) {
                        sum += pixels[((sy * width + sx) as usize) * 3 + channel] as u32;
                        count += 1;
                    }
                }
                out[((y * w + x) as usize) * 3 + channel] = (sum / count.max(1)) as u8;
            }
        }
    }
    out
}

fn downsample(img: &Image, w: u32, h: u32) -> Vec<u8> {
    downsample_raw(&img.pixels, img.width, img.height, w, h)
}

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / a.len() as f64
}

fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (mean_a, mean_b) = (
        a.iter().map(|v| *v as f64).sum::<f64>() / n,
        b.iter().map(|v| *v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (*x as f64 - mean_a, *y as f64 - mean_b);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if va <= f64::EPSILON || vb <= f64::EPSILON {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// The fraction of an RGB8 frame sitting on a rail (0 or 255).
fn rail_clipped_fraction(image: &Image) -> f64 {
    image
        .pixels
        .iter()
        .filter(|value| **value == 0 || **value == 255)
        .count() as f64
        / image.pixels.len() as f64
}

/// Mean absolute distance from the fit's own intercept — "how far this frame gets from the flat grey
/// a fully-zero state projects to".
///
/// **This, not the rail-clipped fraction, is SenseNova's discriminating first-frame statistic.**
/// Pooling one token cell averages 1024 independent prior samples, dividing the prior's standard
/// deviation by 32, so the first frame lands near the intercept and clips nothing. A projection that
/// silently lost its gain would also land near the intercept — but so would nothing else, which is
/// what makes the *rise* in this statistic across a strip meaningful where a rail count is not.
fn contrast_about_intercept(image: &Image) -> f64 {
    let intercept = project_running_image(
        &Tensor::zeros(
            (1, 3, CELL as usize, CELL as usize),
            DType::F32,
            &Device::Cpu,
        )
        .expect("zero state"),
        CELL as usize,
    )
    .expect("the intercept frame");
    let grey = intercept.pixels[0];
    image
        .pixels
        .iter()
        .map(|value| (*value as i32 - grey as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / image.pixels.len() as f64
}

/// Per-lane development criteria, each carrying **its own** measured numbers.
///
/// The headroom is uniform and stated: **0.03 under a measured correlation, 0.06 under a measured
/// rise** (a rise differences two correlations, so it carries the 0.03 allowance of each), **0.06 over
/// a measured distance ratio** rounded up to two decimals, and **±20% on a measured contrast**. No
/// bound is justified by the other lane's measurement.
///
/// Two exceptions, both deliberate and both stated rather than hidden:
///
/// * [`MAX_R_FIRST`] is **shared and deliberately loose** — see its own docs.
/// * `max_first_rail_clipped` is bounded at a small constant against a measurement of *zero*, which
///   is the point: SenseNova's **unpooled** prior at 512² is `N(0,1)·2.0`, whose decode is
///   `clamp(z + 0.5)` and therefore rails on `2·Φ(−0.5) ≈ 62%` of its pixels. This ceiling is what
///   proves the token-cell pool is actually happening.
struct Develops {
    /// Floor under the measured final-frame correlation with the finished render.
    min_r_last: f64,
    /// Ceiling over the measured first-frame correlation — "it did not start as the render".
    max_r_first: f64,
    /// Floor under the measured `r_last − r_first` rise.
    min_rise: f64,
    /// Ceiling over the measured `last / first` mean-|Δ|-to-final ratio — "it converged".
    max_distance_ratio: f64,
    /// Ceiling over the measured FIRST-frame contrast about the intercept — the pooled prior is
    /// near-flat, and this is the bound that says so numerically.
    max_first_contrast: f64,
    /// Floor under the measured LAST-frame contrast about the intercept — a projection that lost its
    /// gain would stay flat all the way through.
    min_last_contrast: f64,
    /// Ceiling over the measured FIRST-frame rail-clipped fraction. Reported and bounded because it
    /// is the statistic the epic's earlier stories used, and stating it is how this story records
    /// that it does **not** discriminate here.
    max_first_rail_clipped: f64,
}

/// The shared "did not start as the render" ceiling, and the one bound here that is **not** derived
/// from a lane's own measurement.
///
/// It is deliberately loose. Both lanes open on the *pooled prior*, a near-flat 16×16 frame whose
/// correlation with the render is essentially arbitrary — measured **+0.234** (base) and **+0.260**
/// (fast), from the same seeded noise, and a bound tightened onto either would be reading a coin
/// flip. This is the same carve-out sc-16959 recorded for SANA's `max_r_first`, for the same reason:
/// a tight `r_first` bound reads a fit's own intercept as if it were resemblance. What the strip has
/// to prove is the *rise*, and that bound is per lane.
const MAX_R_FIRST: f64 = 0.60;

/// `sensenova_u1_8b` — the base id, **true CFG 4.0**, 8 flow-match Euler steps at 512², shift 3.0.
///
/// Measured: r **+0.234 → +0.999** (rise +0.765); mean |Δ| to final **104.04 → 28.63** (ratio 0.275);
/// contrast about the intercept **6.33 → 76.05**; rail-clipped fraction **0.0000 → 0.0000**.
const BASE: Develops = Develops {
    // 0.999 − 0.03.
    min_r_last: 0.969,
    max_r_first: MAX_R_FIRST,
    // 0.999 − 0.234 = 0.765; 0.765 − 0.06 = 0.705.
    min_rise: 0.705,
    // 28.63 / 104.04 = 0.275; 0.275 + 0.06 = 0.335, rounded up to two decimals.
    max_distance_ratio: 0.34,
    // 6.33 × 1.2. Identical to the fast lane's by construction — frame 1 is the seeded prior, emitted
    // before any model forward, and both lanes share seed 16960 at 512².
    max_first_contrast: 7.6,
    // 76.05 × 0.8. The guided lane travels further from the intercept than the distilled one.
    min_last_contrast: 60.8,
    // Measured 0.0000. Non-vacuous: the UNPOOLED prior here decodes to clamp(z + 0.5) and rails on
    // ≈62% of its pixels, so this ceiling fails the moment the token-cell pool stops happening.
    max_first_rail_clipped: 0.02,
};

/// `sensenova_u1_8b_fast` — the 8-step distilled id at **CFG 1.0** (guidance off), 512², shift 3.0.
///
/// Measured: r **+0.260 → +1.000** (rise +0.740); mean |Δ| to final **62.11 → 18.45** (ratio 0.297);
/// contrast about the intercept **6.33 → 44.72**; rail-clipped fraction **0.0000 → 0.0000**.
///
/// Its own numbers, not the base lane's. The distilled lane starts *closer* in absolute distance
/// (62.11 against 104.04, because CFG 1.0 does not push the trajectory as far) and finishes closer
/// still (18.45 against 28.63), while ending at a lower contrast about the intercept (44.72 against
/// 76.05) — a less saturated render, not a weaker projection.
const FAST: Develops = Develops {
    // 1.000 − 0.03.
    min_r_last: 0.970,
    max_r_first: MAX_R_FIRST,
    // 1.000 − 0.260 = 0.740; 0.740 − 0.06 = 0.680.
    min_rise: 0.680,
    // 18.45 / 62.11 = 0.297; 0.297 + 0.06 = 0.357, rounded up to two decimals.
    max_distance_ratio: 0.36,
    // 6.33 × 1.2 — the same seeded prior as the base lane.
    max_first_contrast: 7.6,
    // 44.72 × 0.8.
    min_last_contrast: 35.7,
    // Measured 0.0000, and non-vacuous for the same reason as the base lane's.
    max_first_rail_clipped: 0.02,
};

/// The shared strip analysis, applied identically to both ids so neither can be closed with a weaker
/// measurement than the other.
fn assert_the_strip_converges(
    label: &str,
    events: &[Event],
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    size: u32,
    develops: &Develops,
) {
    // ── One frame per outer step, against the loop's OWN counter ──────────────────────────────────
    //
    // The bespoke loop emits the frame BEFORE the step and reports `Progress::Step` after it, once
    // each, so the single interleaved log must alternate exactly. This is a stronger statement than
    // "N frames arrived": it compares the preview numbering against the very counter the denoise
    // loop advances, and it pins the emit-before-step ordering at the same time.
    let expected: Vec<Event> = (1..=steps)
        .flat_map(|n| [Event::Frame(n, steps), Event::Step(n, steps)])
        .collect();
    assert_eq!(
        events, expected,
        "{label}: the bespoke loop must emit exactly one frame per outer step, numbered against its \
         own Progress::Step counter, and emit each frame BEFORE the step it precedes"
    );

    // ── The state at the emission point ───────────────────────────────────────────────────────────
    //
    // The frames come back at the token grid, which is only possible if the running state really is
    // `[1, 3, H, W]` and the pool is the model's own `cell`.
    let edge = size / CELL;
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (edge, edge),
            "{label}: frames must be token-grid resolution ({size}/{CELL})"
        );
    }

    // Every frame must differ from its predecessor — N copies of one image would satisfy a naive
    // "N frames arrived" check while showing nothing developing.
    for pair in frames.windows(2) {
        let delta = mean_abs_delta(&pair[0].image.pixels, &pair[1].image.pixels);
        eprintln!(
            "  {label} frame {:>2} → {:>2}: mean |Δ| {delta:.2}",
            pair[0].current, pair[1].current
        );
        assert!(
            delta > 0.5,
            "{label}: frames {} and {} are effectively identical (mean |Δ| {delta:.3})",
            pair[0].current,
            pair[1].current
        );
    }

    // ── The first-frame statistics: the one that does not discriminate, and the one that does ─────
    let first_rail = rail_clipped_fraction(&frames[0].image);
    let last_rail = rail_clipped_fraction(&frames[frames.len() - 1].image);
    let contrasts: Vec<f64> = frames
        .iter()
        .map(|f| contrast_about_intercept(&f.image))
        .collect();
    for (frame, contrast) in frames.iter().zip(&contrasts) {
        eprintln!(
            "  {label} frame {:>2}: contrast about the fit intercept {contrast:.2}",
            frame.current
        );
    }
    let (first_contrast, last_contrast) = (contrasts[0], contrasts[contrasts.len() - 1]);
    eprintln!(
        "  {label} rail-clipped fraction: first {first_rail:.4} → last {last_rail:.4}  \
         (NOT discriminating here: the token-cell pool averages {} prior samples per preview pixel)",
        CELL * CELL
    );
    eprintln!(
        "  {label} contrast about the fit intercept: first {first_contrast:.2} → last \
         {last_contrast:.2}"
    );
    assert!(
        first_rail <= develops.max_first_rail_clipped,
        "{label}: the first frame rail-clipped {first_rail:.4} of its pixels, over the measured \
         ceiling {:.4}",
        develops.max_first_rail_clipped
    );
    assert!(
        first_contrast <= develops.max_first_contrast,
        "{label}: the first frame is not the near-flat pooled prior (contrast {first_contrast:.2}, \
         ceiling {:.2})",
        develops.max_first_contrast
    );
    assert!(
        last_contrast >= develops.min_last_contrast,
        "{label}: the last frame never left the intercept (contrast {last_contrast:.2}, floor \
         {:.2}) — a projection that lost its gain looks exactly like this",
        develops.min_last_contrast
    );

    // ── Convergence on the finished render ────────────────────────────────────────────────────────
    let target = downsample(final_image, edge, edge);
    let distances: Vec<f64> = frames
        .iter()
        .map(|f| mean_abs_delta(&f.image.pixels, &target))
        .collect();
    for (frame, distance) in frames.iter().zip(&distances) {
        eprintln!(
            "  {label} frame {:>2}: mean |Δ| to final {distance:.2}",
            frame.current
        );
    }
    let (first, last) = (distances[0], distances[distances.len() - 1]);
    let ratio = last / first;
    assert!(
        ratio < develops.max_distance_ratio,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2}, \
         ratio {ratio:.3}, ceiling {:.3})",
        develops.max_distance_ratio
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    // Correlation over a coarse thumbnail is what "the preview looks like the image" means for a
    // decorative frame: absolute distance can only ever say "closer".
    let coarse = 8u32;
    let coarse_target = downsample(final_image, coarse, coarse);
    let correlations: Vec<f64> = frames
        .iter()
        .map(|f| {
            correlation(
                &downsample_raw(
                    &f.image.pixels,
                    f.image.width,
                    f.image.height,
                    coarse,
                    coarse,
                ),
                &coarse_target,
            )
        })
        .collect();
    for (frame, r) in frames.iter().zip(&correlations) {
        eprintln!(
            "  {label} frame {:>2}: coarse correlation with final {r:+.3}",
            frame.current
        );
    }
    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    assert!(
        r_last > develops.min_r_last,
        "{label}: the last preview frame must resemble the finished render (r {r_last:+.3}, floor \
         {:+.3})",
        develops.min_r_last
    );
    assert!(
        r_first < develops.max_r_first,
        "{label}: the strip must not open on something that already IS the render (r {r_first:+.3}, \
         ceiling {:+.3})",
        develops.max_r_first
    );
    assert!(
        r_last - r_first > develops.min_rise,
        "{label}: resemblance must actually develop across the strip (first {r_first:+.3} → last \
         {r_last:+.3}, rise {:+.3}, floor {:+.3})",
        r_last - r_first,
        develops.min_rise
    );
    // Monotonicity is asserted separately because no pair of endpoint bounds implies it: a strip that
    // wandered and happened to end well would satisfy every bound above.
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    // The same for contrast: the frame has to keep leaving the intercept, not jump once and sit.
    assert!(
        contrasts.windows(2).all(|p| p[1] > p[0]),
        "{label}: contrast about the fit intercept must grow at every step: {contrasts:?}"
    );
}

fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1);
    images.pop().expect("one image")
}

fn request(steps: u32, size: u32, guidance: f32) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        guidance: Some(guidance),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        ..Default::default()
    }
}

/// Render one id twice on one warmed generator at the same seed — once with an inert sink, once with
/// a live one — and hold the strip to [`assert_the_strip_converges`].
fn render_and_assert(
    label: &str,
    id: &str,
    var: &str,
    steps: u32,
    size: u32,
    guidance: f32,
    develops: &Develops,
) {
    let root = required_path(var);
    eprintln!("── {label}: {size}² × {steps} steps, guidance {guidance}");

    let generator = candle_gen_sensenova::provider_registry()
        .expect("sensenova registry")
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap_or_else(|e| panic!("load {id}: {e}"));

    let base = request(steps, size, guidance);

    // Inert first: the byte-identity baseline, on the same warmed generator.
    let inert = one_image(
        generator
            .generate(&base, &mut |_| {})
            .unwrap_or_else(|e| panic!("{label} inert-sink render: {e}")),
    );

    // ONE log for both callbacks, so their interleaving is observable — that is what lets the frame
    // numbering be compared against the bespoke loop's OWN counter rather than merely counted.
    let log = Arc::new(Mutex::new(Vec::<Event>::new()));
    let frames = Arc::new(Mutex::new(Vec::<PreviewFrame>::new()));
    let sink_log = Arc::clone(&log);
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame: PreviewFrame| {
        candle_gen::lock_recover(&sink_log).push(Event::Frame(frame.current, frame.total));
        candle_gen::lock_recover(&captured).push(frame);
    });

    let progress_log = Arc::clone(&log);
    let active = one_image(
        generator
            .generate(
                &GenerationRequest {
                    preview: sink,
                    ..base
                },
                &mut |p| {
                    if let Progress::Step { current, total } = p {
                        candle_gen::lock_recover(&progress_log).push(Event::Step(current, total));
                    }
                },
            )
            .unwrap_or_else(|e| panic!("{label} active-sink render: {e}")),
    );

    assert_eq!(
        inert.pixels, active.pixels,
        "{label}: an active preview sink must not change a single output byte at the same seed"
    );

    let events = candle_gen::lock_recover(&log).clone();
    let frames = candle_gen::lock_recover(&frames).clone();
    let name = format!("{label}_{size}_s{steps}");
    save_strip(&frames, &name);
    save_png(
        &active.pixels,
        active.width,
        active.height,
        &format!("{name}_final"),
    );
    assert_the_strip_converges(label, &events, &frames, &active, steps, size, develops);
}

/// `sensenova_u1_8b` — the base id, rendered with **true CFG 4.0** so the guided lane is the one
/// measured. The unconditional pass is a second forward against a second KV cache inside the step
/// body, blended into one velocity before the state advances, so no fused unconditional half can
/// reach a frame — and the token-grid frame shape asserted above is what a fused `[2, …]` batch would
/// break.
#[test]
#[ignore = "needs SENSENOVA_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn base_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sensenova_u1_8b",
        candle_gen_sensenova::MODEL_ID,
        "SENSENOVA_PREVIEW_SNAPSHOT",
        env_u32("SENSENOVA_PREVIEW_STEPS", 8),
        env_u32("SENSENOVA_PREVIEW_SIZE", 512),
        4.0,
        &BASE,
    );
}

/// `sensenova_u1_8b_fast` — the second registered id, on its own snapshot, at its own CFG-1.0
/// default. Measured separately because both ids advertise the flag and a strip from one is not
/// evidence for the other.
#[test]
#[ignore = "needs SENSENOVA_PREVIEW_FAST_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn fast_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sensenova_u1_8b_fast",
        candle_gen_sensenova::MODEL_ID_FAST,
        "SENSENOVA_PREVIEW_FAST_SNAPSHOT",
        env_u32("SENSENOVA_PREVIEW_FAST_STEPS", 8),
        env_u32("SENSENOVA_PREVIEW_SIZE", 512),
        1.0,
        &FAST,
    );
}

/// **The committed fit reproduces the model's own decode**, at token resolution — weights-free, and
/// the only row in this file that a plain `cargo test` runs.
///
/// SenseNova has no VAE, so `tensor_to_image` (`x·0.5 + 0.5`, clamped, ×255, ties to even) is the
/// whole decode. This row builds a synthetic model-space state, pools it with the engine's own
/// `avg_pool2d`, decodes the pooled state with the engine's own `tensor_to_image`, and compares that
/// against the shipped projector's frame for the same state. Both go through shipped code; there is
/// no second implementation of the maths here.
///
/// They must agree to within **2** RGB8 levels — the analytic worst case the committed coefficients
/// admit over the model's `[-1, 1]` output range is under 1.2 levels
/// (`the_committed_fit_is_within_two_rgb8_levels_of_the_models_own_decode` derives it from the
/// constants), plus one for the final rounding. That difference is the fit's whole visual cost, and
/// it exists only because the fit's target was the *clamped* decode.
///
/// This is what the near-unity R² *means*, expressed without any weights: the preview is not an
/// approximation of a latent decode, it is the model's own decode at a coarser resolution.
#[test]
fn the_projector_is_the_models_own_decode_at_token_resolution() {
    let cell = 8usize;
    let (h, w) = (cell * 3, cell * 5);
    let n = 3 * h * w;
    // A deterministic non-constant state that stays inside the model's own [-1, 1] output range, so
    // the comparison isolates the committed coefficients rather than the clamp's non-commutation
    // with the pool (which is a property of the FIT's target, measured in tests/fit_preview_rgb.rs).
    let values: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.113).sin() * 0.85 + ((i % 11) as f32 - 5.0) * 0.02)
        .collect();
    assert!(
        values.iter().all(|v| v.abs() <= 1.0),
        "this row's state must stay inside the model's own output range"
    );
    assert!(
        values.iter().cloned().fold(f32::MIN, f32::max) > 0.9
            && values.iter().cloned().fold(f32::MAX, f32::min) < -0.9,
        "the state must still span most of that range or the comparison is near-trivial"
    );
    let state = Tensor::from_vec(values, (1, 3, h, w), &Device::Cpu).expect("state");

    // The engine's own pool + the engine's own decode — the reference the projector must reproduce.
    let reference = candle_gen_sensenova::tensor_to_image(
        &state
            .avg_pool2d(cell)
            .expect("the engine's own token-cell pool"),
    )
    .expect("the engine's own decode");

    let projected = project_running_image(&state, cell).expect("the shipped projector");
    assert_eq!(
        (projected.width, projected.height),
        ((w / cell) as u32, (h / cell) as u32)
    );
    assert_eq!(
        (reference.width, reference.height),
        (projected.width, projected.height)
    );

    let worst = projected
        .pixels
        .iter()
        .zip(&reference.pixels)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    let mean = mean_abs_delta(&projected.pixels, &reference.pixels);
    eprintln!(
        "projector vs the engine's own pooled decode: max RGB8 delta {worst}, mean |Δ| {mean:.4}"
    );
    assert!(
        worst <= 2,
        "the committed fit must reproduce the model's own decode at token resolution to within two \
         RGB8 levels; worst channel differs by {worst}"
    );
}
