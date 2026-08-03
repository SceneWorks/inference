//! sc-16956 — candle **PuLID-FLUX** per-step latent preview real-weight validation (epic 16948).
//!
//! PuLID contributes no fit, no projector and no VAE: it composes `candle-gen-flux`'s own FLUX.1-dev
//! backbone, so the latent it integrates, the `flux::sampling::unpack` that recovers it and the
//! `AutoencoderKL` that decodes it are literally the registered `flux1_dev` route's — which is why
//! `crate::preview` re-exports `candle_gen_flux::preview` and why there is **no VAE provenance row
//! here**: `candle-gen-flux/tests/preview_real_weights.rs` already measures every container this
//! backbone can load.
//!
//! What has to be established here is the thing that is genuinely PuLID's:
//!
//! 1. **The frames develop on an identity-conditioned render.**
//!    [`pulid_preview_frames_evolve_toward_the_final_image`] drives the real stack (FLUX.1-dev +
//!    `guozinan/PuLID` + the converted EVA02-CLIP-L-336 + the native SCRFD/ArcFace/BiSeNet face dir)
//!    with a live sink, checks numbering, checks seeded byte-identity against an inert render, and
//!    measures that every frame is closer to — and more like — the finished image than the one before.
//! 2. **The identity embedding never perturbs what is previewed.** The 32-token `id_embedding` is
//!    injected *inside* the DiT forward, so the sampler's latent stays the `[1, S, 64]` image sequence.
//!    Structurally that is closed by construction (`candle-gen-flux`'s
//!    `injected_conditioning_never_reaches_the_previewed_latent` drives the shape through the real
//!    sampler); measured, it shows up here as frames at exactly native-VAE-latent resolution — the
//!    packed-layout contract rejects anything carrying extra tokens, so a leak would emit **no frames
//!    at all** rather than wrong ones.
//! 3. **One frame per OUTER step on a multi-eval solver.**
//!    [`a_multi_eval_solver_emits_one_frame_per_outer_step`] runs `heun` and proves the guard is
//!    non-vacuous first, by asserting the evaluation count exceeds the step count.
//!
//! ```sh
//! $env:PULID_PREVIEW_FLUX_BASE = "<SceneWorks/flux1-dev-mlx snapshot>\q4"
//! $env:PULID_PREVIEW_WEIGHTS   = "<guozinan/PuLID>\pulid_flux_v0.9.1.safetensors"
//! $env:PULID_PREVIEW_EVA       = "<SceneWorks/pulid-flux-mlx>\eva02_clip_l_336.safetensors"
//! $env:PULID_PREVIEW_FACE_DIR  = "<dir: scrfd_10g + arcface_iresnet100 + bisenet_parsing>"
//! $env:PULID_PREVIEW_REF       = "<reference face .ppm (P6)>"
//! $env:PULID_PREVIEW_ARTIFACT_DIR = "E:\out\sc-16956"
//! cargo test -p candle-gen-pulid --release --features cuda --test preview_real_weights \
//!   -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{Image, PreviewFrame, PreviewSink, Progress};

use candle_gen_pulid::{PulidFlux, PulidFluxPaths, PulidFluxRequest};

const PROMPT: &str =
    "portrait of a person, color photo, cinematic lighting, sharp focus, high detail";
const SEED: u64 = 16956;

fn env_path(name: &str) -> PathBuf {
    std::env::var(name).map(PathBuf::from).unwrap_or_else(|_| {
        panic!(
            "{name} must be set for this row — skipping it would report success while proving \
             nothing"
        )
    })
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn artifact_dir() -> PathBuf {
    std::env::var("PULID_PREVIEW_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pulid_preview_sc16956"))
}

// ── Frame analysis ────────────────────────────────────────────────────────────────────────────────

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared buffers must match in length");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn downsample_raw(pixels: &[u8], src_w: u32, src_h: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = (x as u64 * src_w as u64 / w as u64) as u32;
            let sy = (y as u64 * src_h as u64 / h as u64) as u32;
            let idx = ((sy * src_w + sx) * 3) as usize;
            out.extend_from_slice(&pixels[idx..idx + 3]);
        }
    }
    out
}

fn downsample(img: &Image, w: u32, h: u32) -> Vec<u8> {
    downsample_raw(&img.pixels, img.width, img.height, w, h)
}

fn save_png(dir: &Path, pixels: &[u8], width: u32, height: u32, name: &str) {
    std::fs::create_dir_all(dir).expect("create the artifact dir");
    let path = dir.join(name);
    let buf: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .expect("frame buffer matches its dimensions");
    buf.save(&path)
        .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
    eprintln!("  wrote {}", path.display());
}

fn save_strip(dir: &Path, frames: &[PreviewFrame], name: &str) {
    assert!(!frames.is_empty(), "an empty strip cannot be written");
    let (fw, fh) = (frames[0].image.width, frames[0].image.height);
    let total_w = fw * frames.len() as u32;
    let mut sheet = vec![0u8; (total_w * fh * 3) as usize];
    for (i, frame) in frames.iter().enumerate() {
        let x0 = i as u32 * fw;
        for y in 0..fh {
            for x in 0..fw {
                let src = ((y * fw + x) * 3) as usize;
                let dst = (((y * total_w) + x0 + x) * 3) as usize;
                sheet[dst..dst + 3].copy_from_slice(&frame.image.pixels[src..src + 3]);
            }
        }
    }
    save_png(dir, &sheet, total_w, fh, name);
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

const MIN_FRAME_MOVEMENT: f64 = 0.1;
const MIN_DISTANCE_FALL: f64 = 0.25;
/// The terminal previewed step must still carry at least this share of the strip's PEAK movement. It
/// replaces a strict "the last step is the largest" assertion, which measures the model rather than the
/// wiring - see the comment at the assertion.
const MIN_TERMINAL_SHARE: f64 = 0.5;

/// The shared strip analysis — the same measurements `candle-gen-flux`'s harness applies to the
/// registered FLUX.1 lane, so the identity lane cannot be closed with a weaker one.
///
/// The correlation ceiling is the FLUX.1 fit's **in-sample** R² `0.98224` → √ ≈ `0.991`, the
/// like-for-like statistic. `min_r_last` is a per-lane backstop: the hook emits BEFORE each solver
/// step, so the final advancement is never previewed.
#[allow(clippy::too_many_arguments)]
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    latent_w: u32,
    latent_h: u32,
    min_r_last: f64,
    min_acceleration: f64,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );
    // Native-latent resolution. This is also the identity-leak assertion: the packed-layout contract
    // rejects a sequence carrying anything but this render's image tokens, so had the 32-token
    // `id_embedding` joined the running latent there would be no frames at all.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (latent_w, latent_h),
            "{label}: frames must be native-VAE-latent resolution — an identity-perturbed latent \
             could not produce one"
        );
    }

    let movement: Vec<f64> = frames
        .windows(2)
        .map(|p| mean_abs_delta(&p[0].image.pixels, &p[1].image.pixels))
        .collect();
    for (pair, delta) in frames.windows(2).zip(&movement) {
        eprintln!(
            "  {label} frame {:>2} → {:>2}: mean |Δ| {delta:.3}",
            pair[0].current, pair[1].current
        );
    }

    let target = downsample(final_image, latent_w, latent_h);
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

    let coarse = 16u32;
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

    assert!(
        movement.iter().all(|d| *d > MIN_FRAME_MOVEMENT),
        "{label}: some consecutive frames are effectively identical: {movement:?}"
    );
    // Frame-to-frame movement ACCELERATES through the strip - the flow-match time-shifted schedule's
    // signature, and a far stronger statement than a flat floor: a hook reading a stale, duplicated or
    // wrongly scaled latent would not reproduce it.
    //
    // Two exclusions, both measured rather than assumed. The OPENING frames are near-pure noise
    // projected through a global linear map, so the mean |delta| between two of them carries sampling
    // noise comparable to the sigma step itself - hence the second half rather than the whole strip.
    // And the TERMINAL pair is excluded because whether it rises is a property of the model, not of the
    // wiring: on the same nominal 1024^2 x 12-step flow schedule, FLUX.1-dev rises into it
    // (9.729 -> 12.288) while Chroma HD (9.496 -> 8.797) and PuLID (15.989 -> 14.507) dip. By the last
    // previewed step the latent is nearly converged, so the projection's mean |delta| saturates even as
    // the sigma interval grows. Asserting it would be asserting the model.
    //
    // What replaces it is a floor on the terminal step as a share of the strip's PEAK movement, so a
    // genuine collapse - a hook that froze, or one projecting a stale latent - still fails.
    let rising = &movement[..movement.len() - 1];
    let back_half = &rising[rising.len() / 2..];
    assert!(
        back_half.windows(2).all(|p| p[1] > p[0]),
        "{label}: movement must rise monotonically over the second half of the strip, up to but not \
         including the terminal pair: {movement:?}"
    );
    let (opening, closing) = (movement[0], movement[movement.len() - 1]);
    let peak = movement.iter().copied().fold(f64::MIN, f64::max);
    eprintln!(
        "  {label}: movement {opening:.3} -> {closing:.3} ({:.1}x), peak {peak:.3}",
        closing / opening
    );
    assert!(
        closing > opening * min_acceleration,
        "{label}: the terminal step must dominate the opening one by at least {min_acceleration}x \
         ({opening:.3} -> {closing:.3})"
    );
    assert!(
        closing > peak * MIN_TERMINAL_SHARE,
        "{label}: the terminal step must still carry at least {MIN_TERMINAL_SHARE} of the strip's \
         peak movement ({closing:.3} vs peak {peak:.3}) - below that the strip has stalled"
    );

    let (first, last) = (distances[0], distances[distances.len() - 1]);
    let fall = (first - last) / first;
    eprintln!(
        "  {label}: distance fell {:.1}% ({first:.2} → {last:.2})",
        fall * 100.0
    );
    assert!(
        fall > MIN_DISTANCE_FALL,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2}, \
         fall {fall:.3}, floor {MIN_DISTANCE_FALL})"
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    assert!(
        r_last > min_r_last,
        "{label}: the last preview frame must resemble the finished render \
         (r {r_last:+.3}, floor {min_r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    assert!(
        r_first < 0.75,
        "{label}: the first frame is pre-denoise noise and must not already BE the render \
         (r {r_first:+.3})"
    );
    assert!(
        r_last - r_first > 0.30,
        "{label}: resemblance must actually develop across the strip \
         (first {r_first:+.3} → last {r_last:+.3})"
    );
}

// ── Driving the identity lane ─────────────────────────────────────────────────────────────────────

fn base_request(steps: usize, size: u32, sampler: Option<&str>) -> PulidFluxRequest {
    PulidFluxRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        steps,
        guidance: 4.0,
        id_weight: 1.0,
        sampler: sampler.map(str::to_string),
        scheduler: None,
        seed: SEED,
        use_pid: false,
        preview: PreviewSink::default(),
        cancel: CancelFlag::new(),
    }
}

fn assert_pulid_previews_converge(
    label: &str,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
    min_r_last: f64,
    min_acceleration: f64,
) -> usize {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");
    let paths = PulidFluxPaths {
        flux_base: env_path("PULID_PREVIEW_FLUX_BASE"),
        pulid_weights: env_path("PULID_PREVIEW_WEIGHTS"),
        eva_weights: env_path("PULID_PREVIEW_EVA"),
        face_dir: env_path("PULID_PREVIEW_FACE_DIR"),
    };
    let model = PulidFlux::load(&paths).expect("PulidFlux::load");
    let reference = candle_gen::testkit::read_ppm(&env_path("PULID_PREVIEW_REF"));

    let mut noop = |_: Progress| {};
    let inert = model
        .generate(
            &base_request(steps as usize, size, sampler),
            &reference,
            &mut noop,
        )
        .unwrap_or_else(|e| panic!("{label}: inert render: {e}"));

    let (sink, frames) = collecting_sink();
    let mut request = base_request(steps as usize, size, sampler);
    request.preview = sink;
    let mut events = 0usize;
    let mut count_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            events += 1;
        }
    };
    let live = model
        .generate(&request, &reference, &mut count_progress)
        .unwrap_or_else(|e| panic!("{label}: live render: {e}"));

    assert_eq!(
        inert.pixels, live.pixels,
        "{label}: attaching a live preview sink changed the seeded render"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    let dir = artifact_dir();
    assert_the_strip_converges(
        label,
        &frames,
        &live,
        steps,
        size / 8,
        size / 8,
        min_r_last,
        min_acceleration,
    );
    save_strip(&dir, &frames, &format!("{label}-strip.png"));
    save_png(
        &dir,
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
    events
}

/// The identity lane's shipped default: the native flow-match Euler path over FLUX.1-dev's
/// time-shifted schedule, with the 20 PuLID CA modules injecting inside every forward.
#[test]
#[ignore = "needs PULID_PREVIEW_* + a CUDA GPU; run with --features cuda --release --ignored"]
fn pulid_preview_frames_evolve_toward_the_final_image() {
    let steps = env_u32("PULID_PREVIEW_STEPS", 12);
    let size = env_u32("PULID_PREVIEW_SIZE", 1024);
    // 0.90 against a measured +0.962 on this lane - 97% of the fit-derived 0.991 ceiling.
    assert_pulid_previews_converge("pulid-flux-euler", None, steps, size, 0.90, 2.0);
}

/// Exactly one frame per **outer** solver step on a multi-eval solver — PuLID threads the curated
/// `sampler` knob straight through to the shared driver, so `heun` really does re-run the whole
/// injected forward twice per step. The non-vacuity inequality is asserted first.
#[test]
#[ignore = "needs PULID_PREVIEW_* + a CUDA GPU; run with --features cuda --release --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    // 0.90 against a measured +0.952 on this 8-step 768 lane - 96% of the fit-derived 0.991 ceiling,
    // and level with the two sibling heun lanes (+0.961 flux, +0.957 chroma, both floored at 0.90).
    let events =
        assert_pulid_previews_converge("pulid-flux-heun", Some("heun"), steps, 768, 0.90, 1.5);
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
}
