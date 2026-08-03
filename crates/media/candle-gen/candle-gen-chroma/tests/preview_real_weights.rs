//! sc-16956 — candle **Chroma** per-step latent preview real-weight validation (epic 16948).
//!
//! Chroma contributes no fit and no projector: it denoises the FLUX.1 16-channel latent space in the
//! same packed token layout, recovers it with the same `flux::sampling::unpack`, and decodes it through
//! a VAE that is **byte-identical** to `black-forest-labs/FLUX.1-dev`'s — so `crate::preview`
//! re-exports `candle_gen_flux::preview`. Two things still have to be established *here*, and neither
//! transfers from the FLUX.1 rows:
//!
//! 1. **The VAE this snapshot loads is the one the fit was measured over.**
//!    [`the_chroma_snapshot_ships_the_flux1_vae`] is the reuse gate, run against the snapshot the
//!    render below actually uses rather than against a list of files. It is a hash equality, which is
//!    the strongest statement available: these really are one file republished, and the tensor-level
//!    machinery `candle-gen-flux` needs for its q4/BFL containers would prove *less* here.
//! 2. **The frames actually develop on a Chroma render.**
//!    [`chroma_preview_frames_evolve_toward_the_final_image`] drives the registered route through the
//!    `Generator` seam with a live sink, checks numbering, checks seeded byte-identity against an inert
//!    render, and measures that every frame is closer to — and more like — the finished image than the
//!    one before it. The strip is written out for review.
//!
//! Chroma is also the family member that runs **true CFG**, so this is where the epic's "CFG previews
//! must never project the fused unconditional half" criterion is exercised on real weights: a fused
//! `[2, …]` latent fails the packed-layout contract outright, so a strip that exists at all is proof
//! the preview only ever saw the conditional trajectory.
//!
//! ```sh
//! CHROMA_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--chroma1-hd-mlx\snapshots\<rev>\q4 \
//! CHROMA_PREVIEW_ARTIFACT_DIR=E:\out\sc-16956 \
//!   cargo test -p candle-gen-chroma --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};

const PROMPT: &str =
    "A weathered lighthouse on a rocky headland at golden hour, warm sunlight, dramatic clouds, \
     highly detailed photograph.";
const SEED: u64 = 16956;

/// The SHA-256 of the 16-channel `AutoencoderKL` every Chroma re-host publishes — 167,666,902 bytes,
/// 244 bf16 tensors.
///
/// Byte-identical to `black-forest-labs/FLUX.1-dev` @ `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` and
/// `black-forest-labs/FLUX.1-schnell` @ `741f7c3ce8b383c54771c7003378a50191e9efe9`, across Chroma HD
/// @ `9d99afe1ebca67032476756bc70d4a7152bc1bd5`, Base @ `e7330dda29d00ffdeeb719b28e92ee74cff0884c` and
/// Flash @ `6a9cb6178709559461506bf247f708d0d1008d00`. Four repos, one file — which is why Chroma may
/// reuse the epic-16624 FLUX.1 fit.
const FLUX1_VAE_SHA256: &str = "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn required_path(name: &str) -> PathBuf {
    env_path(name).unwrap_or_else(|| {
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
    env_path("CHROMA_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("chroma_preview_sc16956"))
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// **The reuse gate**, taken from the very snapshot the render below loads rather than from a list of
/// paths: the file this Chroma tier decodes through is the FLUX.1 VAE, byte for byte.
#[test]
#[ignore = "needs CHROMA_PREVIEW_SNAPSHOT; run with --ignored"]
fn the_chroma_snapshot_ships_the_flux1_vae() {
    let vae = required_path("CHROMA_PREVIEW_SNAPSHOT")
        .join("vae")
        .join("diffusion_pytorch_model.safetensors");
    assert!(
        vae.is_file(),
        "{} is missing — the snapshot under test does not carry the VAE this row measures",
        vae.display()
    );
    let sha = sha256_of(&vae);
    eprintln!("  chroma vae: {sha}  {}", vae.display());
    assert_eq!(
        sha, FLUX1_VAE_SHA256,
        "this Chroma snapshot's VAE is not the FLUX.1 one — the reused 16-channel fit would then \
         describe a different latent space and must not ship"
    );
    assert_eq!(std::fs::metadata(&vae).expect("stat").len(), 167_666_902);
    assert_eq!(crate_preview_channels(), 16);
}

/// Read through Chroma's own `preview` re-export, so a re-point to another family's projector fails
/// this row rather than only the crate's unit tests.
fn crate_preview_channels() -> usize {
    candle_gen_chroma::preview::PREVIEW_LATENT_CHANNELS
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

fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1, "these rows render a single image");
    images.pop().expect("one image")
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

/// Consecutive frames must not be the same picture — a low floor, because the first pair of a flow
/// strip is the smallest step of the whole trajectory. The strong statement is the acceleration below.
const MIN_FRAME_MOVEMENT: f64 = 0.1;
/// The strip must close a meaningful share of its distance to the finished image, expressed as a
/// fraction of the distance travelled rather than as a ratio of the endpoints (which would measure the
/// fit's irreducible residual as much as the convergence).
const MIN_DISTANCE_FALL: f64 = 0.25;
/// The terminal previewed step must still carry at least this share of the strip's PEAK movement. It
/// replaces a strict "the last step is the largest" assertion, which measures the model rather than the
/// wiring - see the comment at the assertion.
const MIN_TERMINAL_SHARE: f64 = 0.5;

/// The shared strip analysis — the same measurements `candle-gen-flux`'s harness applies to the FLUX.1
/// lane, so neither family can be closed with a weaker one.
///
/// The correlation ceiling is the FLUX.1 fit's **in-sample** R² `0.98224` → √ ≈ `0.991`, the
/// like-for-like statistic (sc-16954 was caught comparing an in-sample number against a holdout one).
/// `min_r_last` is a per-lane backstop rather than the ceiling: the hook emits BEFORE each solver step,
/// so the final advancement is never previewed, and how much of the trajectory that costs is a property
/// of the schedule.
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
    // Native-latent resolution and batch 1. Chroma runs TRUE CFG, so this is also the CFG assertion: a
    // fused `[2, …]` latent fails the packed-layout contract outright and would emit no frames at all.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (latent_w, latent_h),
            "{label}: frames must be native-VAE-latent resolution — a CFG-fused latent could not \
             produce one"
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

// ── Driving the registered route ──────────────────────────────────────────────────────────────────

fn base_request(steps: u32, size: u32, sampler: Option<&str>) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        sampler: sampler.map(str::to_string),
        ..GenerationRequest::default()
    }
}

fn assert_chroma_previews_converge(
    label: &str,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
    min_r_last: f64,
    min_acceleration: f64,
) -> (usize, Vec<PreviewFrame>) {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");
    let spec = LoadSpec::new(WeightsSource::Dir(required_path("CHROMA_PREVIEW_SNAPSHOT")));
    let generator =
        candle_gen_chroma::load_hd(&spec).unwrap_or_else(|e| panic!("load chroma: {e}"));

    let mut noop = |_: Progress| {};
    let inert = one_image(
        generator
            .generate(&base_request(steps, size, sampler), &mut noop)
            .unwrap_or_else(|e| panic!("{label}: inert render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let mut request = base_request(steps, size, sampler);
    request.preview = sink;
    let mut events = 0usize;
    let mut count_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            events += 1;
        }
    };
    let live = one_image(
        generator
            .generate(&request, &mut count_progress)
            .unwrap_or_else(|e| panic!("{label}: live render: {e}")),
    );

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
    (events, frames)
}

/// The registered route's shipped lane, with Chroma's true-CFG blend live inside the predict closure.
#[test]
#[ignore = "needs CHROMA_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn chroma_preview_frames_evolve_toward_the_final_image() {
    let steps = env_u32("CHROMA_PREVIEW_STEPS", 12);
    let size = env_u32("CHROMA_PREVIEW_SIZE", 1024);
    // 0.75 against a measured +0.800 on this lane. Lower than FLUX.1's +0.970 on the same nominal
    // schedule length, and the reason is the schedule: Chroma HD's `linspace(1, 1/N)` under a static
    // shift of 3 leaves a large unpreviewed terminal step, exactly the effect sc-16955 measured on
    // FLUX.2 (+0.556 against a 0.874 ceiling). At 80.7% of this fit's 0.991 ceiling it is comfortably
    // ahead of that precedent; the load-bearing assertions are the monotonicities and the +0.30 rise.
    assert_chroma_previews_converge("chroma1-hd-euler", None, steps, size, 0.75, 1.5);
}

/// Exactly one frame per **outer** solver step on a multi-eval solver, with the non-vacuity inequality
/// asserted first: the driver calls `on_progress` once per *evaluation*, so a solver that silently fell
/// back to Euler would make "frames == steps" prove nothing.
#[test]
#[ignore = "needs CHROMA_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    let (events, _) =
        // 0.90 against a measured +0.957 on this 8-step 768 lane, which does not pay the 1024 lane's
        // terminal-step penalty.
        assert_chroma_previews_converge("chroma1-hd-heun", Some("heun"), steps, 768, 0.90, 1.2);
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
}
