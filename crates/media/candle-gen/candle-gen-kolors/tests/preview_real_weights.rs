//! sc-16954 — candle Kolors per-step latent **preview** real-weight validation (epic 16948).
//!
//! Kolors reuses the SDXL four-channel RGB fit and adds none of its own. **That reuse is a claim about
//! weights, so it needs its own render to stand** — running SDXL and calling Kolors covered would be
//! exactly the "matching Rust type" reasoning this epic rejects.
//!
//! Three rows, mirroring the SDXL harness so neither family can be closed with the weaker measurement:
//!
//! 1. [`the_kolors_vae_is_the_sdxl_fit_vae`] — the reuse gate. The VAE Kolors loads is **byte-identical**
//!    to the SDXL one the epic-16624 fit was measured against, so no tensor-by-tensor argument is
//!    needed. Hashes the file rather than trusting a comment.
//! 2. [`kolors_curated_preview_frames_evolve_toward_the_final_image`] and
//!    [`kolors_native_preview_frames_evolve_toward_the_final_image`] — both lanes of the registered
//!    route, each checked for numbering, seeded byte-identity against an inert render, per-frame
//!    movement, falling distance and rising resemblance. The native lane is the DEFAULT one and it
//!    drives no shared sampler, so wiring only the driver call would have left it dark.
//! 3. [`a_multi_eval_solver_emits_one_frame_per_outer_step`] — `heun`, with the evaluation count
//!    asserted to exceed the step count *first* so "frames == steps" means something.
//!
//! ```sh
//! KOLORS_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--Kwai-Kolors--Kolors-diffusers\snapshots\<rev> \
//! KOLORS_PREVIEW_ARTIFACT_DIR=E:\out\sc-16954 \
//!   cargo test -p candle-gen-kolors --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and a skipped gate is indistinguishable in a log from one that proved
//! something. Asking for `--ignored` is already the opt-in.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};

const PROMPT: &str =
    "A quiet teahouse courtyard at dusk with paper lanterns and a stone path, warm light, highly \
     detailed photograph.";
const NEGATIVE: &str = "lowres, blurry, deformed, watermark, text";
const SEED: u64 = 16954;

/// The SHA-256 of `vae/diffusion_pytorch_model.fp16.safetensors` — 167,335,342 bytes, the file
/// `Kwai-Kolors/Kolors-diffusers` @ `7e091c75199e910a26cd1b51ed52c28de5db3711` publishes and every
/// tier of `SceneWorks/kolors-mlx` mirrors verbatim.
///
/// **Byte-identical** to `stabilityai/stable-diffusion-xl-base-1.0`'s, which is the file the
/// epic-16624 four-channel fit was measured against, and the hash `mlx-gen-sdxl/src/preview.rs`
/// already cites as its Kolors grounding.
const SDXL_FAMILY_VAE_SHA256: &str =
    "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

/// An input a row cannot run without. Missing means **fail**, not skip.
fn required_path(name: &str) -> PathBuf {
    env_path(name).unwrap_or_else(|| {
        panic!(
            "{name} must be set for this row — skipping it would report success while proving \
             nothing"
        )
    })
}

fn artifact_dir() -> PathBuf {
    env_path("KOLORS_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("kolors_preview_sc16954"))
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

// ── Provenance ────────────────────────────────────────────────────────────────────────────────────

/// The reuse gate: Kolors' VAE **is** the SDXL fit VAE, byte for byte.
#[test]
#[ignore = "needs KOLORS_PREVIEW_SNAPSHOT; run with --ignored"]
fn the_kolors_vae_is_the_sdxl_fit_vae() {
    let vae_dir = required_path("KOLORS_PREVIEW_SNAPSHOT").join("vae");
    // The snapshot may publish the fp16 file, the bare diffusers name, or both; hash whichever the
    // loader would mmap. `crate::pipeline` loads every `.safetensors` in this dir, sorted.
    let files: Vec<PathBuf> = std::fs::read_dir(&vae_dir)
        .unwrap_or_else(|e| panic!("read {vae_dir:?}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    assert_eq!(
        files.len(),
        1,
        "expected exactly one VAE shard in {vae_dir:?}, found {files:?}"
    );

    let path = &files[0];
    let size = std::fs::metadata(path).expect("stat the vae").len();
    let digest = sha256_of(path);
    eprintln!("  kolors vae: {digest}  {size} bytes  {}", path.display());
    assert_eq!(size, 167_335_342, "kolors VAE size moved");
    assert_eq!(
        digest, SDXL_FAMILY_VAE_SHA256,
        "the Kolors VAE is no longer byte-identical to the SDXL VAE the epic-16624 four-channel fit \
         was measured against — the reuse in crate::preview no longer holds"
    );

    let config =
        std::fs::read_to_string(vae_dir.join("config.json")).expect("read vae/config.json");
    assert!(
        config.contains("\"latent_channels\": 4") && config.contains("\"scaling_factor\": 0.13025"),
        "kolors vae/config.json must still declare the four-channel 0.13025 space: {config}"
    );
    assert_eq!(candle_gen_kolors::preview::PREVIEW_LATENT_CHANNELS, 4);
}

// ── Frame analysis helpers (same measurements as the SDXL harness) ────────────────────────────────

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

fn save_png(pixels: &[u8], width: u32, height: u32, name: &str) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
    let path = dir.join(name);
    let buf: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .expect("frame buffer matches its dimensions");
    buf.save(&path)
        .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
    eprintln!("  wrote {}", path.display());
}

fn save_strip(frames: &[PreviewFrame], name: &str) {
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
    save_png(&sheet, total_w, fh, name);
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

fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    width: u32,
    height: u32,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    // Latent resolution and batch 1. A CFG-fused `[2, 4, h, w]` latent fails the `[1, 4, h, w]`
    // contract outright, so a strip that exists at all is proof the preview never saw the fused
    // unconditional half — there would be no frames if it had.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (width / 8, height / 8),
            "{label}: frames must be VAE-latent resolution"
        );
    }

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

    let (lw, lh) = (width / 8, height / 8);
    let target = downsample(final_image, lw, lh);
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
    assert!(
        last < first * 0.6,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2})"
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

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
    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    // The floor is derived from the FIT's own explanatory power, not tuned to an observation -- and
    // the two fits are compared on the SAME statistic, which is the part that is easy to get wrong.
    //
    // A projection cannot correlate with the decode better than the fit does. The 16-channel QwenVae
    // families wired earlier in this epic were held to 0.85 against a recorded R^2 of 0.9586, a
    // correlation ceiling of ~0.979 -- so 86.8% of their ceiling. That 0.9586 is an **in-sample** fit
    // R^2 over the whole 32,768-sample corpus (`mlx-gen-qwen-image/src/preview.rs:22`, and
    // `fit_preview_rgb.rs:83` treats it as one); no QwenVae holdout split exists to compare against.
    // The like-for-like number on this side is therefore the SDXL fit R^2, 0.91849, a ceiling of
    // ~0.9584 -- NOT its holdout 0.86065, which is a genuine out-of-sample measurement and would bias
    // the floor low by being matched against an unsplit one. The same 86.8% of 0.9584 is
    // 0.9584 * 0.85 / 0.979 = 0.832, so 0.83 is the matched floor.
    //
    // The hook also emits BEFORE each solver step (sc-16949), so the last frame is one advancement
    // short of the render -- the fully denoised state is never previewed, the finished image lands
    // instead. Measured last frames: SDXL curated +0.885, SDXL heun +0.887, Kolors curated +0.848,
    // Kolors native +0.852. This floor is the "the strip never got close" backstop; the
    // load-bearing assertions are the strictly monotone rise and fall around it.
    //
    // Both Kolors lanes run a full 12-step schedule, so neither needs the per-lane floor the SDXL
    // harness carries for its few-step Lightning lane.
    assert!(
        r_last > 0.83,
        "{label}: the last preview frame must resemble the finished render (r {r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    // A rise, not an absolute floor on the first frame: the fit's intercept is R > G > B, so a frame
    // of pre-denoise noise starts at a non-zero, scene-dependent correlation. sc-16950's
    // `r_first < 0.35` ceiling is deliberately not ported.
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
        negative_prompt: Some(NEGATIVE.into()),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        sampler: sampler.map(str::to_string),
        ..GenerationRequest::default()
    }
}

/// Render one lane twice on one warmed generator at the same seed — once inert, once live — and hold
/// the strip to [`assert_the_strip_converges`]. Returns the live run's progress-event count.
fn assert_lane_previews_converge(
    label: &str,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
) -> usize {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");
    let spec = LoadSpec::new(WeightsSource::Dir(required_path("KOLORS_PREVIEW_SNAPSHOT")));
    let generator = candle_gen_kolors::provider_registry()
        .expect("kolors registry")
        .load("kolors", &spec)
        .unwrap_or_else(|e| panic!("load kolors: {e}"));

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

    let frames = candle_gen::lock_recover(&frames);
    assert_the_strip_converges(label, &frames, &live, steps, size, size);
    save_strip(&frames, &format!("{label}-strip.png"));
    save_png(
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
    events
}

// ── The lanes ─────────────────────────────────────────────────────────────────────────────────────

/// The registered route's **default** lane: the bespoke `KolorsEulerSampler` leading-Euler loop. It
/// drives no shared sampler, so it emits through a direct `emit_preview_at` call. Wiring only the
/// curated driver call would have left the lane most Kolors renders take silently dark.
#[test]
#[ignore = "needs the Kolors snapshot + a CUDA GPU; run with --features cuda --ignored"]
fn kolors_native_preview_frames_evolve_toward_the_final_image() {
    assert_lane_previews_converge("kolors-native-euler", None, 12, 1024);
}

/// The registered route's curated lane: `run_curated_sampler` over the Kolors `DiscreteModelSampling`,
/// reached by naming a curated solver. Opts in through the sc-16949 projector hook.
#[test]
#[ignore = "needs the Kolors snapshot + a CUDA GPU; run with --features cuda --ignored"]
fn kolors_curated_preview_frames_evolve_toward_the_final_image() {
    assert_lane_previews_converge("kolors-curated-ddim", Some("ddim"), 12, 1024);
}

/// Exactly one frame per **outer** step on a multi-eval solver, with the guard made non-vacuous
/// first: the shared driver calls `on_progress` once per *evaluation*, so counting `Progress::Step`
/// events counts evaluations. If `heun` did not evaluate twice per step the counts would be equal and
/// "frames == steps" would prove nothing.
#[test]
#[ignore = "needs the Kolors snapshot + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    let events = assert_lane_previews_converge("kolors-heun", Some("heun"), steps, 768);
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
}
