//! sc-16954 — candle SDXL per-step latent **preview** real-weight validation (epic 16948).
//!
//! Three things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story:
//!
//! 1. **The reused fit belongs to this latent space.** SDXL adds no fit; `crate::preview` reuses the
//!    epic-16624 four-channel constants `mlx-gen-sdxl` committed.
//!    [`the_sdxl_family_ships_one_vae_file`] is the reuse gate: the SDXL and Kolors snapshot VAEs and
//!    every shipped tier of both SceneWorks re-hosts are **one byte-identical file**, the file the
//!    MLX fit was measured against. [`the_decode_vae_is_a_different_checkpoint_and_that_is_recorded`]
//!    pins the one asymmetry rather than glossing it — candle SDXL *decodes* through
//!    `madebyollin/sdxl-vae-fp16-fix`, a genuine fine-tune, so the fit's colour target is settled by
//!    the convergence rows below rather than by assertion.
//! 2. **The frames actually develop.** [`sdxl_curated_preview_frames_evolve_toward_the_final_image`]
//!    and [`sdxl_lightning_preview_frames_evolve_toward_the_final_image`] drive both lanes of the
//!    registered route through the `Generator` seam with a live sink, check numbering, check seeded
//!    byte-identity against an inert render, and measure that every frame is closer to — and more
//!    like — the finished image than the one before it. Both strips are written out for review.
//! 3. **One frame per OUTER step on a multi-eval solver.**
//!    [`a_multi_eval_solver_emits_one_frame_per_outer_step`] runs `heun` and proves the guard is
//!    non-vacuous first: the shared driver calls `on_progress` once per *evaluation*, so a
//!    two-eval solver must produce strictly more progress events than steps before "frames == steps"
//!    means anything.
//!
//! [`the_ve_correction_is_what_makes_the_early_frames_readable`] is the row for this story's own
//! finding — that the ε/DDPM cohort's running latent is NOT the tensor the fit was measured on.
//!
//! ```sh
//! SDXL_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<rev> \
//! SDXL_LIGHTNING_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--realvisxl-lightning-mlx\snapshots\<rev>\bf16 \
//! SDXL_TOKENIZER_CLIP_L_DIR=...\models--openai--clip-vit-large-patch14\snapshots\<rev> \
//! SDXL_TOKENIZER_CLIP_BIGG_DIR=...\models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k\snapshots\<rev> \
//! SDXL_VAE_FP16_FIX_DIR=...\models--madebyollin--sdxl-vae-fp16-fix\snapshots\<rev> \
//! SDXL_KOLORS_VAE=...\models--Kwai-Kolors--Kolors-diffusers\snapshots\<rev>\vae\diffusion_pytorch_model.fp16.safetensors \
//! SDXL_PREVIEW_ARTIFACT_DIR=E:\out\sc-16954 \
//!   cargo test -p candle-gen-sdxl --release --features cuda --test integration preview_real_weights:: \
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
const NEGATIVE: &str = "blurry, lowres, deformed, watermark, text";
const SEED: u64 = 16954;

/// The SHA-256 of `vae/diffusion_pytorch_model.fp16.safetensors` — 167,335,342 bytes.
///
/// This one file is published **byte-identically** by `stabilityai/stable-diffusion-xl-base-1.0`
/// @ `462165984030d82259a11f4367a4eed129e94a7b`, by `Kwai-Kolors/Kolors-diffusers`
/// @ `7e091c75199e910a26cd1b51ed52c28de5db3711`, and by every shipped tier (`bf16`/`q8`/`q4`) of
/// `SceneWorks/sdxl-base-mlx` and `SceneWorks/kolors-mlx` — the MLX packer mirrors the VAE dense
/// rather than packing it. It is also the hash `mlx-gen-sdxl/src/preview.rs` cites as its Kolors
/// grounding, so the fit donor's file *is* this file and no tensor-by-tensor argument is needed.
const SDXL_FAMILY_VAE_SHA256: &str =
    "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68";

/// The SHA-256 of `madebyollin/sdxl-vae-fp16-fix` @ `207b116dae70ace3637169f1ddd2434b91b3a8cd`'s
/// `diffusion_pytorch_model.safetensors` — 334,643,238 bytes, f32.
///
/// A **different checkpoint**, not a precision variant: all 248 tensors differ from the original in
/// both the encoder and the decoder. It is the documented drop-in for the same latent space, and it
/// is what candle SDXL *decodes* with. See the row that pins this.
const SDXL_VAE_FP16_FIX_SHA256: &str =
    "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c";

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
    env_path("SDXL_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("sdxl_preview_sc16954"))
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

/// The reuse gate. SDXL and Kolors are one latent space because they ship **one VAE file**, and it is
/// the file the epic-16624 fit was measured against.
#[test]
#[ignore = "needs SDXL_PREVIEW_SNAPSHOT + SDXL_KOLORS_VAE; run with --ignored"]
fn the_sdxl_family_ships_one_vae_file() {
    let sdxl = required_path("SDXL_PREVIEW_SNAPSHOT")
        .join("vae")
        .join("diffusion_pytorch_model.fp16.safetensors");
    let kolors = required_path("SDXL_KOLORS_VAE");

    for (label, path) in [("sdxl", &sdxl), ("kolors", &kolors)] {
        let size = std::fs::metadata(path)
            .unwrap_or_else(|e| panic!("stat {path:?}: {e}"))
            .len();
        let digest = sha256_of(path);
        eprintln!("  {label}: {digest}  {size} bytes  {}", path.display());
        assert_eq!(
            size, 167_335_342,
            "{label} VAE size moved — the fit's latent space may have changed"
        );
        assert_eq!(
            digest, SDXL_FAMILY_VAE_SHA256,
            "{label} VAE is no longer the file the epic-16624 four-channel fit was measured \
             against; re-derive the fit with mlx-gen-sdxl/tests/fit_preview_rgb.rs before reusing it"
        );
    }

    // The two configs define the space. Both must declare the four-channel latent and the 0.13025
    // scaling factor `VAE_SCALE` hardcodes; a change in either invalidates the reuse.
    let config = std::fs::read_to_string(
        required_path("SDXL_PREVIEW_SNAPSHOT")
            .join("vae")
            .join("config.json"),
    )
    .expect("read the sdxl vae config");
    assert!(
        config.contains("\"latent_channels\": 4"),
        "sdxl vae/config.json must still declare a four-channel latent: {config}"
    );
    assert!(
        config.contains("\"scaling_factor\": 0.13025"),
        "sdxl vae/config.json must still declare scaling_factor 0.13025: {config}"
    );
    assert_eq!(
        candle_gen_sdxl::preview::PREVIEW_LATENT_CHANNELS,
        4,
        "the committed fit must have one RGB row per latent channel"
    );
}

/// The one asymmetry, pinned rather than glossed: candle SDXL **decodes** with a different checkpoint
/// from the one that defines the latent space. Recording it here is what stops a later reader
/// assuming the file candle loads is the fit donor's.
#[test]
#[ignore = "needs SDXL_VAE_FP16_FIX_DIR; run with --ignored"]
fn the_decode_vae_is_a_different_checkpoint_and_that_is_recorded() {
    let fix = required_path("SDXL_VAE_FP16_FIX_DIR").join("diffusion_pytorch_model.safetensors");
    let digest = sha256_of(&fix);
    eprintln!("  vae_fp16_fix: {digest}  {}", fix.display());
    assert_eq!(
        digest, SDXL_VAE_FP16_FIX_SHA256,
        "the staged vae_fp16_fix component is not the pinned madebyollin checkpoint"
    );
    assert_ne!(
        digest, SDXL_FAMILY_VAE_SHA256,
        "if these ever become the same file, delete this row and simplify crate::preview's docs"
    );
    // The fit's INPUT domain is unaffected — the UNet that produces these latents is unchanged and
    // `VAE_SCALE` is still 0.13025 — so what remains to be established is that the fit still predicts
    // what THIS decoder produces. That is exactly what the convergence rows below measure, against
    // the image this decoder actually emits.
}

// ── Frame analysis helpers ────────────────────────────────────────────────────────────────────────

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

/// Nearest-neighbour box resample of an RGB8 buffer.
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

/// Lay the strip out as one horizontal contact sheet so a reviewer sees the progression at a glance.
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

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to
/// the finished image, and rising resemblance to it. Applied identically to every lane so none can be
/// closed with a weaker measurement than the others.
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    width: u32,
    height: u32,
    min_r_last: f64,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    // Latent resolution `H/8 × W/8`, and batch 1. A CFG-fused `[2, 4, h, w]` latent fails the
    // `[1, 4, h, w]` contract outright, so a strip that exists at all is already proof the preview
    // never saw the fused unconditional half — there would be no frames if it had.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (width / 8, height / 8),
            "{label}: frames must be VAE-latent resolution"
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

    // Absolute distance can only ever say "closer", never "resembles": the projection is a global
    // linear approximation of the decode (fit R² 0.918, holdout 0.861), so even a perfectly converged
    // latent keeps an offset and gain error against the true pixels — and here the decode runs a
    // different VAE checkpoint from the fit corpus besides. Correlation over a coarse thumbnail,
    // which averages the residual noise away and leaves subject placement and colour masses, is what
    // "the preview looks like the image" actually means for a decorative frame.
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
    // `min_r_last` is per-lane on top of that, and it is likewise not a knob for making a lane pass:
    // it also measures how far the trajectory has travelled **one step from the end**, which is a
    // property of the SCHEDULE.
    //
    // The hook emits BEFORE each solver step (sc-16949), so the last frame is the latent after
    // `steps - 1` of `steps` advancements — the fully denoised state is never previewed, the finished
    // image lands instead. On a 12-step schedule that final step is a small share of the trajectory
    // and the last frame reaches r ~0.89. On the few-step Euler-**trailing** Lightning schedule, whose
    // terminal sigma is zero, it carries a large share: measured on the distilled
    // `realvisxl-lightning` checkpoint the strip rises +0.243 -> +0.600 with frame-to-frame movement
    // still ACCELERATING (mean |delta| 2.20 -> 9.20). A trajectory nowhere near converged is exactly
    // what that schedule should look like at step 7 of 8; holding it to the 12-step lane's floor would
    // be measuring the schedule and calling it the wiring. The few-step lane pays for the lower floor
    // with the extra monotone-acceleration row in its own test.
    assert!(
        r_last > min_r_last,
        "{label}: the last preview frame must resemble the finished render \
         (r {r_last:+.3}, floor {min_r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    // "The strip develops" is asserted as a **rise**, not as an absolute floor on the first frame.
    // Correlation is taken over flattened RGB triplets, so it carries channel-mean structure as well
    // as spatial structure — and this fit's intercept (0.556, 0.509, 0.492) is itself R > G > B, as
    // every warm-lit render also is. A frame of pre-denoise noise therefore starts at a non-zero,
    // *scene-dependent* floor. sc-16950's `r_first < 0.35` ceiling is deliberately not ported; the
    // rise plus a loose ceiling is what cannot be faked, since a strip that opened on the finished
    // image has nowhere to rise to.
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

/// The `LoadSpec` for the registered `sdxl` route: `snapshot` plus its three required components.
///
/// The snapshot is a parameter because the `lightning` lane must be driven by a genuinely
/// **distilled** checkpoint. `lightning` is a few-step Euler-trailing schedule; running it against
/// non-distilled SDXL base weights concentrates nearly the whole denoise into the last one or two
/// steps, so what a strip measures there is a checkpoint/schedule mismatch rather than the lane as it
/// ships (`realvisxl_lightning`). Measured on base weights: the last frame reached only r +0.633 with
/// its frame-to-frame movement still ACCELERATING (mean |delta| 1.46 -> 9.68 across the strip), which
/// is the signature of a trajectory nowhere near converged, not of a mis-wired preview.
fn sdxl_load_spec(snapshot: PathBuf) -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(snapshot))
        .with_component(
            "tokenizer_clip_l",
            WeightsSource::Dir(required_path("SDXL_TOKENIZER_CLIP_L_DIR")),
        )
        .with_component(
            "tokenizer_clip_bigg",
            WeightsSource::Dir(required_path("SDXL_TOKENIZER_CLIP_BIGG_DIR")),
        )
        .with_component(
            "vae_fp16_fix",
            WeightsSource::Dir(required_path("SDXL_VAE_FP16_FIX_DIR")),
        )
}

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
/// the strip to [`assert_the_strip_converges`]. Returns the progress-event count of the live run.
fn assert_lane_previews_converge(
    label: &str,
    snapshot_var: &str,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
    min_r_last: f64,
) -> (usize, Vec<PreviewFrame>) {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}, from {snapshot_var}");
    let generator = candle_gen_sdxl::provider_registry()
        .expect("sdxl registry")
        .load("sdxl", &sdxl_load_spec(required_path(snapshot_var)))
        .unwrap_or_else(|e| panic!("load sdxl: {e}"));

    // N1: the inert baseline. Same generator, same seed, no sink.
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

    // An active sink must not move a single bit of the render.
    assert_eq!(
        inert.pixels, live.pixels,
        "{label}: attaching a live preview sink changed the seeded render"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    assert_the_strip_converges(label, &frames, &live, steps, size, size, min_r_last);
    save_strip(&frames, &format!("{label}-strip.png"));
    save_png(
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
    (events, frames)
}

// ── The lanes ─────────────────────────────────────────────────────────────────────────────────────

/// The registered route's DEFAULT lane: `run_curated_sampler` over `DiscreteModelSampling`. This is
/// the lane every non-`lightning` SDXL request takes, including one that names no sampler at all.
#[test]
#[ignore = "needs the SDXL snapshot + components + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_curated_preview_frames_evolve_toward_the_final_image() {
    assert_lane_previews_converge(
        "sdxl-curated-ddim",
        "SDXL_PREVIEW_SNAPSHOT",
        None,
        12,
        1024,
        0.83,
    );
}

/// The registered route's OTHER lane: the bespoke Lightning Euler-trailing loop, which drives no
/// shared sampler and therefore emits through a direct `emit_preview_at` call instead of a hook.
/// Wiring only the driver call would have left this lane silently dark on a shipped route.
#[test]
#[ignore = "needs the SDXL snapshot + components + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_lightning_preview_frames_evolve_toward_the_final_image() {
    let (_, frames) = assert_lane_previews_converge(
        "sdxl-lightning",
        "SDXL_LIGHTNING_SNAPSHOT",
        Some("lightning"),
        8,
        1024,
        // See `assert_the_strip_converges`: a zero-terminal few-step schedule leaves a large share of
        // the trajectory in the final step, which the emit-before-step contract never previews.
        0.55,
    );

    // What the few-step lane pays for that lower floor. Frame-to-frame movement must be strictly
    // INCREASING across the whole strip — the Euler-trailing signature, and direct evidence that the
    // last frame's shortfall is the schedule accelerating into its terminal step rather than a preview
    // that stopped tracking. A hook reading a stale or wrongly scaled latent would not reproduce a
    // monotone acceleration.
    let movement: Vec<f64> = frames
        .windows(2)
        .map(|p| mean_abs_delta(&p[0].image.pixels, &p[1].image.pixels))
        .collect();
    assert!(
        movement.windows(2).all(|p| p[1] > p[0]),
        "the trailing schedule must accelerate into its terminal step: {movement:?}"
    );
}

/// Exactly one frame per **outer** solver step on a multi-eval solver.
///
/// The guard is made non-vacuous first, and in the strongest available way: the shared driver calls
/// `on_progress` once per *evaluation* (`sampler.rs` computes the step count on every eval and
/// deliberately repeats it), so counting `Progress::Step` events IS counting evaluations. If `heun`
/// did not evaluate twice per step the event count would equal the step count and "frames == steps"
/// would prove nothing — so that inequality is asserted before the frame count is.
#[test]
#[ignore = "needs the SDXL snapshot + components + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    let (events, _) = assert_lane_previews_converge(
        "sdxl-heun",
        "SDXL_PREVIEW_SNAPSHOT",
        Some("heun"),
        steps,
        768,
        0.83,
    );
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
    // `assert_lane_previews_converge` already required exactly `steps` frames numbered 1..=steps, so
    // the dedup collapsed the extra evaluations. Stated here because that is the point of the row.
}

/// This story's own finding, measured rather than argued: the ε/DDPM running latent is **not** the
/// tensor the fit was measured on, and projecting it raw would open the strip on a saturated field.
///
/// Runs entirely on the committed constants — no weights — because the claim is about the projection,
/// not the model. It is in this file rather than the unit tests because it is the numeric evidence
/// the evidence document cites.
#[test]
fn the_ve_correction_is_what_makes_the_early_frames_readable() {
    use candle_gen::candle_core::{Device, Tensor};

    // A unit-normal latent at the schedule's largest sigma is what the first emission actually sees.
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let noise = candle_gen::seeded_normal_vec(&mut rng, 4 * 64 * 64);
    let unit = Tensor::from_vec(noise, (1, 4, 64, 64), &Device::Cpu).expect("build the latent");
    let sigma_max = 14.6f32;
    let ve = (unit * sigma_max as f64).expect("scale to VE space");

    let raw = candle_gen_sdxl::preview::project_spatial_latents(&ve).expect("raw projection");
    let corrected =
        candle_gen_sdxl::preview::project_ve_latents(&ve, Some(sigma_max)).expect("corrected");

    let rails =
        |p: &[u8]| p.iter().filter(|&&v| v == 0 || v == 255).count() as f64 / p.len() as f64;
    let (raw_rails, corrected_rails) = (rails(&raw.pixels), rails(&corrected.pixels));
    eprintln!("  sigma {sigma_max}: raw projection clipped fraction {raw_rails:.3}");
    eprintln!("  sigma {sigma_max}: corrected projection clipped fraction {corrected_rails:.3}");
    // Measured on this seeded latent: raw 0.894, corrected 0.060. The bounds bracket those two
    // numbers loosely enough that a rounding change cannot flip them, and far enough apart that only
    // the ~15x collapse in clipping the correction actually buys can satisfy both.
    assert!(
        raw_rails > 0.5,
        "an uncorrected VE projection at sigma_max should be mostly clipped ({raw_rails:.3})"
    );
    assert!(
        corrected_rails < 0.10,
        "the corrected projection must be a readable noise field, not a clipped one \
         ({corrected_rails:.3})"
    );
}
