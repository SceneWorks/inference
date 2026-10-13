//! The bounded real-weight validation render (sc-24109) — `#[ignore]`d **and** gated to a GPU
//! backend: it needs the pinned `Qwen/Qwen-Image-2.1` snapshot at
//! `CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT` (inference never self-fetches or derives a cache location,
//! epic 13657) and an accelerator, because the released weights are bf16 and candle's plain CPU
//! backend has no bf16 matmul — the same reason `candle-gen-ltx`'s conformance suite is gated.
//!
//! The gate is `any(cuda, metal)` rather than `cuda` alone so the macOS candle lane type-checks
//! this file too (`cargo check --features metal --all-targets`); CUDA is the lane the render is
//! meant for. Ordinary CI never reaches it: the dispatch-only `qwen-image-2-1` profile of
//! `.github/workflows/real-weights.yml` (sc-24114) provisions both pinned snapshots on the Windows
//! CUDA runner, runs every test in this file and uploads the PNGs, alpha reports and logs as the
//! `qwen-image-2-1-cuda-evidence` artifact.
//!
//! Loads the released bf16 weights through the explicit catalog's production load path and renders
//! one image at the upstream default preset (1:1 2048×2048, 40 steps, seed 42, no guidance),
//! writing the raw RGB8 bytes and a PNG of them to `QWEN_IMAGE_2_1_RENDER_OUT` (default: the
//! current directory). It
//! also pins the released tokenizer's system-prefix drop count (14) — the one claim that needs only
//! the snapshot's `processor/`, not the GPU.
//!
//! ```sh
//! CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1/snapshots/790c9263… \
//! QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-24109 \
//!   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
//!   --test integration e2e_real_weights:: -- --ignored --nocapture
//! ```
//!
//! `QWEN_IMAGE_2_1_RENDER_SIZE=WxH` and `QWEN_IMAGE_2_1_RENDER_STEPS=N` override the preset for a
//! quicker smoke.
//!
//! The `turbo_`-prefixed tests (sc-25901, sc-25902) are the Turbo twins: the default-preset
//! render, the reference edit, the transparency smoke, and a step sweep. They read
//! `CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT` instead — and nothing of the base snapshot — and are
//! run by their own dispatch-only profile, `qwen-image-2-1-turbo`, which provisions
//! `Qwen/Qwen-Image-2.1-Turbo`; the `qwen-image-2-1` profile above runs every test here that is
//! **not** `turbo_`-prefixed. The prefix is what partitions the two lanes
//! (`scripts/tests/test_ci_workflow_policy.py`), so a Turbo test must carry it.

#![cfg(any(feature = "cuda", feature = "metal"))]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use candle_gen_qwen_image_2_1::{load_tokenizer, system_prompt_drop_count, PRESETS};

/// Write `pixels` (RGB8 or RGBA8, per `color`) as a PNG beside the raw bytes, so the evidence
/// artifact carries something a reviewer can open without knowing the geometry (sc-24114).
fn save_png(
    path: &std::path::Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
) {
    image::save_buffer(path, pixels, width, height, color)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

/// Reject a render whose rows are stale GPU memory rather than a picture (sc-24114): the 2048²
/// evidence render passed every shape assertion with 430 correct rows above 1,600 rows of noise
/// — candle's CUDA im2col launch had truncated the full-resolution conv's element count to u32.
/// Per row, the mean absolute second difference of the luma along the columns is ~3 for a
/// rendered scene (the evidence's intact strip peaks at 9.2, the sane 1024² renders at 5.8) and
/// 20–35 for uninitialised memory; a row over 20 is static, and more than a quarter of them fails
/// the render. The fraction is printed so a borderline pass is still readable in the log.
fn assert_not_band_corrupted(label: &str, pixels: &[u8], width: u32, height: u32) {
    let (w, h) = (width as usize, height as usize);
    assert_eq!(pixels.len(), w * h * 3, "{label}: RGB8 geometry");
    assert!(w >= 3, "{label}: width {w} is too narrow to measure");
    let luma =
        |px: &[u8]| 0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]);
    let static_rows = pixels
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
    let fraction = static_rows as f32 / h as f32;
    eprintln!(
        "{label}: static rows {static_rows}/{h} ({:.1}%)",
        100.0 * fraction
    );
    assert!(
        fraction <= 0.25,
        "{label}: {static_rows} of {h} rows are noise — the render is band-corrupted (a stale \
         or truncated device buffer reached the output)"
    );
}

fn snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT to the pinned snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

#[test]
#[ignore]
fn released_tokenizer_drops_fourteen_system_tokens() {
    let tokenizer = load_tokenizer(&snapshot()).unwrap();
    assert_eq!(system_prompt_drop_count(&tokenizer).unwrap(), 14);
}

#[test]
#[ignore]
fn validation_render_default_preset() {
    let root = snapshot();
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    let steps: u32 = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(candle_gen_qwen_image_2_1::DEFAULT_STEPS);
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();

    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let req = GenerationRequest {
        prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
            .to_owned(),
        width,
        height,
        steps: Some(steps),
        seed: Some(42),
        ..Default::default()
    };
    let render_started = Instant::now();
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                eprintln!(
                    "step {current}/{total} ({:.1}s)",
                    render_started.elapsed().as_secs_f32()
                );
            } else {
                eprintln!("{p:?} ({:.1}s)", render_started.elapsed().as_secs_f32());
            }
        })
        .unwrap();
    let GenerationOutput::Images(images) = out else {
        panic!("images expected");
    };
    let image = &images[0];
    assert_eq!((image.width, image.height), (width, height));
    assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
    let path = out_dir.join(format!(
        "qwen_image_2_1_candle_{width}x{height}_{steps}steps_seed42.rgb"
    ));
    std::fs::write(&path, &image.pixels).unwrap();
    save_png(
        &path.with_extension("png"),
        &image.pixels,
        width,
        height,
        image::ColorType::Rgb8,
    );
    assert_not_band_corrupted("default_preset", &image.pixels, width, height);
    eprintln!(
        "wrote {} ({width}x{height} RGB8) after {:.1}s total ({:.1}s render)",
        path.display(),
        started.elapsed().as_secs_f32(),
        render_started.elapsed().as_secs_f32()
    );
}

/// Reject a render that is not a picture: a constant or near-constant RGB8 buffer (a black, white
/// or flat-grey frame). The floor is deliberately low — sixteen luma levels of spread and more
/// than sixteen distinct byte values — so it cannot reject a legitimately dark or soft image, only
/// a degenerate one. ([`assert_not_band_corrupted`] catches the opposite failure, noise.)
fn assert_not_degenerate(label: &str, pixels: &[u8]) {
    let luma = |px: &[u8]| {
        (299 * u32::from(px[0]) + 587 * u32::from(px[1]) + 114 * u32::from(px[2])) / 1000
    };
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let mut seen = [false; 256];
    for px in pixels.chunks_exact(3) {
        let l = luma(px);
        lo = lo.min(l);
        hi = hi.max(l);
        for byte in px {
            seen[usize::from(*byte)] = true;
        }
    }
    let distinct = seen.iter().filter(|s| **s).count();
    eprintln!("{label}: luma range [{lo}, {hi}], {distinct} distinct byte values");
    assert!(
        hi >= lo + 16 && distinct > 16,
        "{label}: the render is a near-constant field (luma [{lo}, {hi}], {distinct} distinct \
         byte values), not a picture"
    );
}

/// [`assert_not_degenerate`] rejects a blank frame and accepts a dark, low-contrast picture. Not
/// ignored and device-free: it is the one part of the Turbo step sweep that can be exercised
/// without the released weights.
#[test]
fn the_degenerate_render_check_rejects_a_flat_frame_only() {
    let flat = vec![7u8; 64 * 64 * 3];
    assert!(std::panic::catch_unwind(|| assert_not_degenerate("flat", &flat)).is_err());
    // Two byte values only: spread without content.
    let two: Vec<u8> = (0..64 * 64 * 3)
        .map(|i| if (i / 3) % 2 == 0 { 0 } else { 255 })
        .collect();
    assert!(std::panic::catch_unwind(|| assert_not_degenerate("two-valued", &two)).is_err());
    // A dim ramp: 24 luma levels of spread, 24 distinct values.
    let dim: Vec<u8> = (0..64 * 64)
        .flat_map(|i| [(i % 24) as u8 + 10; 3])
        .collect();
    assert_not_degenerate("dim ramp", &dim);
}

fn out_dir() -> PathBuf {
    let dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn turbo_snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT to the pinned Qwen/Qwen-Image-2.1-Turbo snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

/// The Turbo twin of [`validation_render_default_preset`] (sc-25901): the released
/// `Qwen/Qwen-Image-2.1-Turbo` snapshot, **unmodified**, loaded as `qwen_image_2_1_turbo` through
/// the catalog path and rendered at the default 1:1 2048×2048 preset with `steps` left unset — so
/// the step count is the one the snapshot's own `sample_sigmas` gives (eight for the pinned
/// revision, asserted from the progress events), true CFG off, seed 42.
///
/// It also proves, on the real bytes, the two storage differences the tiny fixture only imitates:
/// the bf16 `vae/` and the single-file `text_encoder/model.safetensors` both load.
///
/// Run by the dispatch-only `qwen-image-2-1-turbo` profile of `real-weights.yml`; the `turbo_`
/// prefix is what that lane selects on (and what the base `qwen-image-2-1` lane leaves out — it
/// does not provision the Turbo snapshot).
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25901 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_default_preset \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_default_preset() {
    // Unset by default: the count then comes from the snapshot, which is the claim.
    let steps: Option<u32> = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap());
    turbo_validation_renders(&[steps]);
}

/// The Turbo **step sweep** (sc-25902): the default 1:1 2048×2048 preset, the prompt and seed of
/// [`turbo_validation_render_default_preset`], rendered at 4, 6, 12 and 16 steps — the resampled
/// saved schedule on both sides of the checkpoint's own eight — from one loaded generator, one PNG
/// per step count (`qwen_image_2_1_turbo_candle_2048x2048_<N>steps_seed42.png`). It asserts that
/// every count ran as asked and that no render is band-corrupted; the PNGs are what the epic's
/// terminal story reads to choose the manifest's minimum step count.
/// `QWEN_IMAGE_2_1_SWEEP_STEPS=4,6,12,16` names other counts; `QWEN_IMAGE_2_1_RENDER_SIZE` applies.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25902 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_step_sweep \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_step_sweep() {
    let steps: Vec<Option<u32>> = match std::env::var("QWEN_IMAGE_2_1_SWEEP_STEPS") {
        Ok(list) => list
            .split(',')
            .map(|steps| Some(steps.trim().parse().expect("a step count")))
            .collect(),
        Err(_) => vec![Some(4), Some(6), Some(12), Some(16)],
    };
    turbo_validation_renders(&steps);
}

/// One default-preset Turbo render per entry of `steps`, all from one loaded generator; a `None`
/// entry leaves the step count to the snapshot's saved sigmas. Each asserts the step total the
/// sampler reported.
fn turbo_validation_renders(steps: &[Option<u32>]) {
    let root = turbo_snapshot();
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    let out_dir = out_dir();

    let saved = candle_gen_qwen_image_2_1::load_turbo_schedule(&root).unwrap();
    eprintln!("snapshot sample_sigmas: {:?}", saved.sample_sigmas);
    assert_eq!(
        saved.default_steps(),
        8,
        "the pinned Turbo revision saves eight sigmas"
    );

    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(
            candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(root)),
        )
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let mut renders: Vec<(u32, Vec<u8>)> = Vec::with_capacity(steps.len());
    for steps in steps {
        let req = GenerationRequest {
            prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
                .to_owned(),
            width,
            height,
            steps: *steps,
            seed: Some(42),
            ..Default::default()
        };
        let render_started = Instant::now();
        let mut totals = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    totals.push(total);
                    eprintln!(
                        "step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                } else {
                    eprintln!("{p:?} ({:.1}s)", render_started.elapsed().as_secs_f32());
                }
            })
            .unwrap();
        let ran = steps.unwrap_or(saved.default_steps() as u32);
        assert_eq!(
            totals,
            vec![ran; ran as usize],
            "the requested step count, or one step per saved sigma"
        );
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
        let path = out_dir.join(format!(
            "qwen_image_2_1_turbo_candle_{width}x{height}_{ran}steps_seed42.rgb"
        ));
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        assert_not_band_corrupted(
            &format!("turbo_default_preset_{ran}steps"),
            &image.pixels,
            width,
            height,
        );
        // Evidence someone reads to pick a step count must not be a blank frame, and two step
        // counts must not have produced the same picture (a dead `steps` knob).
        assert_not_degenerate(&format!("turbo_default_preset_{ran}steps"), &image.pixels);
        for (other, pixels) in &renders {
            assert!(
                *other == ran || *pixels != image.pixels,
                "{other} and {ran} steps rendered identical pixels"
            );
        }
        renders.push((ran, image.pixels.clone()));
        eprintln!(
            "wrote {} ({width}x{height} RGB8) after {:.1}s total ({:.1}s render)",
            path.display(),
            started.elapsed().as_secs_f32(),
            render_started.elapsed().as_secs_f32()
        );
    }
}

/// A deterministic synthetic reference, so the edit smoke needs no checked-in photographs.
/// `QWEN_IMAGE_2_1_EDIT_REFS` points at a directory of raw `WxH.rgb` files to use instead, in
/// name order — reference images for a real evaluation are user-provided, never synthesised or
/// fetched here. Each file's dimensions come from its stem (`768x768.rgb`).
fn synthetic_reference(seed: u32, width: u32, height: u32) -> candle_gen::gen_core::Image {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5 + seed * 17) % 256) as u8);
            pixels.push(((x * 7 + seed * 29) % 256) as u8);
            pixels.push(((y * 11 + seed * 41) % 256) as u8);
        }
    }
    candle_gen::gen_core::Image {
        width,
        height,
        pixels,
    }
}

fn references(count: usize, width: u32, height: u32) -> Vec<candle_gen::gen_core::Image> {
    let Ok(dir) = std::env::var("QWEN_IMAGE_2_1_EDIT_REFS") else {
        return (0..count)
            .map(|i| synthetic_reference(i as u32 + 1, width, height))
            .collect();
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("QWEN_IMAGE_2_1_EDIT_REFS={dir}: {e}"))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "rgb"))
        .collect();
    paths.sort();
    assert!(
        paths.len() >= count,
        "QWEN_IMAGE_2_1_EDIT_REFS={dir} holds {} .rgb files, {count} needed",
        paths.len()
    );
    paths
        .into_iter()
        .take(count)
        .map(|path| {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_else(|| panic!("{}: a WxH.rgb name", path.display()));
            let (w, h) = stem
                .split_once('x')
                .unwrap_or_else(|| panic!("{}: a WxH.rgb name", path.display()));
            let (w, h): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
            let pixels = std::fs::read(&path).unwrap();
            assert_eq!(
                pixels.len(),
                (w as usize) * (h as usize) * 3,
                "{}: {w}x{h} RGB8",
                path.display()
            );
            candle_gen::gen_core::Image {
                width: w,
                height: h,
                pixels,
            }
        })
        .collect()
}

/// One bounded real-weight **edit** render per reference count (sc-24110) — the candle twin of the
/// MLX smoke, aimed at the CUDA lane: the two-reference 1024x1024 case a caller actually sends,
/// and the ten-reference boundary at a small target so the longest joint sequence is exercised
/// without a long render. Finishes by proving eleven references are refused at `validate`, before
/// any weight is touched.
///
/// Every condition image is fitted to `output_resolution` (1024 px) whatever the target size, so
/// ten references are ~41k prefix tokens on their own. The generator runs upstream's prefix KV
/// cache (sc-25904 — see UPSTREAM.md), so only the first model evaluation pays for that prefix;
/// the boundary case still runs the fewest steps the sampler accepts rather than a smaller target
/// (it was sized when every step recomputed the whole block-causal prefix).
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT=.../snapshots/790c9263... \
/// QWEN_IMAGE_2_1_RENDER_OUT=.../render-validation-sc-24110 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::validation_render_reference_edit \
///   -- --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_reference_edit() {
    // (references, width, height, steps) — the caller-shaped case, then the boundary.
    reference_edit_renders(
        "qwen_image_2_1",
        snapshot(),
        "qwen_image_2_1_candle",
        candle_gen_qwen_image_2_1::DEFAULT_STEPS,
        [(2, 1024, 1024, Some(8)), (10, 512, 512, Some(2))],
    );
}

/// The Turbo twin of [`validation_render_reference_edit`] (sc-25902), through the
/// `qwen_image_2_1_turbo` id on the unmodified Turbo snapshot: the two-reference 1024x1024 case
/// and the ten-reference boundary, **both with no step count on the request** — so each runs the
/// snapshot's saved sigmas (eight on the pinned revision), asserted from the progress events. The
/// boundary case runs the full schedule here because the prefix KV cache (sc-25904) makes only the
/// first evaluation pay for the ~41k-token prefix. Finishes with the eleven-reference refusal.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25902 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_reference_edit \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_reference_edit() {
    let root = turbo_snapshot();
    let saved = candle_gen_qwen_image_2_1::load_turbo_schedule(&root)
        .unwrap()
        .default_steps() as u32;
    assert_eq!(saved, 8, "the pinned Turbo revision saves eight sigmas");
    reference_edit_renders(
        candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
        root,
        "qwen_image_2_1_turbo_candle",
        saved,
        [(2, 1024, 1024, None), (10, 512, 512, None)],
    );
}

/// The reference-edit smoke of engine `id` over the snapshot at `root`: one render per
/// `(references, width, height, steps)` case — `steps: None` leaves the count to the generator,
/// which must then run `default_steps` — then the eleven-reference refusal. `stem` prefixes the
/// evidence file names.
fn reference_edit_renders(
    id: &str,
    root: PathBuf,
    stem: &str,
    default_steps: u32,
    cases: [(usize, u32, u32, Option<u32>); 2],
) {
    use candle_gen::gen_core::Conditioning;

    let out_dir = out_dir();
    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    for (count, width, height, steps) in cases {
        let refs = references(count, 768, 768);
        let req = GenerationRequest {
            prompt: "Combine the subjects of the reference images into one scene, evening light"
                .to_owned(),
            width,
            height,
            steps,
            seed: Some(42),
            conditioning: vec![Conditioning::MultiReference { images: refs }],
            ..Default::default()
        };
        let render_started = Instant::now();
        let mut totals = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    totals.push(total);
                    eprintln!(
                        "{count} refs: step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                } else {
                    eprintln!(
                        "{count} refs: {p:?} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                }
            })
            .unwrap();
        let ran = steps.unwrap_or(default_steps);
        assert_eq!(
            totals,
            vec![ran; ran as usize],
            "{count} refs: the requested step count, or the checkpoint's own"
        );
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
        let path = out_dir.join(format!(
            "{stem}_edit_{count}refs_{width}x{height}_{ran}steps_seed42.rgb"
        ));
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        assert_not_band_corrupted(&format!("{count} refs"), &image.pixels, width, height);
        eprintln!(
            "wrote {} ({width}x{height} RGB8) after {:.1}s render",
            path.display(),
            render_started.elapsed().as_secs_f32()
        );
    }

    let err = generator
        .validate(&GenerationRequest {
            prompt: "too many".to_owned(),
            width: 512,
            height: 512,
            // Refused at `validate`, before any image is looked at, so the eleven are synthetic
            // whatever `QWEN_IMAGE_2_1_EDIT_REFS` holds (a ten-file dir is the boundary itself,
            // and `references(11, …)` would panic on it instead of reaching the refusal).
            conditioning: vec![Conditioning::MultiReference {
                images: (1..=11).map(|i| synthetic_reference(i, 256, 256)).collect(),
            }],
            ..Default::default()
        })
        .err()
        .map(|e| e.to_string())
        .expect("eleven references must be refused");
    eprintln!("eleven references: {err}");
    assert!(err.contains("at most 10"), "{err}");
}

/// A **transparent** RGBA reference of `width`×`height`: the same deterministic picture
/// [`synthetic_reference`] produces, matted onto a soft-edged disc so the alpha is a real ramp
/// rather than a binary cut-out (a hard matte would survive any resampling order and so could not
/// exercise the premultiplied resize). The candle twin of the MLX smoke's reference (sc-24111).
///
/// Set `QWEN_IMAGE_2_1_RGBA_REF` to a PNG with an alpha channel to extract from a real image
/// instead; reference images for a real evaluation are user-provided.
fn transparent_reference(width: u32, height: u32) -> candle_gen::gen_core::RgbaImage {
    if let Ok(path) = std::env::var("QWEN_IMAGE_2_1_RGBA_REF") {
        let rgba = image::open(&path)
            .unwrap_or_else(|e| panic!("QWEN_IMAGE_2_1_RGBA_REF={path}: {e}"))
            .to_rgba8();
        return candle_gen::gen_core::RgbaImage {
            width: rgba.width(),
            height: rgba.height(),
            pixels: rgba.into_raw(),
        };
    }
    let rgb = synthetic_reference(1, width, height);
    let mut out = candle_gen::gen_core::RgbaImage::from_rgb(&rgb).unwrap();
    let (cx, cy) = ((width - 1) as f32 / 2.0, (height - 1) as f32 / 2.0);
    let radius = width.min(height) as f32 * 0.36;
    for (i, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let (x, y) = ((i as u32 % width) as f32, (i as u32 / width) as f32);
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        // A 3-px linear ramp at the disc boundary: opaque inside, transparent outside.
        px[3] = (((radius + 1.5 - dist) / 3.0).clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

/// The alpha histogram claim the transparency smoke exists to make — the same three conditions as
/// the MLX twin's: the alpha is not constant; at least 2 % of pixels are substantially transparent
/// (`A < 128`) **and** at least 2 % substantially opaque (`A > 200`); and compositing over white
/// moves at least 2 % of pixels. "Four channels came back" is not evidence of transparency — a
/// port that widened an RGB decode with a constant `A = 255` would satisfy it. The full 16-bucket
/// histogram is printed and written beside the PNG.
fn assert_nontrivial_alpha(
    label: &str,
    image: &candle_gen::gen_core::RgbaImage,
    out_dir: &std::path::Path,
) {
    let alpha: Vec<u8> = image.pixels.chunks_exact(4).map(|px| px[3]).collect();
    let total = alpha.len();
    assert!(total > 0, "{label}: empty image");

    let mut histogram = [0usize; 16];
    for &a in &alpha {
        histogram[(a as usize) / 16] += 1;
    }
    let (min, max) = (*alpha.iter().min().unwrap(), *alpha.iter().max().unwrap());
    let transparent = alpha.iter().filter(|&&a| a < 128).count();
    let opaque = alpha.iter().filter(|&&a| a > 200).count();

    let flattened = image.to_rgb_over_white().unwrap();
    let raw: Vec<u8> = image
        .pixels
        .chunks_exact(4)
        .flat_map(|px| px[..3].to_vec())
        .collect();
    let moved = flattened
        .pixels
        .chunks_exact(3)
        .zip(raw.chunks_exact(3))
        .filter(|(a, b)| a != b)
        .count();

    let report = format!(
        "{label}: alpha min={min} max={max} \
         transparent(<128)={transparent}/{total} ({:.1}%) \
         opaque(>200)={opaque}/{total} ({:.1}%) \
         composite-moved={moved}/{total} ({:.1}%)\nhistogram(16 buckets)={histogram:?}\n",
        100.0 * transparent as f32 / total as f32,
        100.0 * opaque as f32 / total as f32,
        100.0 * moved as f32 / total as f32,
    );
    eprint!("{report}");
    std::fs::write(out_dir.join(format!("{label}_alpha.txt")), &report).unwrap();

    assert!(min != max, "{label}: the alpha channel is constant ({min})");
    let floor = total / 50; // 2 %
    assert!(
        transparent >= floor,
        "{label}: only {transparent}/{total} pixels are substantially transparent; the render \
         carries no usable matte"
    );
    assert!(
        opaque >= floor,
        "{label}: only {opaque}/{total} pixels are substantially opaque; the render is a uniform \
         wash rather than a subject on transparency"
    );
    assert!(
        moved >= floor,
        "{label}: compositing over white changed only {moved}/{total} pixels — the alpha is not \
         load-bearing"
    );
}

/// The bounded real-weight **transparency** smoke on candle (sc-24114, the twin of the MLX
/// sc-24111 smoke): one transparent text-to-image and one RGBA-reference extraction, both at
/// 1024×1024 / 8 steps, each asserting a non-trivial alpha histogram on the emitted `RgbaImage`.
///
/// Transparency is requested by **prompt** (upstream ships no flag — see `UPSTREAM.md`);
/// `output_channels: Rgba` only decides that the decoder's alpha reaches the caller instead of
/// being composited over white. Both PNGs are written as RGBA8 so the alpha survives to disk.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT=.../snapshots/790c9263... \
/// QWEN_IMAGE_2_1_RENDER_OUT=.../render-validation-sc-24114 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::validation_render_transparency \
///   -- --ignored --exact --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_transparency() {
    transparency_renders(
        "qwen_image_2_1",
        snapshot(),
        "qwen_image_2_1_candle",
        "",
        candle_gen_qwen_image_2_1::DEFAULT_STEPS,
        Some(8),
    );
}

/// The Turbo twin of [`validation_render_transparency`] (sc-25902), through the
/// `qwen_image_2_1_turbo` id on the unmodified Turbo snapshot: the transparent text-to-image and
/// the RGBA-reference extraction at 1024×1024 with **no step count on the request** (the saved
/// eight sigmas, asserted from the progress events), each asserting the same non-trivial alpha
/// histogram.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25902 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_transparency \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_transparency() {
    let root = turbo_snapshot();
    let saved = candle_gen_qwen_image_2_1::load_turbo_schedule(&root)
        .unwrap()
        .default_steps() as u32;
    assert_eq!(saved, 8, "the pinned Turbo revision saves eight sigmas");
    transparency_renders(
        candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
        root,
        "qwen_image_2_1_turbo_candle",
        "turbo_",
        saved,
        None,
    );
}

/// The transparency smoke of engine `id` over the snapshot at `root`; `steps: None` leaves the
/// step count to the generator, which must then run `default_steps`. `stem` prefixes the PNG names
/// and `report_prefix` the `<label>_alpha.txt` reports.
fn transparency_renders(
    id: &str,
    root: PathBuf,
    stem: &str,
    report_prefix: &str,
    default_steps: u32,
    steps: Option<u32>,
) {
    use candle_gen::gen_core::{Conditioning, OutputChannels};

    let out_dir = out_dir();
    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    // The model card's transparent-image prompt form, and an extraction prompt over a reference
    // that already carries alpha (transparent-layer editing).
    let cases: [(&str, String, Vec<Conditioning>); 2] = [
        (
            "t2i_transparent",
            "This is an RGBA image with transparency. A cute cartoon fox sticker, bold clean \
             outline. The image has an alpha channel and the background is transparent."
                .to_owned(),
            Vec::new(),
        ),
        (
            "rgba_reference_extraction",
            "Extract the subject onto a transparent background. This is an RGBA image with \
             transparency; the background is fully transparent."
                .to_owned(),
            vec![Conditioning::ReferenceRgba {
                image: transparent_reference(1024, 1024),
                strength: None,
            }],
        ),
    ];

    for (label, prompt, conditioning) in cases {
        let req = GenerationRequest {
            prompt,
            width: 1024,
            height: 1024,
            steps,
            seed: Some(42),
            conditioning,
            output_channels: OutputChannels::Rgba,
            ..Default::default()
        };
        generator
            .validate(&req)
            .unwrap_or_else(|e| panic!("{label}: the RGBA request was refused at validate: {e}"));

        let render_started = Instant::now();
        let mut totals = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    totals.push(total);
                    eprintln!(
                        "{label}: step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                } else {
                    eprintln!(
                        "{label}: {p:?} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                }
            })
            .unwrap();
        let ran = steps.unwrap_or(default_steps);
        assert_eq!(
            totals,
            vec![ran; ran as usize],
            "{label}: the requested step count, or the checkpoint's own"
        );
        let GenerationOutput::ImagesRgba(images) = out else {
            panic!("{label}: an `output_channels: Rgba` request must emit ImagesRgba");
        };
        let image = &images[0];
        image.validate().unwrap();
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.channels(), 4);

        save_png(
            &out_dir.join(format!("{stem}_{label}_1024x1024_{ran}steps_seed42.png")),
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgba8,
        );
        assert_nontrivial_alpha(&format!("{report_prefix}{label}"), image, &out_dir);
        eprintln!(
            "{label}: done after {:.1}s render",
            render_started.elapsed().as_secs_f32()
        );
    }
}

/// The published Q8/Q4 tier snapshot (`SceneWorks/qwen-image-2-1-mlx`): one directory holding a
/// complete `q8/` and `q4/` snapshot, each in the packed `model.safetensors` layout the candle
/// loader reads natively (sc-24112).
fn tier_snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT to the pinned SceneWorks/qwen-image-2-1-mlx snapshot dir (holding q8/ and q4/); inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

/// The **published** tiers load and render on real weights (sc-24114). `tests/tiers.rs` proves the
/// packed loader against the committed miniature fixture; this is the one test that opens the
/// artefacts a user actually installs. Each tier must self-report as what its directory claims
/// (read from the packed shapes, not the label), load packed through the production catalog path,
/// and render a 1024×1024 / 8-step image that is not degenerate — a missed packed site reads codes
/// as floats and collapses the image to a flat field, which the spread floor rejects.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT=.../snapshots/1691de01... \
/// QWEN_IMAGE_2_1_RENDER_OUT=.../render-validation-sc-24114 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::validation_render_installed_tiers \
///   -- --ignored --exact --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_installed_tiers() {
    use candle_gen::gen_core::Quant;
    use candle_gen_qwen_image_2_1::quant::{installed_tier, Tier};

    let root = tier_snapshot();
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();
    let (width, height, steps) = (1024u32, 1024u32, 8u32);

    for (label, tier, quant) in [("q8", Tier::Q8, Quant::Q8), ("q4", Tier::Q4, Quant::Q4)] {
        let dir = root.join(label);
        assert_eq!(
            installed_tier(&dir).unwrap(),
            tier,
            "{label}: {} does not hold the tier its name claims",
            dir.display()
        );

        let started = Instant::now();
        let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
        let generator = registry
            .load(
                "qwen_image_2_1",
                &LoadSpec::new(WeightsSource::Dir(dir)).with_quant(quant),
            )
            .unwrap();
        eprintln!("{label}: loaded in {:.1}s", started.elapsed().as_secs_f32());

        let req = GenerationRequest {
            prompt:
                "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
                    .to_owned(),
            width,
            height,
            steps: Some(steps),
            seed: Some(42),
            ..Default::default()
        };
        let render_started = Instant::now();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    eprintln!(
                        "{label}: step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                }
            })
            .unwrap();
        let GenerationOutput::Images(images) = out else {
            panic!("{label}: images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);

        let n = image.pixels.len() as f64;
        let mean = image.pixels.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = image
            .pixels
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();
        eprintln!("{label}: pixel mean={mean:.2} std={std:.2}");
        assert!(
            std > 8.0,
            "{label}: the render is a near-flat field (std {std:.2}) — a packed site was read as \
             dense"
        );

        let path = out_dir.join(format!(
            "qwen_image_2_1_candle_{label}_{width}x{height}_{steps}steps_seed42.rgb"
        ));
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        assert_not_band_corrupted(label, &image.pixels, width, height);
        eprintln!(
            "{label}: {:.1}s total ({:.1}s render)",
            started.elapsed().as_secs_f32(),
            render_started.elapsed().as_secs_f32()
        );
    }
}
