//! The bounded real-weight validation render (sc-24109) — `#[ignore]`d **and** gated to a GPU
//! backend: it needs the pinned `Qwen/Qwen-Image-2.1` snapshot at
//! `CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT` (inference never self-fetches or derives a cache location,
//! epic 13657) and an accelerator, because the released weights are bf16 and candle's plain CPU
//! backend has no bf16 matmul — the same reason `candle-gen-ltx`'s conformance suite is gated.
//!
//! The gate is `any(cuda, metal)` rather than `cuda` alone so the macOS candle lane type-checks
//! this file too (`cargo check --features metal --all-targets`); CUDA is the lane the render is
//! meant for, and neither is reachable in ordinary CI — no runner provisions the 32 GB snapshot
//! (`release/real-weight-models.toml` records the key as deliberately unwired).
//!
//! Loads the released bf16 weights through the explicit catalog's production load path and renders
//! one image at the upstream default preset (1:1 2048×2048, 40 steps, seed 42, no guidance),
//! writing the raw RGB8 bytes to `QWEN_IMAGE_2_1_RENDER_OUT` (default: the current directory). It
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

#![cfg(any(feature = "cuda", feature = "metal"))]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use candle_gen_qwen_image_2_1::{load_tokenizer, system_prompt_drop_count, PRESETS};

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
    eprintln!(
        "wrote {} ({width}x{height} RGB8) after {:.1}s total ({:.1}s render)",
        path.display(),
        started.elapsed().as_secs_f32(),
        render_started.elapsed().as_secs_f32()
    );
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
/// ten references are ~41k prefix tokens on their own, and this port recomputes the whole
/// block-causal prefix every step (no KV cache — see UPSTREAM.md). The boundary case therefore
/// runs the fewest steps the sampler accepts rather than a smaller target: only the step count
/// moves its cost.
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
    use candle_gen::gen_core::Conditioning;

    let root = snapshot();
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

    // (references, width, height, steps) — the caller-shaped case, then the boundary.
    for (count, width, height, steps) in [(2usize, 1024u32, 1024u32, 8u32), (10, 512, 512, 2)] {
        let refs = references(count, 768, 768);
        let req = GenerationRequest {
            prompt: "Combine the subjects of the reference images into one scene, evening light"
                .to_owned(),
            width,
            height,
            steps: Some(steps),
            seed: Some(42),
            conditioning: vec![Conditioning::MultiReference { images: refs }],
            ..Default::default()
        };
        let render_started = Instant::now();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
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
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
        let path = out_dir.join(format!(
            "qwen_image_2_1_candle_edit_{count}refs_{width}x{height}_{steps}steps_seed42.rgb"
        ));
        std::fs::write(&path, &image.pixels).unwrap();
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
            conditioning: vec![Conditioning::MultiReference {
                images: references(11, 256, 256),
            }],
            ..Default::default()
        })
        .err()
        .map(|e| e.to_string())
        .expect("eleven references must be refused");
    eprintln!("eleven references: {err}");
    assert!(err.contains("at most 10"), "{err}");
}
