//! Z-Image **packed-tier** real-weight GPU validation (sc-9408, sc-9089 umbrella).
//!
//! Loads the pre-quantized MLX-packed `SceneWorks/z-image-turbo-mlx` q4 (and, when present, q8) tier
//! **directly from the packed parts** (no dense bf16 staging) through the registered `z_image_turbo`
//! generator, and asserts it renders a **coherent, non-degenerate** image — the end-to-end proof that
//! the packed-detect path fired (a silent fall-back to dense would fail to load the u32-packed weights,
//! and a broken packed forward would render solid black / NaN).
//!
//! `#[ignore]`d (needs a real GPU + the cached packed tier). On the Windows/Blackwell box (v143 vcvars
//! + CUDA on PATH), point at the **tier subdir** (the packed snapshot nests `bf16/`, `q4/`, `q8/`):
//!
//! ```text
//! set Z_IMAGE_PACKED_Q4=D:\.cache\huggingface\hub\models--SceneWorks--z-image-turbo-mlx\snapshots\<hash>\q4
//! set Z_IMAGE_PACKED_Q8=...\q8    (optional)
//! cargo test -p candle-gen-z-image --features cuda --release --test packed_tier_validate -- --ignored --nocapture
//! ```
#![cfg(any(feature = "cuda", feature = "metal"))]

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Image, LoadSpec, MemoryBudget,
    MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryRunContext,
    MemoryRunOutcome, MemorySelection, MemoryStrategy, MemoryStrategyParameters, Precision,
    Progress, Quant, WeightsSource,
};

use candle_gen_z_image::{ZImageControl, ZImageControlPaths, ZImageControlRequest};

/// Basic non-degeneracy: the render is not solid-black / constant (a broken packed forward — NaN or
/// zeroed activations — decodes to a flat image), and has some spread of pixel values.
fn assert_coherent(img: &Image, tag: &str) {
    assert_eq!(
        img.pixels.len(),
        (img.width * img.height * 3) as usize,
        "{tag}: RGB buffer size mismatch"
    );
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / img.pixels.len() as f64;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / img.pixels.len() as f64;
    eprintln!(
        "[{tag}] {}x{} pixel min={min} max={max} mean={mean:.1} std={:.1}",
        img.width,
        img.height,
        var.sqrt()
    );
    assert!(
        max > min + 16,
        "{tag}: render is (near-)constant [{min}, {max}] — packed forward likely degenerate (black)"
    );
    assert!(
        var.sqrt() > 8.0,
        "{tag}: pixel std {:.1} too low — degenerate render",
        var.sqrt()
    );
}

fn render_tier(env: &str, tag: &str) {
    let Ok(dir) = std::env::var(env) else {
        eprintln!("SKIP {tag}: set {env} to the packed tier subdir");
        return;
    };
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir)));
    let gen = candle_gen_z_image::provider_registry()
        .unwrap()
        .load("z_image_turbo", &spec)
        .expect("z_image_turbo registered");

    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(4),
        ..Default::default()
    };

    let t = Instant::now();
    let mut on_progress = |_p: Progress| {};
    let out = gen
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("{tag}: packed-tier generate failed: {e}"));
    let secs = t.elapsed().as_secs_f32();
    eprintln!("[{tag}] load+render wall-clock {secs:.1}s (cold: includes packed load)");

    let images = match out {
        GenerationOutput::Images(imgs) => imgs,
        _ => panic!("{tag}: expected images"),
    };
    assert_eq!(images.len(), 1, "{tag}: expected 1 image");
    assert_coherent(&images[0], tag);

    // The same warm generator must honor a staged second request without changing fixed-seed pixels.
    let staged = GenerationRequest {
        memory: Some(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        }),
        ..req
    };
    let staged_out = gen
        .generate(&staged, &mut on_progress)
        .unwrap_or_else(|e| panic!("{tag}: request-staged generate failed: {e}"));
    let staged_images = match staged_out {
        GenerationOutput::Images(images) => images,
        _ => panic!("{tag}: expected staged image output"),
    };
    assert_eq!(staged_images.len(), 1, "{tag}: expected 1 staged image");
    assert_eq!(
        staged_images[0].pixels, images[0].pixels,
        "{tag}: warm and request-staged fixed-seed pixels diverged"
    );

    // Write the render next to the tier so it can be eyeballed.
    if let Some(buf) =
        image::RgbImage::from_raw(images[0].width, images[0].height, images[0].pixels.clone())
    {
        let out_path = std::env::temp_dir().join(format!("z_image_packed_{tag}.png"));
        let _ = buf.save(&out_path);
        eprintln!("[{tag}] wrote {}", out_path.display());
    }
}

fn render_base_ladder(
    generator: &dyn candle_gen::gen_core::Generator,
    strategy: MemoryStrategy,
    transformer_window_size: Option<u32>,
) -> Image {
    let contract = generator
        .memory_strategy_contract()
        .expect("CUDA Z-Image base must publish a memory contract");
    let parameters = match strategy {
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
            MemoryStrategyParameters::default()
        }
        MemoryStrategy::BoundedDecode => MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            ..Default::default()
        },
        MemoryStrategy::BoundedAttention => MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            attention_chunk_size: Some(64 * 1024 * 1024),
            ..Default::default()
        },
        MemoryStrategy::BoundedTransformerResidency => MemoryStrategyParameters {
            decode_tile_edge: Some(512),
            decode_overlap: Some(128),
            attention_chunk_size: Some(64 * 1024 * 1024),
            transformer_window_size,
            ..Default::default()
        },
    };
    let selection = MemorySelection {
        strategy,
        parameters,
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        },
    };
    contract.validate_selection(&selection).unwrap();
    let calibration = contract.calibration.as_ref().unwrap();
    let context = MemoryRunContext {
        selection,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: MemoryGeometry {
            width: 256,
            height: 256,
            batch: 1,
            frames: 1,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: u64::MAX,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "sc-15815-real-weight-conformance".to_owned(),
    };
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .unwrap()
        .expect("Z-Image contract selection must create a request scope");
    if contract.engages(strategy, MemoryStrategy::BoundedDecode) {
        scope.configure_decode(512, 128, context.geometry).unwrap();
    }
    if contract.engages(strategy, MemoryStrategy::BoundedAttention) {
        scope.configure_attention(64 * 1024 * 1024).unwrap();
    }
    if let Some(window) = transformer_window_size {
        scope.materialize_transformer_window(0, window).unwrap();
    }
    let mut request = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
        width: 256,
        height: 256,
        count: 1,
        seed: Some(15815),
        steps: Some(1),
        guidance: Some(1.0),
        ..Default::default()
    };
    scope.configure_request(&mut request).unwrap();
    let output = generator.generate(&request, &mut |_| {});
    let image = match output {
        Ok(GenerationOutput::Images(mut images)) => {
            assert_eq!(images.len(), 1);
            images.remove(0)
        }
        Ok(other) => panic!("expected image output, got {other:?}"),
        Err(error) => {
            let message = error.to_string();
            scope
                .finish(MemoryRunOutcome::Error {
                    message: message.clone(),
                })
                .unwrap();
            panic!("real-weight ladder render failed: {message}");
        }
    };
    scope.finish(MemoryRunOutcome::Complete).unwrap();
    image
}

#[test]
#[ignore = "needs Z_IMAGE_BASE_PACKED_Q4 + CUDA; exercises every implemented production rung"]
fn packed_base_all_rungs_preserve_fixed_seed_output() {
    let tier = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_PACKED_Q4")
            .expect("set Z_IMAGE_BASE_PACKED_Q4 to the z-image q4 tier directory"),
    );
    let spec = LoadSpec::new(WeightsSource::Dir(tier));
    let generator = candle_gen_z_image::provider_registry()
        .unwrap()
        .load("z_image", &spec)
        .expect("load packed Z-Image base");
    let reference = render_base_ladder(generator.as_ref(), MemoryStrategy::Resident, None);
    assert_coherent(&reference, "base-q4-resident");
    for (strategy, window) in [
        (MemoryStrategy::StagedResidency, None),
        (MemoryStrategy::BoundedDecode, None),
        (MemoryStrategy::BoundedAttention, None),
        (MemoryStrategy::BoundedTransformerResidency, Some(1)),
    ] {
        let adapted = render_base_ladder(generator.as_ref(), strategy, window);
        assert_eq!(
            adapted, reference,
            "real packed Z-Image output changed at {strategy:?} / window {window:?}"
        );
    }
}

#[test]
#[ignore = "needs Z_IMAGE_BASE_PACKED_Q4 + Z_IMAGE_BASE_CONTROL + CUDA"]
fn packed_base_control_q4_loads_without_dense_shape_mismatch() {
    load_packed_base_control("Z_IMAGE_BASE_PACKED_Q4", "q4");
}

#[test]
#[ignore = "needs Z_IMAGE_BASE_PACKED_Q8 + Z_IMAGE_BASE_CONTROL + CUDA"]
fn packed_base_control_q8_loads_without_dense_shape_mismatch() {
    load_packed_base_control("Z_IMAGE_BASE_PACKED_Q8", "q8");
}

fn load_packed_base_control(snapshot_env: &str, tier: &str) {
    let snapshot = PathBuf::from(
        std::env::var(snapshot_env)
            .unwrap_or_else(|_| panic!("set {snapshot_env} to the z-image {tier} tier directory")),
    );
    let control = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_CONTROL")
            .expect("set Z_IMAGE_BASE_CONTROL to the base control checkpoint"),
    );

    ZImageControl::load(&ZImageControlPaths {
        snapshot,
        control,
        base: true,
    })
    .unwrap_or_else(|error| {
        panic!("packed {tier} base-control provider must load through packed-aware TE and DiT seams: {error}")
    });
}

fn control_fixture(width: u32, height: u32) -> Image {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    let mut set = |x: u32, y: u32| {
        if x < width && y < height {
            let offset = ((y * width + x) * 3) as usize;
            pixels[offset..offset + 3].fill(255);
        }
    };
    let cx = width / 2;
    for y in height / 8..height * 7 / 8 {
        for dx in 0..3 {
            set(cx + dx, y);
        }
    }
    for x in width / 4..width * 3 / 4 {
        for dy in 0..3 {
            set(x, height / 3 + dy);
        }
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn mean_abs_diff(lhs: &Image, rhs: &Image) -> f64 {
    assert_eq!((lhs.width, lhs.height), (rhs.width, rhs.height));
    lhs.pixels
        .iter()
        .zip(&rhs.pixels)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .sum::<f64>()
        / lhs.pixels.len() as f64
}

#[test]
#[ignore = "needs Z_IMAGE_BASE_PACKED_Q4 + Z_IMAGE_BASE_CONTROL + CUDA"]
fn packed_base_control_q4_honors_control_cfg_warm_repeat_and_cleanup() {
    let snapshot = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_PACKED_Q4")
            .expect("set Z_IMAGE_BASE_PACKED_Q4 to the z-image q4 tier directory"),
    );
    let control = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_CONTROL")
            .expect("set Z_IMAGE_BASE_CONTROL to the base control checkpoint"),
    );
    let model = ZImageControl::load(&ZImageControlPaths {
        snapshot,
        control,
        base: true,
    })
    .expect("load packed q4 base-control provider");
    let control_image = control_fixture(256, 256);
    let request = ZImageControlRequest {
        prompt: "a studio photograph of a dancer, full body, crisp details".into(),
        width: 256,
        height: 256,
        steps: 2,
        control_scale: 1.0,
        guidance: Some(4.0),
        negative_prompt: Some("blurry, malformed".into()),
        seed: 16170,
        use_pid: false,
        cancel: candle_gen::gen_core::CancelFlag::new(),
    };
    let render = |request: &ZImageControlRequest, control_image: &Image| {
        model
            .generate(request, control_image, &mut |_| {})
            .expect("base-control render")
    };

    let controlled = render(&request, &control_image);
    assert_coherent(&controlled, "base-control-q4");
    let warm_repeat = render(&request, &control_image);
    assert_eq!(warm_repeat, controlled, "fixed-seed warm repeat changed");

    let uncontrolled = render(
        &ZImageControlRequest {
            control_scale: 0.0,
            ..request.clone()
        },
        &control_image,
    );
    let control_delta = mean_abs_diff(&controlled, &uncontrolled);
    assert!(
        control_delta > 0.1,
        "control_scale was ignored (mean absolute delta {control_delta:.4})"
    );

    let alternate_cfg = render(
        &ZImageControlRequest {
            guidance: Some(1.0),
            negative_prompt: Some("this branch must be ignored at guidance one".into()),
            ..request.clone()
        },
        &control_image,
    );
    let cfg_delta = mean_abs_diff(&controlled, &alternate_cfg);
    assert!(
        cfg_delta > 0.1,
        "base guidance/negative-prompt fields were ignored (mean absolute delta {cfg_delta:.4})"
    );

    let invalid = model.generate(&request, &control_fixture(128, 256), &mut |_| {});
    assert!(invalid.is_err(), "wrong control geometry must fail loudly");
    assert_eq!(
        render(&request, &control_image),
        controlled,
        "render changed after a recoverable control-input error"
    );

    let cancelled = ZImageControlRequest {
        cancel: {
            let cancel = candle_gen::gen_core::CancelFlag::new();
            cancel.cancel();
            cancel
        },
        ..request
    };
    let error = model
        .generate(&cancelled, &control_image, &mut |_| {})
        .expect_err("pre-cancel must stop before generation");
    assert!(matches!(error, candle_gen::CandleError::Canceled));
}

/// The q4 packed tier renders a coherent image straight from the packed parts (the primary sc-9408
/// deliverable — the tier is cached, so this is the routine GPU check).
#[test]
#[ignore = "needs Z_IMAGE_PACKED_Q4 (packed q4 tier subdir) + a CUDA/Metal GPU; run with the matching feature --ignored"]
fn packed_q4_renders_coherent() {
    render_tier("Z_IMAGE_PACKED_Q4", "q4");
}

/// The q8 packed tier renders a coherent image (double-quant Q8_0 path); only runs when the q8 tier is
/// present locally.
#[test]
#[ignore = "needs Z_IMAGE_PACKED_Q8 (packed q8 tier subdir) + a CUDA/Metal GPU; run with the matching feature --ignored"]
fn packed_q8_renders_coherent() {
    render_tier("Z_IMAGE_PACKED_Q8", "q8");
}
