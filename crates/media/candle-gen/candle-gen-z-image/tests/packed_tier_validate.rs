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

#[cfg(all(windows, feature = "cuda"))]
mod host_memory {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCountersEx,
            size: u32,
        ) -> i32;
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Peak {
        pub working_set_start: u64,
        pub working_set_peak: u64,
        pub working_set_end: u64,
        pub private_start: u64,
        pub private_peak: u64,
        pub private_end: u64,
    }

    fn counters() -> (u64, u64) {
        let mut counters = ProcessMemoryCountersEx {
            cb: std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
            private_usage: 0,
        };
        // SAFETY: the current-process pseudo handle is always valid and the struct/size match the API.
        let ok = unsafe {
            K32GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<ProcessMemoryCountersEx>() as u32,
            )
        };
        assert_ne!(ok, 0, "K32GetProcessMemoryInfo failed");
        (
            counters.working_set_size as u64,
            counters.private_usage as u64,
        )
    }

    fn update_max(target: &AtomicU64, value: u64) {
        target.fetch_max(value, Ordering::Relaxed);
    }

    pub fn sample<T>(f: impl FnOnce() -> T) -> (T, Peak) {
        let (working_set_start, private_start) = counters();
        let stop = Arc::new(AtomicBool::new(false));
        let working_set_peak = Arc::new(AtomicU64::new(working_set_start));
        let private_peak = Arc::new(AtomicU64::new(private_start));
        let sampler = {
            let stop = Arc::clone(&stop);
            let working_set_peak = Arc::clone(&working_set_peak);
            let private_peak = Arc::clone(&private_peak);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let (working_set, private) = counters();
                    update_max(&working_set_peak, working_set);
                    update_max(&private_peak, private);
                    std::thread::sleep(Duration::from_millis(5));
                }
            })
        };
        let result = f();
        stop.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        let (working_set_end, private_end) = counters();
        (
            result,
            Peak {
                working_set_start,
                working_set_peak: working_set_peak.load(Ordering::Relaxed),
                working_set_end,
                private_start,
                private_peak: private_peak.load(Ordering::Relaxed),
                private_end,
            },
        )
    }
}

#[cfg(all(not(windows), feature = "cuda"))]
mod host_memory {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Peak {
        pub working_set_start: u64,
        pub working_set_peak: u64,
        pub working_set_end: u64,
        pub private_start: u64,
        pub private_peak: u64,
        pub private_end: u64,
    }

    pub fn sample<T>(f: impl FnOnce() -> T) -> (T, Peak) {
        (f(), Peak::default())
    }
}
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
    quant: Quant,
    step_count: u32,
) -> (Image, Vec<f64>) {
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
            quant: Some(quant),
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
        steps: Some(step_count),
        guidance: Some(1.0),
        ..Default::default()
    };
    scope.configure_request(&mut request).unwrap();
    let started = Instant::now();
    let mut last_step = started;
    let mut step_seconds = Vec::new();
    let output = generator.generate(&request, &mut |progress| {
        if matches!(progress, Progress::Step { .. }) {
            let now = Instant::now();
            step_seconds.push(now.duration_since(last_step).as_secs_f64());
            last_step = now;
        }
    });
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
    (image, step_seconds)
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
    let (reference, _) = render_base_ladder(
        generator.as_ref(),
        MemoryStrategy::Resident,
        None,
        Quant::Q4,
        1,
    );
    assert_coherent(&reference, "base-q4-resident");
    for (strategy, window) in [
        (MemoryStrategy::StagedResidency, None),
        (MemoryStrategy::BoundedDecode, None),
        (MemoryStrategy::BoundedAttention, None),
        (MemoryStrategy::BoundedTransformerResidency, Some(1)),
    ] {
        let (adapted, _) = render_base_ladder(generator.as_ref(), strategy, window, Quant::Q4, 1);
        assert_eq!(
            adapted, reference,
            "real packed Z-Image output changed at {strategy:?} / window {window:?}"
        );
    }
}

#[test]
#[ignore = "needs Z_IMAGE_BASE_PACKED_Q8 + CUDA; proves q8 resident/sidecar output parity"]
fn packed_base_q8_rung_four_preserves_fixed_seed_output() {
    let tier = PathBuf::from(
        std::env::var("Z_IMAGE_BASE_PACKED_Q8")
            .expect("set Z_IMAGE_BASE_PACKED_Q8 to the z-image q8 tier directory"),
    );
    let spec = LoadSpec::new(WeightsSource::Dir(tier));
    let generator = candle_gen_z_image::provider_registry()
        .unwrap()
        .load("z_image", &spec)
        .expect("load packed Z-Image base q8");
    let (resident, _) = render_base_ladder(
        generator.as_ref(),
        MemoryStrategy::Resident,
        None,
        Quant::Q8,
        1,
    );
    let (streamed, _) = render_base_ladder(
        generator.as_ref(),
        MemoryStrategy::BoundedTransformerResidency,
        Some(1),
        Quant::Q8,
        1,
    );
    assert_eq!(streamed, resident, "real q8 sidecar output changed");
}

#[cfg(feature = "cuda")]
fn profile_sidecar_preparation(tier: &std::path::Path, tag: &str) {
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::Device;
    use candle_gen::quant::{PackedConfig, PackedWeightSidecars};
    use candle_gen::testkit::{used_mib, PeakSampler};

    let transformer = tier.join("transformer");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(transformer.join("config.json")).expect("read transformer config"),
    )
    .expect("parse transformer config");
    let packed = PackedConfig::from_config(&config).expect("packed transformer config");
    let files = candle_gen::sorted_safetensors(&transformer, "z-image profile")
        .expect("resolve transformer files");
    // SAFETY: the real-weight evidence files are immutable during the measurement.
    let source = unsafe { MmapedSafetensors::multi(&files).expect("mmap transformer") };
    let device = Device::new_cuda(0).expect("CUDA device");
    let baseline = used_mib(candle_gen::testkit::probe_gpu()).expect("nvidia-smi baseline");
    let gpu = PeakSampler::start_rendered();
    let started = Instant::now();
    let (prepared, host) = host_memory::sample(|| {
        PackedWeightSidecars::prepare_prefix_cancelable(
            &source,
            &transformer,
            packed,
            &device,
            &candle_gen::gen_core::CancelFlag::default(),
            "layers.",
        )
    });
    let elapsed = started.elapsed().as_secs_f64();
    let gpu_peak = gpu.stop();
    let prepared = prepared.expect("prepare device-format sidecars");
    const MIB: f64 = 1024.0 * 1024.0;
    eprintln!(
        "[sc-16510 {tag} preparation] {elapsed:.3}s; created={} reused={}; hashed={:.1} MiB \
         sidecars={:.1} MiB; host WS peak/end Δ={:.1}/{:.1} MiB; private peak/end Δ={:.1}/{:.1} \
         MiB; CUDA baseline/peak/Δ={baseline}/{gpu_peak}/{} MiB",
        prepared.created_count(),
        prepared.reused_count(),
        prepared.source_bytes_hashed() as f64 / MIB,
        prepared.sidecar_bytes() as f64 / MIB,
        host.working_set_peak.saturating_sub(host.working_set_start) as f64 / MIB,
        host.working_set_end.saturating_sub(host.working_set_start) as f64 / MIB,
        host.private_peak.saturating_sub(host.private_start) as f64 / MIB,
        host.private_end.saturating_sub(host.private_start) as f64 / MIB,
        gpu_peak.saturating_sub(baseline),
    );
    device.synchronize().expect("finish sidecar preparation");
}

#[cfg(feature = "cuda")]
fn profile_rung_four_steps(tier: &std::path::Path, quant: Quant, tag: &str) {
    use candle_gen::testkit::{used_mib, PeakSampler};

    let spec = LoadSpec::new(WeightsSource::Dir(tier.to_path_buf()));
    let generator = candle_gen_z_image::provider_registry()
        .unwrap()
        .load("z_image", &spec)
        .expect("load packed Z-Image base");
    let baseline = used_mib(candle_gen::testkit::probe_gpu()).expect("nvidia-smi baseline");
    let gpu = PeakSampler::start_rendered();
    let ((image, step_seconds), host) = host_memory::sample(|| {
        render_base_ladder(
            generator.as_ref(),
            MemoryStrategy::BoundedTransformerResidency,
            Some(1),
            quant,
            4,
        )
    });
    let gpu_peak = gpu.stop();
    assert_coherent(&image, tag);
    assert_eq!(step_seconds.len(), 4, "expected four timed denoise steps");
    let mut steady = step_seconds[1..].to_vec();
    steady.sort_by(f64::total_cmp);
    const MIB: f64 = 1024.0 * 1024.0;
    eprintln!(
        "[sc-16510 {tag} rung4] first-callback={:.3}s; steady steps={:?}; median/min/max={:.3}/{:.3}/{:.3}s; \
         host WS peak/end Δ={:.1}/{:.1} MiB; private peak/end Δ={:.1}/{:.1} MiB; CUDA \
         baseline/peak/Δ={baseline}/{gpu_peak}/{} MiB",
        step_seconds[0],
        &step_seconds[1..],
        steady[steady.len() / 2],
        steady[0],
        steady[steady.len() - 1],
        host.working_set_peak
            .saturating_sub(host.working_set_start) as f64
            / MIB,
        host.working_set_end
            .saturating_sub(host.working_set_start) as f64
            / MIB,
        host.private_peak.saturating_sub(host.private_start) as f64 / MIB,
        host.private_end.saturating_sub(host.private_start) as f64 / MIB,
        gpu_peak.saturating_sub(baseline),
    );
}

/// SC-16510 acceptance evidence on real packed Base weights. Run this test alone with one test thread
/// so q4/q8 preparation, host sampling, and device-level CUDA peaks do not overlap another test.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs Z_IMAGE_BASE_PACKED_Q4/Q8 + CUDA; records SC-16510 lifecycle evidence"]
fn sc16510_real_q4_q8_sidecar_lifecycle_evidence() {
    for (env, quant, tag) in [
        ("Z_IMAGE_BASE_PACKED_Q4", Quant::Q4, "q4"),
        ("Z_IMAGE_BASE_PACKED_Q8", Quant::Q8, "q8"),
    ] {
        let tier = PathBuf::from(std::env::var(env).unwrap_or_else(|_| panic!("set {env}")));
        profile_sidecar_preparation(&tier, tag);
        profile_rung_four_steps(&tier, quant, tag);
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
