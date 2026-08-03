//! SC-15509 serial real-weight proof for Mage-Flow's MLX shared image-memory ladder.
//!
//! The runner measures one exact provider/tier/mode per process so two Mage models never contend
//! for Metal. It executes cold + warm controls for every shared selection, then injects a typed
//! cancellation and a decode-boundary fault into rung 4 and proves a successful recovery after
//! each. No artifact is downloaded; callers pass exact already-cached snapshot roots.
//!
//! ```text
//! MAGE_LADDER_PROVIDER=mage_flow_edit MAGE_LADDER_MODE=edit MAGE_LADDER_TIER=q4 \
//! MAGE_LADDER_VARIANT_ROOT=/.../SceneWorks--Mage-Flow-Edit/snapshots/<sha> \
//! MAGE_LADDER_COMPONENTS_ROOT=/.../Mage-Flow-Components-mlx/snapshots/<sha> \
//! cargo test -p mlx-gen-mage --release --test shared_memory_ladder_real_weights \
//!   -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use mlx_gen::gen_core::{
    MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryRunContext, MemoryRunOutcome, MemoryStrategy, MemoryStrategySupport,
};
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Generator, Image, LoadShape, LoadSpec,
    Precision, Progress, Quant, WeightsSource,
};
use mlx_gen_mage::model::{COMPONENT_TEXT_ENCODER, COMPONENT_VAE};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_memory_limit, get_peak_memory,
    reset_peak_memory,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("set {key}; see this test's module documentation"))
}

fn mode() -> MemoryMode {
    match env("MAGE_LADDER_MODE").as_str() {
        "t2i" => MemoryMode::TextToImage,
        "edit" => MemoryMode::Edit,
        other => panic!("MAGE_LADDER_MODE must be t2i or edit, got {other}"),
    }
}

fn tier() -> Option<Quant> {
    match env("MAGE_LADDER_TIER").as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("MAGE_LADDER_TIER must be q4, q8, or bf16, got {other}"),
    }
}

fn size() -> u32 {
    std::env::var("MAGE_LADDER_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(768)
}

fn spec() -> LoadSpec {
    let variant = PathBuf::from(env("MAGE_LADDER_VARIANT_ROOT"));
    let components = PathBuf::from(env("MAGE_LADDER_COMPONENTS_ROOT"));
    let tier_name = match tier() {
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        None => "bf16",
        Some(other) => panic!("unsupported Mage ladder tier {other:?}"),
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(variant.join(tier_name)))
        .with_component(
            COMPONENT_TEXT_ENCODER,
            WeightsSource::Dir(components.join(tier_name).join("text_encoder")),
        )
        .with_component(
            COMPONENT_VAE,
            WeightsSource::Dir(components.join(tier_name).join("vae")),
        )
        .with_load_shape(LoadShape::DeferredMaterialization);
    if let Some(quant) = tier() {
        spec = spec.with_quant(quant);
    }
    spec
}

fn reference_image(edge: u32) -> Image {
    let mut pixels = vec![0_u8; edge as usize * edge as usize * 3];
    for y in 0..edge {
        for x in 0..edge {
            let offset = (y as usize * edge as usize + x as usize) * 3;
            let wall = 160_u8.saturating_add(((x + y) % 64) as u8);
            pixels[offset..offset + 3].copy_from_slice(&[wall, wall.saturating_sub(12), 150]);
            if x > edge / 3 && x < edge * 2 / 3 && y > edge / 3 && y < edge * 2 / 3 {
                pixels[offset..offset + 3].copy_from_slice(&[205, 38, 32]);
            }
        }
    }
    Image {
        width: edge,
        height: edge,
        pixels,
    }
}

fn base_request() -> GenerationRequest {
    let edge = size();
    let edit = mode() == MemoryMode::Edit;
    GenerationRequest {
        prompt: if edit {
            "change the red square into a blue ceramic sphere on the same background".into()
        } else {
            "a red fox in a snowy pine forest, detailed photograph, soft morning light".into()
        },
        negative_prompt: Some("blurry, distorted, low contrast".into()),
        width: edge,
        height: edge,
        count: 1,
        steps: Some(1),
        guidance: Some(5.0),
        seed: Some(15509),
        conditioning: if edit {
            vec![Conditioning::Reference {
                image: reference_image(edge),
                strength: None,
            }]
        } else {
            Vec::new()
        },
        ..Default::default()
    }
}

fn selection(
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::MemorySelection {
    let mut selection = contract
        .representative_selection(
            strategy,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: tier(),
                component_precision_floors: (mlx_gen_mage::model::REGISTRATION.descriptor)()
                    .capabilities
                    .component_precision_floors,
            },
            false,
        )
        .unwrap_or_else(|error| panic!("representative {strategy:?} selection: {error}"));
    if contract.engages(strategy, MemoryStrategy::BoundedDecode) {
        selection.parameters.decode_tile_edge = Some(512);
        selection.parameters.decode_overlap = Some(mlx_gen_mage::model::DECODE_OVERLAP);
    }
    contract
        .validate_selection(&selection)
        .unwrap_or_else(|error| panic!("validate {strategy:?} selection: {error}"));
    selection
}

fn context(
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    strategy: MemoryStrategy,
    cache_state: MemoryCacheState,
) -> MemoryRunContext {
    let selection = selection(contract, strategy);
    let edge = size();
    let required = (mlx_gen_mage::memory::generation_peak_gb(tier(), edge, edge, 1)
        * 1_000_000_000.0)
        .round() as u64;
    MemoryRunContext {
        selection,
        calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: mlx_gen_mage::model::MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
        load_shape: LoadShape::DeferredMaterialization,
        mode: mode(),
        has_reference: mode() == MemoryMode::Edit,
        use_pid: false,
        has_phases: contract.engages(strategy, MemoryStrategy::StagedResidency),
        geometry: MemoryGeometry {
            width: edge,
            height: edge,
            batch: 1,
            frames: 1,
            reference_count: u32::from(mode() == MemoryMode::Edit),
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: get_memory_limit() as u64,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: required,
        cache_state,
        evidence_revision: "sc-15509-mage-real-metal-v1".to_owned(),
    }
}

#[derive(Debug)]
struct Retention {
    active: usize,
    cache: usize,
}

#[derive(Debug)]
struct Run {
    image: Image,
    peak: usize,
    immediate: Retention,
    after_finish: Retention,
    after_clear: Retention,
    wall: Duration,
}

fn retention() -> Retention {
    Retention {
        active: get_active_memory(),
        cache: get_cache_memory(),
    }
}

fn image(output: GenerationOutput) -> Image {
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected image output, got {other:?}"),
    }
}

fn run_success(
    generator: &dyn Generator,
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    strategy: MemoryStrategy,
    cache_state: MemoryCacheState,
) -> Run {
    let context = context(contract, strategy, cache_state);
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .expect("begin request scope")
        .expect("Mage request scope");
    let mut request = base_request();
    scope
        .configure_request(&mut request)
        .expect("configure request");
    if contract.engages(strategy, MemoryStrategy::BoundedDecode) {
        let memory = request.memory.expect("bounded request memory");
        assert!(memory.tile_vae_decode);
        assert_eq!(memory.decode_tile_edge, Some(512));
        assert!(request.width > 512 && request.height > 512);
    }
    if contract.engages(strategy, MemoryStrategy::BoundedAttention) {
        let memory = request.memory.expect("bounded request memory");
        assert!(memory.chunk_attention);
        assert_eq!(
            memory.attention_chunk_size,
            Some(mlx_gen_mage::model::ATTENTION_CHUNK_SIZE)
        );
        let latent_tokens = (request.width / 16) * (request.height / 16);
        let cfg_fused_image_tokens = u64::from(latent_tokens) * 2;
        assert!(
            cfg_fused_image_tokens * cfg_fused_image_tokens
                > u64::from(mlx_gen_mage::model::ATTENTION_CHUNK_SIZE),
            "calibration geometry must physically chunk even before text/reference tokens"
        );
    }
    if strategy == MemoryStrategy::BoundedTransformerResidency {
        let memory = request.memory.expect("rung-4 request memory");
        assert!(memory.stage_residency && memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(1));
        assert_eq!(
            memory.transformer_window_component,
            Some(mlx_gen::gen_core::TransformerComponent::Both)
        );
    }
    clear_cache();
    reset_peak_memory();
    let started = Instant::now();
    let output = generator
        .generate(&request, &mut |_| {})
        .unwrap_or_else(|error| panic!("{strategy:?} generation: {error}"));
    let wall = started.elapsed();
    let peak = get_peak_memory();
    let immediate = retention();
    scope
        .finish(MemoryRunOutcome::Complete)
        .expect("finish scope");
    let after_finish = retention();
    clear_cache();
    let after_clear = retention();
    Run {
        image: image(output),
        peak,
        immediate,
        after_finish,
        after_clear,
        wall,
    }
}

fn run_terminal(
    generator: &dyn Generator,
    contract: &mlx_gen::gen_core::MemoryProviderContract,
    cancel: bool,
) -> (usize, Retention, Retention, Retention) {
    let strategy = MemoryStrategy::BoundedTransformerResidency;
    let context = context(contract, strategy, MemoryCacheState::Warm);
    let mut scope = generator
        .begin_memory_strategy_request(&context)
        .expect("begin terminal scope")
        .expect("Mage request scope");
    let mut request = base_request();
    scope
        .configure_request(&mut request)
        .expect("configure terminal request");
    if !cancel {
        request
            .memory
            .as_mut()
            .expect("rung 4 memory")
            .authorize_calibration_fault(MemoryPhase::Decode);
    }
    let flag = request.cancel.clone();
    clear_cache();
    reset_peak_memory();
    let output = generator.generate(&request, &mut |progress| {
        if cancel && matches!(progress, Progress::Step { .. }) {
            flag.cancel();
        }
    });
    match (cancel, &output) {
        (true, Err(mlx_gen::gen_core::Error::Canceled)) => {}
        (false, Err(mlx_gen::gen_core::Error::Msg(message)))
            if message.contains("authorized calibration fault") => {}
        _ => panic!("unexpected terminal result: {output:?}"),
    }
    let peak = get_peak_memory();
    let immediate = retention();
    let outcome = if cancel {
        MemoryRunOutcome::Canceled
    } else {
        MemoryRunOutcome::Error {
            message: output.unwrap_err().to_string(),
        }
    };
    scope.finish(outcome).expect("finish terminal scope");
    let after_finish = retention();
    clear_cache();
    let after_clear = retention();
    (peak, immediate, after_finish, after_clear)
}

fn delta(a: &Image, b: &Image) -> (u8, f64) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let mut max = 0_u8;
    let mut sum = 0_u64;
    for (left, right) in a.pixels.iter().zip(&b.pixels) {
        let d = left.abs_diff(*right);
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / a.pixels.len() as f64)
}

fn correlation(a: &Image, b: &Image) -> f64 {
    let luma = |pixel: &[u8]| {
        0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
    };
    let left = a.pixels.chunks_exact(3).map(luma).collect::<Vec<_>>();
    let right = b.pixels.chunks_exact(3).map(luma).collect::<Vec<_>>();
    let lm = left.iter().sum::<f64>() / left.len() as f64;
    let rm = right.iter().sum::<f64>() / right.len() as f64;
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for (x, y) in left.iter().zip(right.iter()) {
        let (x, y) = (x - lm, y - rm);
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    xy / (xx * yy).sqrt()
}

fn print_run(label: &str, run: &Run) {
    println!(
        "{label:<18} peak={:.3} GiB immediate={:.1}+{:.1} MiB finish={:.1}+{:.1} MiB clear={:.1}+{:.1} MiB wall={:.2}s",
        run.peak as f64 / GIB,
        run.immediate.active as f64 / MIB,
        run.immediate.cache as f64 / MIB,
        run.after_finish.active as f64 / MIB,
        run.after_finish.cache as f64 / MIB,
        run.after_clear.active as f64 / MIB,
        run.after_clear.cache as f64 / MIB,
        run.wall.as_secs_f64(),
    );
}

fn total_retained(retention: &Retention) -> usize {
    retention.active.saturating_add(retention.cache)
}

fn within_two_percent(actual: usize, baseline: usize, what: &str) {
    let limit = (baseline as u128 * 102).div_ceil(100);
    assert!(
        actual as u128 <= limit,
        "{what}: {actual} bytes exceeds clean-warm baseline {baseline} by more than 2%"
    );
}

#[test]
#[ignore = "needs cached Mage weights and one authorized Apple/Metal device"]
fn serial_shared_ladder_and_terminal_recovery() {
    let provider = env("MAGE_LADDER_PROVIDER");
    let spec = spec();
    let generator = mlx_gen_mage::provider_registry()
        .expect("Mage registry")
        .load(&provider, &spec)
        .unwrap_or_else(|error| panic!("load {provider}: {error}"));
    let contract = generator
        .memory_strategy_contract()
        .expect("Mage memory contract")
        .clone();
    for strategy in MemoryStrategy::ALL {
        assert_eq!(
            contract.capability(strategy).unwrap().support,
            MemoryStrategySupport::Implemented,
            "{strategy:?} must be loadable for this exact artifact"
        );
    }

    println!(
        "\nMage MLX shared ladder provider={provider} mode={} tier={:?} size={} load_shape={:?}",
        mode().as_key(),
        tier(),
        size(),
        contract.load_shape
    );
    println!("composition/parameters are contract-derived; one model is loaded at a time");

    let mut runs = Vec::new();
    for strategy in MemoryStrategy::ALL {
        let first = run_success(
            generator.as_ref(),
            &contract,
            strategy,
            MemoryCacheState::Cold,
        );
        let warm = run_success(
            generator.as_ref(),
            &contract,
            strategy,
            MemoryCacheState::Warm,
        );
        print_run(&format!("{strategy:?} cold"), &first);
        print_run(&format!("{strategy:?} warm"), &warm);
        assert_eq!(
            first.image.pixels, warm.image.pixels,
            "cold/warm output drift"
        );
        runs.push((strategy, first, warm));
    }

    let resident = &runs[0].1.image;
    for (strategy, run, _) in &runs[1..] {
        let (max, mean) = delta(resident, &run.image);
        let corr = correlation(resident, &run.image);
        println!("{strategy:?} vs Resident: max={max} mean={mean:.6} luma_corr={corr:.6}");
        if *strategy == MemoryStrategy::StagedResidency {
            assert_eq!(max, 0, "phase staging must be byte-identical");
        } else {
            assert!(
                corr >= 0.99,
                "{strategy:?} quality/parity correlation {corr:.6} is below 0.99"
            );
        }
    }

    let rung4 = &runs.last().unwrap().1;
    let rung4_warm = &runs.last().unwrap().2;
    assert!(
        rung4.peak < runs[0].1.peak,
        "rung 4 must lower the request peak: {:.3} vs {:.3} GiB",
        rung4.peak as f64 / GIB,
        runs[0].1.peak as f64 / GIB
    );
    for cancel in [true, false] {
        let (peak, immediate, finish, clear) = run_terminal(generator.as_ref(), &contract, cancel);
        println!(
            "terminal={} peak={:.3} GiB immediate={:.1}+{:.1} MiB finish={:.1}+{:.1} MiB clear={:.1}+{:.1} MiB",
            if cancel { "cancel" } else { "decode_fault" },
            peak as f64 / GIB,
            immediate.active as f64 / MIB,
            immediate.cache as f64 / MIB,
            finish.active as f64 / MIB,
            finish.cache as f64 / MIB,
            clear.active as f64 / MIB,
            clear.cache as f64 / MIB,
        );
        within_two_percent(
            total_retained(&immediate),
            total_retained(&rung4_warm.immediate),
            "terminal immediate active+cache",
        );
        within_two_percent(
            total_retained(&clear),
            total_retained(&rung4_warm.after_clear),
            "terminal post-clear active+cache",
        );
        let recovery = run_success(
            generator.as_ref(),
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
            MemoryCacheState::Warm,
        );
        print_run("terminal recovery", &recovery);
        assert_eq!(recovery.image.pixels, rung4.image.pixels);
        within_two_percent(recovery.peak, rung4_warm.peak, "terminal recovery peak");
        within_two_percent(
            total_retained(&recovery.after_clear),
            total_retained(&rung4_warm.after_clear),
            "terminal recovery post-clear active+cache",
        );
        println!(
            "TERMINAL_RESULT kind={} clean_warm_peak_bytes={} clean_warm_immediate_active_bytes={} clean_warm_immediate_cache_bytes={} immediate_active_bytes={} immediate_cache_bytes={} post_clear_active_bytes={} post_clear_cache_bytes={} recovery_peak_bytes={} recovery_post_clear_active_bytes={} recovery_post_clear_cache_bytes={}",
            if cancel { "cancel" } else { "decode_fault" },
            rung4_warm.peak,
            rung4_warm.immediate.active,
            rung4_warm.immediate.cache,
            immediate.active,
            immediate.cache,
            clear.active,
            clear.cache,
            recovery.peak,
            recovery.after_clear.active,
            recovery.after_clear.cache,
        );
    }
    println!(
        "RESULT status=pass provider={provider} mode={} tier={:?} fingerprint={} load_shape={:?} size={} decode_edge=512 decode_overlap=256 attention_chunk_size={} transformer_window_size=1 transformer_component=both resident_peak_bytes={} rung4_peak_bytes={} rung4_post_clear_active_bytes={} rung4_post_clear_cache_bytes={}",
        mode().as_key(),
        tier(),
        mlx_gen_mage::model::MEMORY_CALIBRATION_FINGERPRINT,
        contract.load_shape,
        size(),
        mlx_gen_mage::model::ATTENTION_CHUNK_SIZE,
        runs[0].1.peak,
        rung4.peak,
        rung4_warm.after_clear.active,
        rung4_warm.after_clear.cache,
    );
}
