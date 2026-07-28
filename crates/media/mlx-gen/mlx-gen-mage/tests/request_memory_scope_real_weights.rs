//! Hardware-gated Mage request-memory lifecycle runner (SC-15507).
//!
//! This is intentionally ignored: it requires Apple MLX plus a complete Mage-Flow snapshot. It
//! exercises one loaded provider through A → B → A request scopes, then sets cancellation from a
//! progress callback (Mage observes it at the terminal post-trace check), injects an error outcome
//! after a real generation, and runs successful follow-ups. SceneWorks owns the actual cache labels
//! and selector; this runner validates the provider half of that boundary. Passing locally is
//! evidence that the seam executed; merely compiling is not a Verified hardware claim.

#![cfg(target_os = "macos")]

use mlx_gen::gen_core::{
    Error, ImageMemoryBudget, ImageMemoryCacheState, ImageMemoryGeometry, ImageMemoryMode,
    ImageMemoryNumericTier, ImageMemoryRunContext, ImageMemoryRunOutcome, ImageMemorySelection,
    ImageMemoryStrategy, ImageMemoryStrategyParameters,
};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, Generator, LoadSpec, Precision, Quant,
    WeightsSource,
};
use mlx_gen_mage::memory::{generation_peak_gb, production_safe_budget_gb};
use mlx_gen_mage::model::REGISTRATION;

fn snapshot() -> String {
    std::env::var("MAGE_REQUEST_SCOPE_SNAPSHOT")
        .expect("set MAGE_REQUEST_SCOPE_SNAPSHOT to a complete Mage-Flow snapshot")
}

fn quant() -> Option<Quant> {
    match std::env::var("MAGE_REQUEST_SCOPE_QUANT")
        .unwrap_or_else(|_| "bf16".to_owned())
        .as_str()
    {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        value => panic!("MAGE_REQUEST_SCOPE_QUANT must be q4, q8, or bf16, got {value}"),
    }
}

fn context(
    width: u32,
    height: u32,
    cache_state: ImageMemoryCacheState,
    tier: Option<Quant>,
) -> ImageMemoryRunContext {
    let safe_bytes = (production_safe_budget_gb().unwrap() * 1_000_000_000.0).round() as u64;
    let predicted_peak_bytes =
        (generation_peak_gb(tier, width, height, 1) * 1_000_000_000.0).round() as u64;
    ImageMemoryRunContext {
        selection: ImageMemorySelection {
            strategy: ImageMemoryStrategy::Resident,
            parameters: ImageMemoryStrategyParameters::default(),
            tier: ImageMemoryNumericTier {
                precision: Precision::Bf16,
                quant: tier,
            },
        },
        calibration_abi: mlx_gen::gen_core::IMAGE_MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: mlx_gen_mage::model::IMAGE_MEMORY_CALIBRATION_FINGERPRINT
            .to_owned(),
        mode: ImageMemoryMode::TextToImage,
        has_reference: false,
        use_pid: false,
        has_phases: false,
        geometry: ImageMemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
        },
        overlay: None,
        budget: ImageMemoryBudget {
            total_bytes: safe_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes,
        cache_state,
        evidence_revision: "sc-15507-real-apple-runner".to_owned(),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TerminalCase {
    Complete,
    CancelAfterProgress,
    InjectErrorAfterGeneration,
}

fn run_scoped(
    generator: &dyn Generator,
    width: u32,
    height: u32,
    seed: u64,
    cache_state: ImageMemoryCacheState,
    tier: Option<Quant>,
    terminal: TerminalCase,
) -> mlx_gen::gen_core::Result<Vec<u8>> {
    let context = context(width, height, cache_state, tier);
    let mut scope = generator
        .begin_image_memory_request(&context)?
        .expect("Mage must open its adopted resident request scope");
    let mut request = GenerationRequest {
        prompt: "a red ceramic cube on a white table".to_owned(),
        width,
        height,
        count: 1,
        steps: Some(2),
        guidance: Some(5.0),
        seed: Some(seed),
        cancel: CancelFlag::new(),
        ..Default::default()
    };
    if let Err(error) = scope.configure_request(&mut request) {
        let _ = scope.finish(ImageMemoryRunOutcome::Error {
            message: error.to_string(),
        });
        return Err(error);
    }
    let callback_cancel = request.cancel.clone();
    let mut saw_progress = false;
    let output = generator.generate(&request, &mut |_| {
        saw_progress = true;
        if terminal == TerminalCase::CancelAfterProgress {
            callback_cancel.cancel();
        }
    });
    if terminal == TerminalCase::CancelAfterProgress {
        assert!(
            saw_progress,
            "cancellation must happen after MLX work starts"
        );
    }
    let injected_error = terminal == TerminalCase::InjectErrorAfterGeneration && output.is_ok();
    let outcome = if injected_error {
        ImageMemoryRunOutcome::Error {
            message: "injected post-generation provider error".to_owned(),
        }
    } else {
        match &output {
            Ok(_) => ImageMemoryRunOutcome::Complete,
            Err(Error::Canceled) => ImageMemoryRunOutcome::Canceled,
            Err(error) => ImageMemoryRunOutcome::Error {
                message: error.to_string(),
            },
        }
    };
    scope.finish(outcome)?;
    if injected_error {
        return Err(Error::Msg(
            "injected post-generation provider error".to_owned(),
        ));
    }
    let GenerationOutput::Images(images) = output? else {
        panic!("Mage returned non-image output");
    };
    Ok(images.into_iter().next().unwrap().pixels)
}

#[test]
#[ignore = "requires Apple MLX and MAGE_REQUEST_SCOPE_SNAPSHOT real weights"]
fn one_loaded_provider_reapplies_a_b_a_and_recovers_after_terminal_failures() {
    let tier = quant();
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot().into()));
    spec.quantize = tier;
    let generator = (REGISTRATION.load)(&spec).unwrap();

    let a1 = run_scoped(
        generator.as_ref(),
        512,
        512,
        7,
        ImageMemoryCacheState::Cold,
        tier,
        TerminalCase::Complete,
    )
    .unwrap();
    let b = run_scoped(
        generator.as_ref(),
        1024,
        768,
        11,
        ImageMemoryCacheState::Warm,
        tier,
        TerminalCase::Complete,
    )
    .unwrap();
    let a2 = run_scoped(
        generator.as_ref(),
        512,
        512,
        7,
        ImageMemoryCacheState::Warm,
        tier,
        TerminalCase::Complete,
    )
    .unwrap();
    assert_eq!(a1, a2, "B must not poison the repeated A selection/output");
    assert_ne!(a1.len(), b.len(), "B exercised a distinct geometry");

    assert!(matches!(
        run_scoped(
            generator.as_ref(),
            512,
            512,
            13,
            ImageMemoryCacheState::Warm,
            tier,
            TerminalCase::CancelAfterProgress,
        ),
        Err(Error::Canceled)
    ));
    assert_eq!(
        mlx_rs::memory::get_cache_memory(),
        0,
        "scope finish must evict allocator-retained scratch after cancellation"
    );
    run_scoped(
        generator.as_ref(),
        512,
        512,
        17,
        ImageMemoryCacheState::Warm,
        tier,
        TerminalCase::Complete,
    )
    .expect("follow-up after canceled scope");

    assert!(matches!(
        run_scoped(
            generator.as_ref(),
            512,
            512,
            19,
            ImageMemoryCacheState::Warm,
            tier,
            TerminalCase::InjectErrorAfterGeneration,
        ),
        Err(Error::Msg(message)) if message.contains("injected post-generation")
    ));
    assert_eq!(
        mlx_rs::memory::get_cache_memory(),
        0,
        "scope finish must evict allocator-retained scratch after an error"
    );
    run_scoped(
        generator.as_ref(),
        512,
        512,
        23,
        ImageMemoryCacheState::Warm,
        tier,
        TerminalCase::Complete,
    )
    .expect("follow-up after injected provider error");
}
