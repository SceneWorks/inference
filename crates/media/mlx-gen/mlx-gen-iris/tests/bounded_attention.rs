//! Tiny, CPU-only parity for the request-selected kernel (no model weights).
use mlx_gen::{AttentionBudget, AttentionPlan};
use mlx_gen_iris::nn::attention_budgeted;
use mlx_rs::{Array, Dtype};

#[test]
fn bounded_attention_preserves_gqa_mha_and_query_masks() {
    crate::common::cpu_process(
        "bounded_attention::bounded_attention_preserves_gqa_mha_and_query_masks",
        run,
    );
}

fn run() {
    for (heads, kv_heads) in [(4, 1), (4, 2), (4, 4)] {
        let q = Array::from_slice(
            &(0..7 * heads * 8)
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
            &[1, 7, heads, 8],
        );
        let k = Array::from_slice(
            &(0..7 * kv_heads * 8)
                .map(|i| (i as f32 * 0.17).cos())
                .collect::<Vec<_>>(),
            &[1, 7, kv_heads, 8],
        );
        let v = k.multiply(Array::from_f32(0.7)).unwrap();
        let mask = Array::from_slice(
            &(0..49)
                .map(|i| if i % 7 > i / 7 { -10000.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            &[1, 1, 7, 7],
        );
        for mask in [None, Some(&mask)] {
            let full =
                attention_budgeted(&q, &k, &v, Dtype::Float32, mask, AttentionPlan::UNBOUNDED)
                    .unwrap();

            let bounded = attention_budgeted(
                &q,
                &k,
                &v,
                Dtype::Float32,
                mask,
                AttentionPlan::budgeted(AttentionBudget::from_score_elements(28, true)),
            )
            .unwrap();
            let error = full
                .subtract(&bounded)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(error < 2e-5, "{heads}/{kv_heads}: {error}");
            // A cancelled bounded call must enter the chunk path, not silently run unbounded.
            let cancel = mlx_gen::CancelFlag::new();
            cancel.cancel();
            assert!(matches!(
                attention_budgeted(
                    &q,
                    &k,
                    &v,
                    Dtype::Float32,
                    mask,
                    AttentionPlan::budgeted(AttentionBudget::from_score_elements(28, true))
                        .with_cancel(&cancel)
                ),
                Err(mlx_gen::Error::Canceled)
            ));
            assert!(attention_budgeted(
                &q,
                &k,
                &v,
                Dtype::Float32,
                mask,
                AttentionPlan::UNBOUNDED.with_cancel(&cancel)
            )
            .is_ok());
        }
    }
}

#[test]
fn tiny_full_backbone_bounded_matches_unbounded() {
    crate::common::cpu_process(
        "bounded_attention::tiny_full_backbone_bounded_matches_unbounded",
        full_backbone,
    );
}

fn full_backbone() {
    let config = crate::common::tiny_config();
    let golden = crate::common::fixture("iris_dit_golden.safetensors");
    let flat = crate::common::host_i32(golden.require("y_mask").unwrap());
    let mask: Vec<Vec<i32>> = flat.chunks(flat.len() / 2).map(|m| m.to_vec()).collect();
    let text = mlx_gen_iris::TextBatch {
        states: golden.require("y").unwrap(),
        mask: &mask,
    };
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let dit =
            mlx_gen_iris::load_backbone(&crate::common::tiny_backbone(), &config, dtype).unwrap();
        let x = golden.require("x").unwrap();
        let t = golden.require("t").unwrap();
        let full = dit.forward(x, t, &text).unwrap();
        let bounded = dit
            .forward_budgeted(x, t, &text, AttentionBudget::from_score_elements(32, true))
            .unwrap();
        crate::common::assert_close(
            "bounded full Iris",
            &bounded,
            &full,
            if dtype == Dtype::Float32 { 2e-4 } else { 0.03 },
        );
    }
}

#[cfg(debug_assertions)]
#[test]
fn production_solver_observes_every_site_and_cancels_inside_the_forward() {
    crate::common::cpu_process(
        "bounded_attention::production_solver_observes_every_site_and_cancels_inside_the_forward",
        production_solver,
    );
}

#[cfg(debug_assertions)]
fn production_solver() {
    use mlx_gen::gen_core::iris::GenerationParams;
    use mlx_gen_iris::nn::diagnostics;
    use mlx_gen_iris::{
        encode,
        pipeline::{denoise_budgeted, noise_batch},
        IrisTextEncoder,
    };
    let config = crate::common::tiny_config();
    let dit = mlx_gen_iris::load_backbone(&crate::common::tiny_backbone(), &config, Dtype::Float32)
        .unwrap();
    let te =
        IrisTextEncoder::load(&crate::common::tiny_text_encoder(), &config.text_encoder).unwrap();
    let sites = ["adapter", "refiner", "dual", "single", "pixel"];
    for guidance in [1.0, 3.0] {
        let req = mlx_gen::GenerationRequest {
            prompt: "a fox".into(),
            width: 32,
            height: 16,
            steps: Some(1),
            guidance: Some(guidance),
            ..Default::default()
        };
        let params = GenerationParams::resolve(&req, 0, &config).unwrap();
        let conditioning = encode(&te, &params).unwrap();
        let noise = noise_batch(&[1], config.model.in_channels, 32, 16).unwrap();
        let budget = AttentionBudget::from_score_elements(1, true);
        diagnostics::reset();
        denoise_budgeted(
            &dit,
            &config.flow,
            &conditioning,
            &noise,
            &params,
            &mlx_gen::CancelFlag::new(),
            |_| {},
            &mlx_gen::PreviewSink::default(),
            budget,
        )
        .unwrap();
        let observations = diagnostics::observations();
        for site in sites {
            let rows: Vec<_> = observations.iter().filter(|r| r.site == site).collect();
            assert!(!rows.is_empty(), "CFG {guidance}: missing {site}");
            for row in rows {
                assert_eq!(
                    row.budget.max_score_elements(),
                    1,
                    "CFG {guidance}: {site} dropped plan"
                );
                assert!(row.budget.eval_per_chunk());
                assert!(
                    row.chunks > 1,
                    "CFG {guidance}: {site} never actually chunked: {row:?}"
                );
            }
        }
        // One solver evaluation only: if a site ignores cancellation there is no later solver
        // iteration whose check could hide that defect. Cancel immediately before each site.
        for site in sites {
            let cancel = mlx_gen::CancelFlag::new();
            diagnostics::reset();
            diagnostics::cancel_at(site, cancel.clone());
            let result = denoise_budgeted(
                &dit,
                &config.flow,
                &conditioning,
                &noise,
                &params,
                &cancel,
                |_| {},
                &mlx_gen::PreviewSink::default(),
                budget,
            );
            assert!(
                matches!(result, Err(mlx_gen::Error::Canceled)),
                "CFG {guidance}: {site} failed to cancel inside the forward"
            );
            assert!(
                !diagnostics::observations().iter().any(|r| r.site == site),
                "{site} completed despite its cancelled plan"
            );
        }
    }
    diagnostics::reset();
}

#[cfg(debug_assertions)]
#[test]
fn generator_request_budget_reaches_cfg_and_non_cfg_solver_sites() {
    crate::common::cpu_process(
        "bounded_attention::generator_request_budget_reaches_cfg_and_non_cfg_solver_sites",
        generator_request,
    );
}

#[cfg(debug_assertions)]
fn generator_request() {
    use mlx_gen::gen_core::{iris::TEXT_ENCODER_COMPONENT, GenerationMemory};
    use mlx_gen_iris::nn::diagnostics;
    let mut spec =
        mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(crate::common::tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        mlx_gen::WeightsSource::Dir(crate::common::tiny_text_encoder()),
    );
    let generator = mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .unwrap();
    for guidance in [1.0, 3.0] {
        let req = mlx_gen::GenerationRequest {
            prompt: "a fox".into(),
            width: 32,
            height: 16,
            steps: Some(1),
            guidance: Some(guidance),
            memory: Some(GenerationMemory {
                stage_residency: true,
                chunk_attention: true,
                attention_chunk_size: Some(mlx_gen_iris::memory_strategy::ATTENTION_CHUNK_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        diagnostics::reset();
        generator.generate(&req, &mut |_| {}).unwrap();
        let observations = diagnostics::observations();
        for site in ["adapter", "refiner", "dual", "single", "pixel"] {
            let rows: Vec<_> = observations.iter().filter(|r| r.site == site).collect();
            assert!(!rows.is_empty());
            for row in rows {
                assert_eq!(
                    row.budget.max_score_elements(),
                    u64::from(mlx_gen_iris::memory_strategy::ATTENTION_CHUNK_SIZE),
                    "request budget lost at CFG {guidance}/{site}"
                );
                assert!(row.budget.eval_per_chunk());
            }
        }
    }
    diagnostics::reset();
}

/// E2: bounded attention being SELECTABLE on the Mac changes nothing by default. A request with no
/// memory block, or an all-off one, runs every attention site unbounded in one call (never
/// chunked), and its image is bit-identical to the legacy unbounded pipeline
/// (`pipeline::denoise`, which predates the request-selected plan).
///
/// *Mutation that reds this:* making `attention_budget` return any budget but
/// `AttentionBudget::UNBOUNDED` for a request without `chunk_attention`.
#[cfg(debug_assertions)]
#[test]
fn the_default_mac_request_runs_unbounded_and_bit_identical_to_the_legacy_path() {
    crate::common::cpu_process(
        "bounded_attention::the_default_mac_request_runs_unbounded_and_bit_identical_to_the_legacy_path",
        default_request,
    );
}

#[cfg(debug_assertions)]
fn default_request() {
    use mlx_gen::gen_core::iris::{GenerationParams, TEXT_ENCODER_COMPONENT};
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen_iris::nn::diagnostics;
    use mlx_gen_iris::{encode, pipeline, IrisTextEncoder};
    let config = crate::common::tiny_config();
    let mut spec =
        mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(crate::common::tiny_backbone()));
    spec.precision = mlx_gen::Precision::Fp32;
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        mlx_gen::WeightsSource::Dir(crate::common::tiny_text_encoder()),
    );
    let generator = mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .unwrap();
    let pixels = |output: mlx_gen::GenerationOutput| match output {
        mlx_gen::GenerationOutput::Images(images) => {
            images.into_iter().map(|i| i.pixels).collect::<Vec<_>>()
        }
        _ => panic!("expected images"),
    };
    for guidance in [1.0, 3.0] {
        let base = mlx_gen::GenerationRequest {
            prompt: "a fox".into(),
            width: 32,
            height: 16,
            steps: Some(2),
            seed: Some(7),
            guidance: Some(guidance),
            ..Default::default()
        };
        let mut images = Vec::new();
        for memory in [None, Some(GenerationMemory::default())] {
            let req = mlx_gen::GenerationRequest {
                memory,
                ..base.clone()
            };
            diagnostics::reset();
            images.push(pixels(generator.generate(&req, &mut |_| {}).unwrap()));
            let observed = diagnostics::observations();
            assert!(!observed.is_empty());
            for row in observed {
                assert_eq!(
                    row.budget,
                    AttentionBudget::UNBOUNDED,
                    "CFG {guidance}: {} ran under a budget by default",
                    row.site
                );
                assert_eq!(
                    row.chunks, 1,
                    "CFG {guidance}: {} chunked by default",
                    row.site
                );
            }
        }
        assert_eq!(images[0], images[1]);
        // The legacy unbounded pipeline, driven directly.
        let dit =
            mlx_gen_iris::load_backbone(&crate::common::tiny_backbone(), &config, Dtype::Float32)
                .unwrap();
        let te = IrisTextEncoder::load(&crate::common::tiny_text_encoder(), &config.text_encoder)
            .unwrap();
        let params = GenerationParams::resolve(&base, 0, &config).unwrap();
        let conditioning = encode(&te, &params).unwrap();
        let noise =
            pipeline::noise_batch(&[params.seed], config.model.in_channels, 32, 16).unwrap();
        let x = pipeline::denoise(
            &dit,
            &config.flow,
            &conditioning,
            &noise,
            &params,
            &mlx_gen::CancelFlag::new(),
            |_| {},
            &mlx_gen::PreviewSink::default(),
        )
        .unwrap();
        let legacy: Vec<Vec<u8>> = pipeline::to_images(&x)
            .unwrap()
            .into_iter()
            .map(|i| i.pixels)
            .collect();
        assert_eq!(
            images[0], legacy,
            "CFG {guidance}: the default request left the legacy path"
        );
    }
    diagnostics::reset();
}
