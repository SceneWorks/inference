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

/// E2 against the BASE rev, not just new-vs-new: the default (no memory block) generation path —
/// CFG off and on — and the depth and restoration task defaults, on the miniature fixtures, match
/// `iris_default_base_91ffb4a81.safetensors`, produced by this same [`default_path_cases`] run on
/// `mlx-gen-iris` at inference `91ffb4a81` (before bounded attention, its plans and the packed
/// tiers existed) in a CPU child, FP32.
///
/// Tolerance: the runs are MLX CPU-stream FP32, whose GEMMs go through the host BLAS — the same
/// binary on another Apple CPU generation may pick a different kernel and so a different summation
/// order (~1e-5 relative, the `dit_parity` FP32 distance). That moves an RGB8 output by at most one
/// level where a value sits on a rounding boundary (the restoration parity rule), so images allow
/// `max |Δ| <= 1` level on at most 1% of their bytes, and the f32 depth map 1e-4 of peak. Exact
/// equality would be a property of one machine, not of the code. A changed default path (seed,
/// guidance, steps, schedule, a dropped or extra op) moves many bytes by many levels.
///
/// *Mutations that red this:* any change to what the default request computes — measured: the
/// generation noise seed offset by one (255 levels), the depth output scaled by 1.001 (9e-4 of
/// peak), the restoration velocity scaled by 1.01 (one level on 20% of the bytes). A default that
/// merely chunks attention stays numerically the unbounded image and is not this test's to catch:
/// `the_default_mac_request_runs_unbounded_and_bit_identical_to_the_legacy_path` asserts the plan.
#[test]
fn the_default_paths_match_the_base_rev_golden() {
    crate::common::cpu_process(
        "bounded_attention::the_default_paths_match_the_base_rev_golden",
        default_paths_golden,
    );
}

fn default_paths_golden() {
    let golden = crate::common::fixture("iris_default_base_91ffb4a81.safetensors");
    let cases = default_path_cases();
    let names: Vec<_> = cases.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "default_g1",
            "default_g3",
            "depth_default",
            "restore_default"
        ]
    );
    for (name, shape, got) in cases {
        let want = golden.require(&name).unwrap();
        assert_eq!(want.shape(), shape.as_slice(), "{name}: shape");
        if name == "depth_default" {
            let got = Array::from_slice(&got, &shape);
            crate::common::assert_close(&name, &got, want, 1e-4);
            continue;
        }
        let want = crate::common::host_f32(want);
        assert!(want.iter().any(|v| *v != want[0]), "{name}: flat golden");
        let levels: Vec<f32> = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).collect();
        let max = levels.iter().fold(0f32, |m, d| m.max(*d));
        let off = levels.iter().filter(|d| **d > 0.0).count();
        eprintln!("{name}: RGB8 max|Δ|={max} differing={off}/{}", want.len());
        assert!(
            max <= 1.0,
            "{name}: RGB8 differs from the base rev by {max} levels"
        );
        assert!(
            off * 100 <= want.len(),
            "{name}: {off}/{} bytes differ from the base rev",
            want.len()
        );
    }
}

/// The default production paths on the miniature fixtures, as `(case, shape, values)`. Every case
/// is one fixed request with no memory block: generation (CFG off and on), depth and restoration.
fn default_path_cases() -> Vec<(String, Vec<i32>, Vec<f32>)> {
    use mlx_gen::gen_core::iris::depth::{DepthRequest, DepthResolution};
    use mlx_gen::gen_core::iris::TEXT_ENCODER_COMPONENT;
    use mlx_gen::{Image, LoadSpec, Precision, TargetSize, TransformRequest, WeightsSource};
    let fixtures = crate::common::fixtures();
    let rgb = |width: u32, height: u32| {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height as u64 {
            for x in 0..width as u64 {
                for c in 0..3u64 {
                    pixels.push(((x * 7 + y * 13 + c * 101 + (x * y) % 97) % 256) as u8);
                }
            }
        }
        Image {
            width,
            height,
            pixels,
        }
    };
    let fp32 = |dir: std::path::PathBuf| {
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir));
        spec.precision = Precision::Fp32;
        spec
    };
    let bytes = |image: &Image| {
        image
            .pixels
            .iter()
            .map(|&p| f32::from(p))
            .collect::<Vec<_>>()
    };
    let mut cases = Vec::new();

    let mut spec = fp32(crate::common::tiny_backbone());
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(crate::common::tiny_text_encoder()),
    );
    let generator = mlx_gen_iris::provider_registry()
        .unwrap()
        .load("iris_3b", &spec)
        .unwrap();
    for (name, guidance) in [("default_g1", 1.0), ("default_g3", 3.0)] {
        let req = mlx_gen::GenerationRequest {
            prompt: "a fox".into(),
            width: 32,
            height: 16,
            steps: Some(2),
            seed: Some(7),
            guidance: Some(guidance),
            ..Default::default()
        };
        let mlx_gen::GenerationOutput::Images(images) =
            generator.generate(&req, &mut |_| {}).unwrap()
        else {
            panic!("expected images");
        };
        assert_eq!(images.len(), 1);
        let image = &images[0];
        cases.push((
            name.to_owned(),
            vec![image.height as i32, image.width as i32, 3],
            bytes(image),
        ));
    }

    let depth = mlx_gen_iris::depth::load(&fp32(fixtures.join("tiny-depth"))).unwrap();
    let out = depth
        .estimate(
            &DepthRequest {
                image: rgb(20, 14),
                resolution: DepthResolution::Native,
                cancel: Default::default(),
            },
            &mut |_| {},
        )
        .unwrap();
    cases.push((
        "depth_default".to_owned(),
        vec![out.map.height as i32, out.map.width as i32],
        out.map.values,
    ));

    let restorer =
        mlx_gen_iris::restoration::load(&fp32(fixtures.join("tiny-snapshot/upscaler"))).unwrap();
    let image = restorer
        .apply(
            &TransformRequest {
                image: rgb(24, 16),
                target: TargetSize::Scale(2.0),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
    cases.push((
        "restore_default".to_owned(),
        vec![image.height as i32, image.width as i32, 3],
        bytes(&image),
    ));
    cases
}
