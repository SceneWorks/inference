//! Terminal-only native smoke. Never executed by ordinary CI or the implementation story.
use candle_gen::{
    candle_core::Device,
    gen_core::{self, *},
};
use candle_gen_moss_image::MossImageGenerator;
use gen_core_moss_image::moss_image::*;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

fn required(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("set {name}; no cache discovery or fetching is performed")),
    )
}
fn prepared(variant: MossImageVariant, base: &Path, shared: &Path, dmd: &Path) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(base.into()));
    spec.components.insert(
        MOSS_SHARED_SNAPSHOT_COMPONENT.into(),
        WeightsSource::Dir(shared.into()),
    );
    if variant == MossImageVariant::Sft {
        for (id, name) in [
            (DMD_LORA_COMPONENT, "lora.safetensors"),
            (DMD_TIME_COMPONENT, "time_embedding.safetensors"),
            (DMD_DATASPACE_COMPONENT, "dataspace.safetensors"),
        ] {
            spec.components
                .insert(id.into(), WeightsSource::File(dmd.join(name)));
        }
        spec.components.insert(
            DMD_TRANSFORMER_COMPONENT.into(),
            WeightsSource::Dir(dmd.into()),
        );
    }
    let mut paths = BTreeSet::new();
    for relative in SNAPSHOT_FILES.iter().filter(|p| **p != "LICENSE") {
        paths.insert(if relative.starts_with("transformer/") {
            base.join(relative)
        } else {
            shared.join(relative)
        });
    }
    if variant == MossImageVariant::Sft {
        for name in [
            "lora.safetensors",
            "time_embedding.safetensors",
            "dataspace.safetensors",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00004.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00004.safetensors",
            "transformer/diffusion_pytorch_model-00004-of-00004.safetensors",
        ] {
            paths.insert(dmd.join(name));
        }
    }
    spec.prepare_with_file_pins(
        paths
            .into_iter()
            .map(|p| PinnedWeightsFile::pin(p).unwrap()),
    )
    .unwrap();
    spec
}
fn record(output: &Path, value: serde_json::Value) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("backend-cases.jsonl"))
        .unwrap();
    writeln!(file, "{value}").unwrap();
}
fn save(output: &Path, name: &str, images: &[Image], report: &MossImageExecutionReport) {
    let skip_png = std::env::var("MOSS_NATIVE_SKIP_PNG").as_deref() == Ok("1");
    for (i, image) in images.iter().enumerate().filter(|_| !skip_png) {
        image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
            .unwrap()
            .save(output.join(format!("{name}-{i}.png")))
            .unwrap();
    }
    std::fs::write(
        output.join(format!("{name}.receipt.json")),
        serde_json::to_vec_pretty(report).unwrap(),
    )
    .unwrap();
    record(
        output,
        serde_json::json!({"case":name,"status":"completed","receipt":format!("{name}.receipt.json"),"pngs_skipped":skip_png,"images":images.len()}),
    );
}
fn images(output: GenerationOutput) -> Vec<Image> {
    match output {
        GenerationOutput::Images(i) => i,
        other => panic!("unexpected native output {other:?}"),
    }
}

#[test]
#[ignore = "requires root-owned fresh physical GPU0 lease, exact staged assets and terminal sc-26325 admission"]
fn native_complete_surface_campaign() {
    assert_eq!(
        std::env::var("MOSS_NATIVE_TERMINAL_ADMITTED").as_deref(),
        Ok("sc-26325-physical-gpu0")
    );
    assert_eq!(std::env::var("MOSS_PHYSICAL_GPU_INDEX").as_deref(), Ok("0"));
    let expected =
        std::env::var("MOSS_EXPECTED_CUDA_UUID").expect("fresh allocator's physical GPU0 UUID");
    assert!(!expected.is_empty());
    assert_eq!(
        std::env::var("CUDA_VISIBLE_DEVICES").unwrap(),
        expected,
        "allocator must expose only the admitted physical GPU0 UUID"
    );
    let base = required("MOSS_BASE_SNAPSHOT");
    let shared = required("MOSS_SHARED_SNAPSHOT");
    let dmd = required("MOSS_DMD_SNAPSHOT");
    let output = required("MOSS_NATIVE_OUTPUT_DIR");
    std::fs::create_dir_all(&output).unwrap();
    let request = GenerationRequest {
        prompt: "A red ceramic teapot beside a blue cup on a wooden table".into(),
        width: 256,
        height: 256,
        steps: Some(2),
        seed: Some(26324),
        ..Default::default()
    };
    for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
        let spec = prepared(
            variant,
            if variant == MossImageVariant::Base {
                &base
            } else {
                &shared
            },
            &shared,
            &dmd,
        );
        let generator =
            MossImageGenerator::load_on_device(variant, &spec, Device::new_cuda(0).unwrap())
                .unwrap();
        let ordinary = images(
            generator
                .generate(&request, &mut |p| eprintln!("{variant:?} {p:?}"))
                .unwrap(),
        );
        let report = generator
            .moss_image_execution_report()
            .expect("actual complete native receipt");
        assert_eq!(
            report.data().execution_device,
            MossImageExecutionDevice::Cuda
        );
        assert_eq!(
            report.data().resolved.acceleration,
            MossImageAcceleration::Off
        );
        save(
            &output,
            &format!("{variant:?}-ordinary"),
            &ordinary,
            &report,
        );
        if variant == MossImageVariant::Sft {
            for precision in [MossDmdPrecision::UnmergedBf16, MossDmdPrecision::FullFp32] {
                let mut req = request.clone();
                req.steps = Some(8);
                req.moss_image = Some(MossImageOptions {
                    acceleration: MossImageAcceleration::Dmd(precision),
                    ..Default::default()
                });
                let accelerated = images(
                    generator
                        .generate(&req, &mut |p| eprintln!("{precision:?} {p:?}"))
                        .unwrap(),
                );
                let report = generator.moss_image_execution_report().unwrap();
                assert_eq!(
                    report.data().resolved.acceleration,
                    MossImageAcceleration::Dmd(precision)
                );
                assert_eq!(
                    report.data().adapter_reports.len(),
                    usize::from(precision == MossDmdPrecision::UnmergedBf16)
                );
                save(
                    &output,
                    &format!("Sft-{precision:?}"),
                    &accelerated,
                    &report,
                );
                let off = images(generator.generate(&request, &mut |_| {}).unwrap());
                assert_eq!(
                    ordinary, off,
                    "OFF must recover exact ordinary weights after {precision:?}"
                );
                save(
                    &output,
                    &format!("Sft-off-after-{precision:?}"),
                    &off,
                    &generator.moss_image_execution_report().unwrap(),
                );
            }
            // The native advanced route consumes real provided embeddings, masks and exact FP32
            // noise; output stays latent so this adds no image-quality/adoption comparison.
            let embed: Vec<f32> = (0..2 * 3 * 4096)
                .map(|i| (i as f32 * 0.001).sin() * 0.1)
                .collect();
            let mask = [1, 0, 1, 1, 1, 0];
            let negative = vec![0.0; 2 * 2 * 4096];
            let noise: Vec<f32> = (0..4 * 128 * 16 * 16)
                .map(|i| (i as f32 * 0.017).sin())
                .collect();
            let mut req = request.clone();
            req.prompt.clear();
            req.count = 2;
            req.guidance = Some(1.0);
            req.moss_image = Some(MossImageOptions {
                uncond_skip_layers: Some(vec![]),
                slg: Some(MossSkipLayerGuidance {
                    layers: vec![24],
                    scale: 1.5,
                    start: 0.0,
                    stop: 1.0,
                }),
                ..Default::default()
            });
            let advanced = MossImageAdvancedInputs {
                positive: Some(MossImageEmbeddings {
                    values: &embed,
                    batch: 2,
                    sequence: 3,
                    mask: Some(&mask),
                }),
                negative: Some(MossImageEmbeddings {
                    values: &negative,
                    batch: 2,
                    sequence: 2,
                    mask: None,
                }),
                negative_prompts: None,
                noise: Some(&noise),
                output: MossImageOutputKind::Latents,
            };
            let output_latents = generator
                .generate_moss_advanced(&req, advanced, &mut |_| {})
                .unwrap();
            match output_latents {
                MossImageAdvancedOutput::Latents { values, shape } => {
                    assert_eq!(shape, [4, 128, 16, 16]);
                    assert!(values.iter().all(|v| v.is_finite()));
                }
                other => panic!("unexpected advanced output {other:?}"),
            };
            let report = generator.moss_image_execution_report().unwrap();
            assert!(report.data().samples.iter().all(|s| s.seed.is_none()));
            std::fs::write(
                output.join("Sft-advanced.receipt.json"),
                serde_json::to_vec_pretty(&report).unwrap(),
            )
            .unwrap();
            record(
                &output,
                serde_json::json!({"case":"Sft-advanced","status":"completed","receipt":"Sft-advanced.receipt.json","images":0,"pngs_skipped":true}),
            );
        }
        let mut canceled = request.clone();
        // Each variant must retain its own cancellation attempt; cloned flags share state.
        canceled.cancel = CancelFlag::default();
        let result = generator.generate(&canceled, &mut |p| {
            if matches!(p, Progress::Step { current: 1, .. }) {
                canceled.cancel.cancel()
            }
        });
        assert!(matches!(result, Err(gen_core::Error::Canceled)));
        assert!(generator.moss_image_execution_report().is_none());
        record(
            &output,
            serde_json::json!({"case":format!("{variant:?}-cancel"),"status":"canceled","typed_cancelled":true,"receipt_cleared":true}),
        );
        let mut recovered = request.clone();
        recovered.cancel = CancelFlag::default();
        let recovered = images(generator.generate(&recovered, &mut |_| {}).unwrap());
        assert_eq!(ordinary, recovered);
        save(
            &output,
            &format!("{variant:?}-warm-after-cancel"),
            &recovered,
            &generator.moss_image_execution_report().unwrap(),
        );
    }
}

#[cfg(feature = "model-inference-proof")]
fn bounded_root() -> std::path::PathBuf {
    let root = std::path::PathBuf::from(
        std::env::var_os("MODEL_PROOF_OUTPUT_DIR").expect("manual executor output directory"),
    )
    .join("bespoke_acceptance");
    std::fs::create_dir_all(&root).unwrap();
    root
}
#[cfg(feature = "model-inference-proof")]
fn bounded_peak(sampler: candle_gen::testkit::PeakSampler, root: &std::path::Path) {
    let peak = sampler.stop();
    assert!(
        peak > 0,
        "selected physical CUDA device peak must be observed"
    );
    std::fs::write(
        root.join("peak_device_bytes.txt"),
        (peak * 1024 * 1024).to_string(),
    )
    .unwrap();
}

#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn moss_bounded_acceleration_acceptance() {
    let root = bounded_root();
    let comparison = candle_gen::train::comparison::requested_stage().is_some();
    let cadence = candle_gen::train::comparison::ComparisonCadence::Ordinary;
    let steps = if comparison { cadence.updates() } else { 4 };
    // One source-owned representative workload, never a resolution grid. The
    // controller must admit the worst legacy arm before executing any stage.
    let resolution = if comparison { 512 } else { 64 };
    let base = required("MOSS_BASE_SNAPSHOT");
    let shared = required("MOSS_SHARED_SNAPSHOT");
    let input = root.join("input.png");
    image::RgbImage::from_pixel(
        resolution as u32,
        resolution as u32,
        image::Rgb([80, 140, 210]),
    )
    .save(&input)
    .unwrap();
    let config = TrainingConfig {
        rank: 2,
        alpha: 2.,
        learning_rate: 1e-3,
        steps,
        resolution,
        batch_size: 1,
        gradient_accumulation: 1,
        lr_warmup_steps: 0,
        save_every: 0,
        seed: 7,
        train_dtype: "bf16".into(),
        gradient_checkpointing: false,
        ..Default::default()
    };
    let items = vec![TrainingItem::captioned(input, "a blue tile".into())];
    let sampler = (!comparison).then(candle_gen::testkit::PeakSampler::start_rendered);
    // A new child owns a fresh native trainer/Run, initial parameters, native
    // optimizer, EMA/data cursors and (restoration) critic/SN state. No quality
    // render or adapter reload is repeated by this timing-only supplement.
    let run = || {
        let spec = prepared(MossImageVariant::Base, &base, &shared, &shared);
        if comparison {
            comparison_initialization(&config, &items, &spec);
        }
        let mut trainer = candle_gen_moss_image::register_providers(
            candle_gen::gen_core::ProviderRegistryBuilder::new(),
        )
        .build()
        .unwrap()
        .load_trainer(MossImageVariant::Base.provider_id(), &spec)
        .unwrap();
        let output = trainer
            .train(
                &TrainingRequest {
                    items,
                    config,
                    output_dir: root.join("out"),
                    file_name: "adapter.safetensors".into(),
                    trigger_words: vec![],
                    cancel: CancelFlag::new(),
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(output.steps, steps);
        assert!(output.final_loss.is_finite());
        (output, spec)
    };
    let (output, spec) = if comparison {
        candle_gen::train::comparison::with_stage(
            &root.join("comparison_metrics.json"),
            "moss",
            cadence,
            run,
        )
    } else {
        candle_gen::train::profile::with_bounded_cohort_for(
            &root.join("metrics.json"),
            &[
                "moss:geometry-storage-hit",
                "moss:frozen-caption-storage-hit",
                "moss:frozen-projection-pack",
            ],
            run,
        )
    };
    if comparison {
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::moss_bounded_acceleration_acceptance"
        );
        return;
    }
    bounded_peak(sampler.expect("ordinary fixture sampler"), &root);
    let tensors =
        candle_gen::candle_core::safetensors::load(&output.adapter_path, &Device::Cpu).unwrap();
    assert!(!tensors.is_empty());
    drop(tensors);
    let generator = candle_gen_moss_image::register_providers(
        candle_gen::gen_core::ProviderRegistryBuilder::new(),
    )
    .build()
    .unwrap()
    .load(
        MossImageVariant::Base.provider_id(),
        &spec.with_adapters(vec![AdapterSpec::new(
            output.adapter_path,
            1.,
            AdapterKind::Lora,
        )]),
    )
    .unwrap();
    let rendered = images(
        generator
            .generate(
                &GenerationRequest {
                    prompt: "a blue tile".into(),
                    width: 256,
                    height: 256,
                    steps: Some(2),
                    seed: Some(7),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap(),
    );
    assert_eq!(rendered.len(), 1);
    assert_eq!(rendered[0].pixels.len(), 256 * 256 * 3);
    assert!(rendered[0].pixels.iter().any(|p| *p != 0));
    candle_gen::testkit::save_image_png(&root.join("adapter_sample.png"), &rendered[0]);
    println!("\nMODEL_INFERENCE_BODY_COMPLETED real_weights::moss_bounded_acceleration_acceptance");
}

#[cfg(feature = "model-inference-proof")]
fn comparison_input_sha256(path: &std::path::Path) -> String {
    use sha2::Digest;
    format!(
        "{:x}",
        sha2::Sha256::digest(std::fs::read(path).expect("comparison input"))
    )
}

#[cfg(feature = "model-inference-proof")]
fn comparison_initialization(
    config: &candle_gen::gen_core::TrainingConfig,
    items: &[candle_gen::gen_core::TrainingItem],
    spec: &candle_gen::gen_core::LoadSpec,
) {
    let data = items.iter().map(|item| {
        let sidecars = item.model_options.iter().map(|(key,value)| {
            let value = if let Some(path) = value.as_str().filter(|path| std::path::Path::new(path).is_file()) {
                serde_json::json!({"file_sha256":comparison_input_sha256(std::path::Path::new(path))})
            } else { value.clone() };
            (key.clone(),value)
        }).collect::<serde_json::Map<String,serde_json::Value>>();
        serde_json::json!({"image_sha256":comparison_input_sha256(&item.image_path),
            "caption":item.caption,"sidecars":sidecars,
            "references":item.reference_image_paths.iter().map(|path|comparison_input_sha256(path)).collect::<Vec<_>>(),
            "control":item.control_image_path.as_deref().map(comparison_input_sha256),
            "subject_mask":item.subject_mask_path.as_deref().map(comparison_input_sha256)})
    }).collect::<Vec<_>>();
    candle_gen::train::comparison::record_initialization(&serde_json::json!({
        "training_config":format!("{config:?}"),"source_spec":format!("{spec:?}"),
        "seed":config.seed,"resolution":config.resolution,"batch_size":config.batch_size,
        "gradient_accumulation":config.gradient_accumulation,"data":data,
        "factory":"fresh native trainer in one comparison child; no resumed mutable run",
        "claim_limit":"content/configuration witnesses; not a claim of whole checkpoint bytes or tensor equality from source flags"
    }));
}
