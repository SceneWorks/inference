//! Terminal-only native smoke. Never executed by ordinary CI or the implementation story.
use candle_gen::{
    candle_core::Device,
    gen_core::{self, moss_image::*, *},
};
use candle_gen_moss_image::MossImageGenerator;
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
