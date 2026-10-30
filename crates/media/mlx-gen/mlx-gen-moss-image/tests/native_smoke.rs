//! Explicitly admitted terminal acceptance only; never acquires or discovers model assets.
use gen_core_moss_image::moss_image::*;
use mlx_gen::{GenerationRequest, Generator, LoadSpec, Progress, WeightsSource};
use mlx_gen_moss_image::MossImage;
use std::path::PathBuf;
fn path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("supply explicit {name}"))
}
#[test]
#[ignore = "requires coordinator-admitted Metal ownership, exact local model assets and explicit output directory"]
fn native_generation_or_cancel_with_actual_receipt() {
    assert_eq!(
        std::env::var("MOSS_NATIVE_TEST_ADMITTED").as_deref(),
        Ok("1"),
        "coordinator must admit accelerator ownership first"
    );
    let variant = match std::env::var("MOSS_NATIVE_TEST_VARIANT").as_deref() {
        Ok("base") => MossImageVariant::Base,
        Ok("sft") => MossImageVariant::Sft,
        _ => panic!("set MOSS_NATIVE_TEST_VARIANT=base or sft"),
    };
    let mode = match std::env::var("MOSS_NATIVE_TEST_MODE").as_deref() {
        Ok("off") => MossImageAcceleration::Off,
        Ok("unmerged_bf16") => MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16),
        Ok("full_fp32") => MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32),
        _ => panic!("explicit MOSS_NATIVE_TEST_MODE required"),
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(path("MOSS_NATIVE_TEST_SNAPSHOT")));
    if variant == MossImageVariant::Base {
        spec.components.insert(
            MOSS_SHARED_SNAPSHOT_COMPONENT.into(),
            WeightsSource::Dir(path("MOSS_NATIVE_TEST_SHARED_SNAPSHOT")),
        );
    }
    if mode != MossImageAcceleration::Off {
        let root = path("MOSS_NATIVE_TEST_DMD_SNAPSHOT");
        spec.components.insert(
            DMD_DATASPACE_COMPONENT.into(),
            WeightsSource::File(root.join("dataspace.safetensors")),
        );
        match mode {
            MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) => {
                spec.components.insert(
                    DMD_LORA_COMPONENT.into(),
                    WeightsSource::File(root.join("lora.safetensors")),
                );
                spec.components.insert(
                    DMD_TIME_COMPONENT.into(),
                    WeightsSource::File(root.join("time_embedding.safetensors")),
                );
            }
            MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) => {
                spec.components
                    .insert(DMD_TRANSFORMER_COMPONENT.into(), WeightsSource::Dir(root));
            }
            _ => {}
        }
    }
    // This test uses compatibility pin acquisition, then authenticates all exact contents in the
    // same production provider. Product prepared-token acceptance is tested independently.
    let model = MossImage::load(variant, &spec).unwrap();
    let mut request=GenerationRequest{prompt:std::env::var("MOSS_NATIVE_TEST_PROMPT").unwrap_or_else(|_|"A red ceramic mug on a wooden table, soft window light, detailed studio photograph.".into()),width:std::env::var("MOSS_NATIVE_TEST_WIDTH").map(|v|v.parse().unwrap()).unwrap_or(512),height:std::env::var("MOSS_NATIVE_TEST_HEIGHT").map(|v|v.parse().unwrap()).unwrap_or(512),seed:Some(173),moss_image:Some(MossImageOptions{acceleration:mode,..Default::default()}),..Default::default()};
    if let Ok(steps) = std::env::var("MOSS_NATIVE_TEST_STEPS") {
        request.steps = Some(steps.parse().unwrap());
    }
    let out = path("MOSS_NATIVE_TEST_OUTPUT_DIR");
    std::fs::create_dir_all(&out).unwrap();
    if std::env::var("MOSS_NATIVE_TEST_TRANSITIONS").as_deref() == Ok("1") {
        assert_eq!(variant, MossImageVariant::Sft);
        assert_ne!(
            mode,
            MossImageAcceleration::Off,
            "select the accelerated policy for OFF→ON→OFF"
        );
        let mut original = None;
        for (index, policy) in [MossImageAcceleration::Off, mode, MossImageAcceleration::Off]
            .into_iter()
            .enumerate()
        {
            request.moss_image.as_mut().unwrap().acceleration = policy;
            let result = model
                .generate(&request, &mut |event| {
                    eprintln!("transition {index}: {event:?}")
                })
                .unwrap();
            let mlx_gen::GenerationOutput::Images(images) = result else {
                panic!("expected native images")
            };
            let image = &images[0];
            if index == 0 {
                original = Some(image.pixels.clone());
            }
            if index == 2 {
                assert_eq!(
                    original.as_ref().unwrap(),
                    &image.pixels,
                    "ordinary pixels changed after accelerated mode"
                );
            }
            image::save_buffer(
                out.join(format!("transition-{index}.png")),
                &image.pixels,
                image.width,
                image.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
            let receipt = model.moss_image_execution_report().unwrap();
            assert_eq!(receipt.data().resolved.acceleration, policy);
            std::fs::write(
                out.join(format!("transition-{index}.json")),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
        }
        return;
    }
    let cancel_at = std::env::var("MOSS_NATIVE_TEST_CANCEL_AFTER_STEP")
        .ok()
        .map(|v| v.parse::<u32>().unwrap());
    let result = model.generate(&request, &mut |event| {
        eprintln!("MOSS progress {event:?}");
        if let (Some(limit), Progress::Step { current, .. }) = (cancel_at, event) {
            if current >= limit {
                request.cancel.cancel();
            }
        }
    });
    if cancel_at.is_some() {
        assert!(matches!(result, Err(mlx_gen::gen_core::Error::Canceled)));
        assert!(model.moss_image_execution_report().is_none());
        std::fs::write(
            out.join("cancelled.json"),
            "{\"status\":\"typed_cancel_no_outputs_no_receipt\"}\n",
        )
        .unwrap();
        if std::env::var("MOSS_NATIVE_TEST_WARM_REENTRY").as_deref() == Ok("1") {
            request.cancel = mlx_gen::CancelFlag::new();
            request.steps = Some(1);
            let output = model
                .generate_moss_advanced(
                    &request,
                    MossImageAdvancedInputs {
                        output: MossImageOutputKind::Latents,
                        ..Default::default()
                    },
                    &mut |_| {},
                )
                .unwrap();
            assert!(matches!(output, MossImageAdvancedOutput::Latents { .. }));
            let receipt = model.moss_image_execution_report().unwrap();
            assert_eq!(receipt.data().output_kind, MossImageOutputKind::Latents);
            std::fs::write(
                out.join("warm-reentry.json"),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
        }
        return;
    }
    let mlx_gen::GenerationOutput::Images(images) = result.unwrap() else {
        panic!("expected actual images")
    };
    assert_eq!(images.len(), 1);
    for (i, image) in images.into_iter().enumerate() {
        image::save_buffer(
            out.join(format!("image-{i}.png")),
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
    let receipt = model
        .moss_image_execution_report()
        .expect("native execution must publish actual receipt");
    assert_eq!(receipt.data().resolved.acceleration, mode);
    std::fs::write(
        out.join("execution.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
}

/// sc-26325: latent-only native transport acceptance, separate from image quality or adoption.
#[test]
#[ignore = "requires coordinator-admitted Metal ownership and exact local assets; no PNG comparison"]
fn native_advanced_supplied_order_receipt() {
    assert_eq!(
        std::env::var("MOSS_NATIVE_TEST_ADMITTED").as_deref(),
        Ok("1")
    );
    let variant = match std::env::var("MOSS_NATIVE_TEST_VARIANT").as_deref() {
        Ok("base") => MossImageVariant::Base,
        Ok("sft") => MossImageVariant::Sft,
        _ => panic!("set MOSS_NATIVE_TEST_VARIANT=base or sft"),
    };
    let mut spec = LoadSpec::new(WeightsSource::Dir(path("MOSS_NATIVE_TEST_SNAPSHOT")));
    if variant == MossImageVariant::Base {
        spec.components.insert(
            MOSS_SHARED_SNAPSHOT_COMPONENT.into(),
            WeightsSource::Dir(path("MOSS_NATIVE_TEST_SHARED_SNAPSHOT")),
        );
    }
    let model = MossImage::load(variant, &spec).unwrap();
    let request = GenerationRequest {
        prompt: String::new(),
        width: 256,
        height: 256,
        count: 2,
        steps: Some(1),
        guidance: Some(1.0),
        ..Default::default()
    };
    let embeddings: Vec<f32> = (0..2 * 3 * 4096)
        .map(|i| 1.001 + (i as f32 * 0.019).sin() * 0.1)
        .collect();
    let mask = [1, 0, 1, 0, 1, 1];
    let stride = 128 * 16 * 16;
    let noise: Vec<f32> = (0..4 * stride).map(|i| (i as f32 * 0.017).sin()).collect();
    let output = model
        .generate_moss_advanced(
            &request,
            MossImageAdvancedInputs {
                positive: Some(MossImageEmbeddings {
                    values: &embeddings,
                    batch: 2,
                    sequence: 3,
                    mask: Some(&mask),
                }),
                noise: Some(&noise),
                output: MossImageOutputKind::Latents,
                ..Default::default()
            },
            &mut |event| eprintln!("advanced native progress: {event:?}"),
        )
        .unwrap();
    let MossImageAdvancedOutput::Latents { values, shape } = output else {
        panic!("expected supplied native latents")
    };
    assert_eq!(shape, [4, 128, 16, 16]);
    assert_eq!(values.len(), 4 * stride);
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 0.0001));
    let report = model.moss_image_execution_report().unwrap();
    assert_eq!(
        report.data().resolved.acceleration,
        MossImageAcceleration::Off
    );
    assert_eq!(report.data().output_kind, MossImageOutputKind::Latents);
    assert_supplied_order(
        &report.data().conditioning,
        &report.data().samples,
        report.data().images_per_prompt,
        &embeddings,
        &mask,
        &noise,
        stride,
    );
    let out = path("MOSS_NATIVE_TEST_OUTPUT_DIR");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("advanced-supplied.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

fn assert_supplied_order(
    conditioning: &[MossImageConditioningReceipt],
    samples: &[MossImageSampleReceipt],
    images_per_prompt: u32,
    embeddings: &[f32],
    mask: &[u8],
    noise: &[f32],
    stride: usize,
) {
    use sha2::{Digest, Sha256};
    assert_eq!(samples.len(), 4);
    assert_eq!(conditioning.len(), 2);
    assert_eq!(images_per_prompt, 2);
    assert_eq!(noise.len(), 4 * stride);
    for i in 0..4 {
        let mut hash = Sha256::new();
        for value in &noise[i * stride..(i + 1) * stride] {
            hash.update(value.to_le_bytes());
        }
        assert_eq!(samples[i].noise_sha256, format!("{:x}", hash.finalize()));
        assert!(samples[i].seed.is_none());
        let b = i / 2;
        let mut hash = Sha256::new();
        for value in &embeddings[b * 3 * 4096..(b + 1) * 3 * 4096] {
            hash.update(value.to_le_bytes());
        }
        hash.update(&mask[b * 3..(b + 1) * 3]);
        assert_eq!(
            conditioning[b].positive.sha256,
            format!("{:x}", hash.finalize())
        );
        assert_eq!(conditioning[b].positive.valid_tokens, 2);
    }
}
#[test]
fn supplied_receipt_prompt_major_order_is_distinct_from_sample_order_sc26325() {
    use sha2::{Digest, Sha256};
    let embeddings: Vec<f32> = (0..2 * 3 * 4096).map(|i| i as f32 * 0.01).collect();
    let mask = [1, 0, 1, 0, 1, 1];
    let noise = [1., 2., 3., 4., 5., 6., 7., 8.];
    let conditioning: Vec<_> = (0..2)
        .map(|b| {
            let mut h = Sha256::new();
            for v in &embeddings[b * 3 * 4096..(b + 1) * 3 * 4096] {
                h.update(v.to_le_bytes());
            }
            h.update(&mask[b * 3..(b + 1) * 3]);
            MossImageConditioningReceipt {
                positive: MossImageConditioningInputReceipt {
                    source: MossImageConditioningSource::Embeddings,
                    sha256: format!("{:x}", h.finalize()),
                    sequence_length: 3,
                    valid_tokens: 2,
                },
                negative: None,
            }
        })
        .collect();
    let samples: Vec<_> = noise
        .chunks_exact(2)
        .map(|chunk| {
            let mut h = Sha256::new();
            for v in chunk {
                h.update(f32::to_le_bytes(*v));
            }
            MossImageSampleReceipt {
                seed: None,
                noise_sha256: format!("{:x}", h.finalize()),
            }
        })
        .collect();
    assert_supplied_order(&conditioning, &samples, 2, &embeddings, &mask, &noise, 2);
    let mut wrong = conditioning.clone();
    wrong.swap(0, 1);
    assert!(std::panic::catch_unwind(|| assert_supplied_order(
        &wrong,
        &samples,
        2,
        &embeddings,
        &mask,
        &noise,
        2
    ))
    .is_err());
    assert!(std::panic::catch_unwind(|| assert_supplied_order(
        &conditioning[..1],
        &samples,
        2,
        &embeddings,
        &mask,
        &noise,
        2
    ))
    .is_err());
}
