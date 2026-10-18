//! Explicitly admitted terminal acceptance only; never acquires or discovers model assets.
use mlx_gen::gen_core::moss_image::*;
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
