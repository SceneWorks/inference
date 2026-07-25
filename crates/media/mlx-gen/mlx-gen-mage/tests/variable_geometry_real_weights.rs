//! Real-weight acceptance gate for sc-14043's variable-geometry public pipeline.
//!
//! Run alone on an authorized Metal host:
//! `MAGE_SNAPSHOT=/path/to/Mage-Flow cargo test -p mlx-gen-mage --test
//! variable_geometry_real_weights --release -- --ignored --nocapture`

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_mage::model::{load, MageVariant, MAX_COUNT};
use mlx_gen_mage::{GenerationSample, GsKey, MageFlowPipeline};

#[test]
#[ignore = "needs a complete MAGE_SNAPSHOT and an authorized Metal device"]
fn both_four_to_one_orientations_render_exact_uncorrupted_shapes() {
    let root = std::env::var("MAGE_SNAPSHOT")
        .expect("set MAGE_SNAPSHOT to a complete microsoft/Mage-Flow snapshot");
    let pipeline = MageFlowPipeline::load(root).unwrap();
    let samples = [
        GenerationSample {
            prompt: "a red sailboat on calm blue water at sunset",
            negative_prompt: "fog, text",
            width: 2048,
            height: 2048,
            seed: 14043,
        },
        GenerationSample {
            prompt: "a white marble statue in a dark museum",
            negative_prompt: "buildings, text",
            width: 2048,
            height: 2048,
            // Deliberately the same seed as sample zero: a changed result discriminates prompt
            // conditioning from seed-only output.
            seed: 14043,
        },
        GenerationSample {
            prompt: "a macro photograph of a green beetle",
            negative_prompt: "painting, text",
            width: 2048,
            height: 2048,
            seed: 14045,
        },
        GenerationSample {
            prompt: "a panoramic red train crossing a snowy valley",
            negative_prompt: "text, blur",
            width: 2048,
            height: 512,
            seed: 14046,
        },
        GenerationSample {
            prompt: "a tall waterfall in a narrow green canyon",
            negative_prompt: "buildings, text",
            width: 512,
            height: 2048,
            seed: 14047,
        },
    ];

    mlx_rs::memory::reset_peak_memory();
    let trace = pipeline
        .generate_batch_trace(
            &samples,
            20,
            3.0,
            &GsKey::from_u64(14043),
            false,
            &mut |_| {},
        )
        .unwrap();
    let peak_gb = mlx_rs::memory::get_peak_memory() as f64 / 1e9;
    println!("variable-geometry packed generation peak: {peak_gb:.3} GB");
    // The accepted two-step run peaked at 28.060 GB. Steps are evaluated eagerly, so the default
    // 20-step schedule must remain in the same live-set class; 35 GB leaves ~25% allocator/OS
    // variation while catching a retained-step graph or accidental all-samples-at-once pack.
    assert!(
        peak_gb < 35.0,
        "57,344-token generation exceeded the 35 GB peak ceiling: {peak_gb:.3} GB"
    );
    assert_eq!(trace.packs.len(), 2);
    assert_eq!(trace.packs[0].sample_range, 0..3);
    assert_eq!(trace.packs[0].image_tokens, 49_152);
    assert_eq!(trace.packs[1].sample_range, 3..5);
    assert_eq!(trace.packs[1].grids, vec![(32, 128), (128, 32)]);
    assert_eq!(trace.samples.len(), samples.len());

    for (index, (output, sample)) in trace.samples.iter().zip(samples).enumerate() {
        mlx_rs::transforms::eval([&output.image_u8, &output.final_tokens]).unwrap();
        assert_eq!(
            output.image_u8.shape(),
            [sample.height as i32, sample.width as i32, 3]
        );
        assert_eq!(
            output.final_tokens.shape(),
            [1, (sample.height * sample.width / 256) as i32, 128]
        );
        let bytes = output.image_u8.as_slice::<u8>();
        let (min, max) = bytes
            .iter()
            .fold((u8::MAX, u8::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            max.saturating_sub(min) >= 16,
            "sample {index}, {}x{} render has collapsed dynamic range ({min}..={max})",
            sample.width,
            sample.height
        );
        let row_bytes = sample.width as usize * 3;
        let identical_adjacent_rows = bytes
            .chunks_exact(row_bytes)
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|rows| rows[0] == rows[1])
            .count();
        assert!(
            identical_adjacent_rows < sample.height as usize / 10,
            "sample {index} has {identical_adjacent_rows} duplicated adjacent rows"
        );
    }
    let first = trace.samples[0].image_u8.as_slice::<u8>();
    let second = trace.samples[1].image_u8.as_slice::<u8>();
    let changed = first.iter().zip(second).filter(|(a, b)| a != b).count();
    assert!(
        changed > first.len() / 100,
        "same-seed, different-prompt renders differ at only {changed}/{} bytes",
        first.len()
    );
}

#[test]
#[ignore = "needs a complete MAGE_SNAPSHOT and an authorized Metal device"]
fn registered_generator_routes_multi_output_requests_through_budgeted_packs() {
    let root = std::env::var("MAGE_SNAPSHOT")
        .expect("set MAGE_SNAPSHOT to a complete microsoft/Mage-Flow snapshot");
    let generator = load(
        MageVariant::Rl,
        &LoadSpec::new(WeightsSource::Dir(root.into())),
    )
    .unwrap();
    assert_eq!(generator.descriptor().capabilities.max_count, MAX_COUNT);
    let request = GenerationRequest {
        prompt: "a detailed botanical illustration".into(),
        negative_prompt: Some("text, blur".into()),
        width: 2048,
        height: 2048,
        count: 4,
        steps: Some(1),
        guidance: Some(1.0),
        seed: Some(14043),
        ..Default::default()
    };
    mlx_rs::memory::reset_peak_memory();
    let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
    else {
        panic!("Mage image generator returned a non-image output");
    };
    assert_eq!(images.len(), request.count as usize);
    assert!(images
        .iter()
        .all(|image| image.width == 2048 && image.height == 2048));
    let provider_peak_gb = mlx_rs::memory::get_peak_memory() as f64 / 1e9;
    assert!(
        provider_peak_gb < 30.0,
        "registered 65,536-token request exceeded the 30 GB peak ceiling: {provider_peak_gb:.3} GB"
    );

    // Independently execute the explicit seed sequence through the public packed pipeline. Keeping
    // the same pack is required by the frozen reference: pack-relative MSRoPE frame indices mean
    // isolating each sample would deliberately change its positions. This proves the provider used
    // base_seed+i and returned each result in exact request order.
    drop(generator);
    mlx_rs::memory::clear_cache();
    let pipeline = MageFlowPipeline::load(
        std::env::var("MAGE_SNAPSHOT").expect("MAGE_SNAPSHOT disappeared during the test"),
    )
    .unwrap();
    let explicit = (0..request.count)
        .map(|index| GenerationSample {
            prompt: &request.prompt,
            negative_prompt: request.negative_prompt.as_deref().unwrap(),
            width: request.width,
            height: request.height,
            seed: request.seed.unwrap() as i64 + index as i64,
        })
        .collect::<Vec<_>>();
    let expected = pipeline
        .generate_batch(
            &explicit,
            request.steps.unwrap() as usize,
            request.guidance.unwrap(),
            &mlx_gen_mage::resolve_gs_key(None).unwrap(),
            false,
        )
        .unwrap();
    for (index, (image, expected)) in images.iter().zip(expected).enumerate() {
        mlx_rs::transforms::eval([&expected]).unwrap();
        assert_eq!(
            image.pixels,
            expected.as_slice::<u8>(),
            "registered output {index} did not match explicit base_seed+{index}"
        );
    }
    println!(
        "registered four-output 65,536-token request peak: {provider_peak_gb:.3} GB; exact seed/order replay passed"
    );
}
