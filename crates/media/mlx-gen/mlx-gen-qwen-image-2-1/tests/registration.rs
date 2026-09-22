//! Explicit provider-catalog coverage for Qwen-Image 2.1. No weights required.

use mlx_gen::gen_core::{ConditioningKind, Quant, SizeFloor};

#[test]
fn qwen_image_2_1_is_exported_by_provider_catalog_as_a_distinct_engine() {
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let ids: Vec<_> = registry.generators().map(|r| (r.descriptor)().id).collect();
    assert_eq!(ids, ["qwen_image_2_1"]);
    let reg = registry.generators().next().unwrap();
    let d = (reg.descriptor)();
    assert_eq!(
        d.family, "qwen-image-2-1",
        "not an alias of the 2512 `qwen-image` family"
    );
    assert_eq!(d.backend, "mlx");
    assert!(
        reg.footprint.is_some(),
        "per-component footprint is published"
    );
}

#[test]
fn advertised_surface_matches_the_story_contract() {
    let d = mlx_gen_qwen_image_2_1::descriptor();
    let caps = &d.capabilities;
    // Size: the seven presets plus the 32-px grid; steps unconstrained (>= 2 at validate time);
    // seed + guidance (true CFG) reach the runtime.
    assert_eq!(caps.min_size, 32);
    assert_eq!(caps.max_size, 2752);
    assert_eq!(
        caps.size_floor,
        SizeFloor::RangeCheckedOnGrid { multiple: 32 }
    );
    assert!(caps.supports_negative_prompt && caps.supports_guidance && caps.supports_true_cfg);
    assert!(caps.requires_sigma_shift);
    assert!(caps.supports_sequential_offload);
    assert_eq!(caps.supported_quants, &[Quant::Q4, Quant::Q8]);
    // Reference conditioning (sc-24110): one upstream call takes one ordered list of one to ten
    // condition images, reached through either kind. No `Mask` — upstream has no mask input.
    assert_eq!(
        caps.conditioning,
        vec![
            ConditioningKind::Reference,
            ConditioningKind::MultiReference
        ]
    );
    assert!(caps.accepts(ConditioningKind::Reference));
    assert!(caps.accepts(ConditioningKind::MultiReference));
    assert!(!caps.accepts(ConditioningKind::Mask));
    assert_eq!(mlx_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES, 10);
    assert!(
        !caps.supports_preview,
        "no fitted 64-channel preview projection yet"
    );
    assert!(!caps.samplers.is_empty() && !caps.schedulers.is_empty());
    assert_eq!(
        d.denoiser_output_latent_space,
        Some(&mlx_gen::gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE)
    );
    assert_eq!(mlx_gen_qwen_image_2_1::SIZE_MULTIPLE, 32);
    assert_eq!(mlx_gen_qwen_image_2_1::DEFAULT_STEPS, 40);
    assert_eq!(mlx_gen_qwen_image_2_1::PRESETS.len(), 7);
}

#[test]
fn frozen_upstream_revisions_are_recorded() {
    assert_eq!(
        mlx_gen_qwen_image_2_1::UPSTREAM_HF_REVISION,
        "790c92633540aa0cb11d9abf19eb46d861714758"
    );
    assert_eq!(
        mlx_gen_qwen_image_2_1::UPSTREAM_GITHUB_REVISION,
        "fb7ae1d1f9611cd91524d03c53c5246b36ac8577"
    );
    assert_eq!(
        mlx_gen_qwen_image_2_1::UPSTREAM_DIFFUSERS_REVISION,
        "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
    );
    assert!(mlx_gen_qwen_image_2_1::UPSTREAM_LICENSE.contains("Qwen Research License"));
    let notice = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/NOTICE")).unwrap();
    assert!(notice.contains(mlx_gen_qwen_image_2_1::UPSTREAM_LICENSE_NOTICE));
    let upstream =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/UPSTREAM.md")).unwrap();
    for sha in [
        mlx_gen_qwen_image_2_1::UPSTREAM_HF_REVISION,
        mlx_gen_qwen_image_2_1::UPSTREAM_GITHUB_REVISION,
        mlx_gen_qwen_image_2_1::UPSTREAM_DIFFUSERS_REVISION,
    ] {
        assert!(upstream.contains(sha), "UPSTREAM.md records {sha}");
    }
}

/// The released `processor/preprocessor_config.json` pixel budget, which
/// [`VisionConfig::output_resolution`] derives upstream's 1024-px condition-image fit from.
///
/// It is pinned here rather than only inside a hand-written test JSON blob: the derivation is
/// `min(1024, sqrt(max_pixels))` clamped up by `sqrt(min_pixels)`, so a snapshot bump that moved
/// the budget would silently move every reference's fit — and with it the whole joint layout —
/// while every parity fixture (dumped at the tiny snapshot's own budget) stayed green.
#[test]
fn the_released_processor_pixel_budget_pins_the_condition_image_fit() {
    let vision = mlx_gen_qwen_image_2_1::VisionConfig::from_value(
        &serde_json::json!({
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_size": 1152,
                "in_channels": 3,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        }),
        Some(&serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            // `Qwen/Qwen-Image-2.1` @ UPSTREAM_HF_REVISION, verbatim.
            "size": { "shortest_edge": 65536, "longest_edge": 16777216 }
        })),
    )
    .unwrap();
    assert_eq!(vision.processor.min_pixels, 65_536, "256 x 256");
    assert_eq!(vision.processor.max_pixels, 16_777_216, "4096 x 4096");
    assert_eq!(vision.processor.patch_size, 16);
    assert_eq!(vision.processor.merge_size, 2);
    assert_eq!(vision.processor.temporal_patch_size, 2);
    assert_eq!(vision.processor.mean, [0.5, 0.5, 0.5]);
    assert_eq!(vision.processor.std, [0.5, 0.5, 0.5]);
    assert_eq!(vision.tower.deepstack_visual_indexes, vec![8, 16, 24]);
    // The whole point of the budget: it is what puts the fit on upstream's literal default.
    assert_eq!(
        vision.output_resolution(),
        mlx_gen_qwen_image_2_1::OUTPUT_RESOLUTION,
        "the released budget must derive upstream's own `output_resolution` default"
    );
    assert_eq!(mlx_gen_qwen_image_2_1::OUTPUT_RESOLUTION, 1024);
}
