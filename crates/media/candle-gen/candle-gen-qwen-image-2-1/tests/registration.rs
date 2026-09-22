//! Explicit provider-catalog coverage for the Candle Qwen-Image 2.1 route, and the
//! field-for-field descriptor agreement with its MLX twin. No weights required.

use candle_gen::gen_core::{ConditioningKind, Quant, SizeFloor};

#[test]
fn qwen_image_2_1_is_exported_by_provider_catalog_as_a_distinct_engine() {
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let ids: Vec<_> = registry.generators().map(|r| (r.descriptor)().id).collect();
    assert_eq!(ids, ["qwen_image_2_1"]);
    let d = (registry.generators().next().unwrap().descriptor)();
    assert_eq!(
        d.family, "qwen-image-2-1",
        "not an alias of the 2512 `qwen-image` family"
    );
    assert_eq!(d.backend, "candle");
}

/// The descriptor axes the two backends must agree on field-for-field.
///
/// After sc-24112 there is exactly **one** deliberate difference left: `backend`/`mac_only`. The two
/// backends now advertise the same `supported_quants` and the same `component_precision_floors`,
/// because the Q4/Q8 tiers are *installable* on both — the tier artefacts are produced once by
/// `mlx_gen_qwen_image_2_1::convert` and loaded by both backends' packed-detect projections. What
/// still differs is how a tier is *reached*, which is a load-path property rather than a descriptor
/// one: MLX can also quantize a dense snapshot at load, candle cannot and refuses that with a typed
/// `Unsupported` (asserted in `tiers::the_descriptor_advertises_what_is_actually_installable`).
#[test]
fn advertised_surface_matches_the_story_contract() {
    let d = candle_gen_qwen_image_2_1::descriptor();
    let caps = &d.capabilities;
    assert_eq!(caps.min_size, 32);
    assert_eq!(caps.max_size, 2752);
    assert_eq!(
        caps.size_floor,
        SizeFloor::RangeCheckedOnGrid { multiple: 32 }
    );
    assert!(caps.supports_negative_prompt && caps.supports_guidance && caps.supports_true_cfg);
    assert!(caps.requires_sigma_shift);
    assert!(caps.supports_sequential_offload);
    assert!(!caps.supports_lora && !caps.supports_lokr);
    assert_eq!(caps.max_count, 8);
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
    assert_eq!(candle_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES, 10);
    assert!(
        !caps.supports_preview,
        "no fitted 64-channel preview projection yet"
    );
    assert!(!caps.samplers.is_empty() && !caps.schedulers.is_empty());
    assert_eq!(
        d.denoiser_output_latent_space,
        Some(&candle_gen::gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE)
    );
    assert_eq!(candle_gen_qwen_image_2_1::SIZE_MULTIPLE, 32);
    assert_eq!(candle_gen_qwen_image_2_1::DEFAULT_STEPS, 40);
    assert_eq!(candle_gen_qwen_image_2_1::DEFAULT_TRUE_CFG, 1.0);
    assert_eq!(candle_gen_qwen_image_2_1::PRESETS.len(), 7);

    // The one remaining deliberate difference, asserted rather than assumed.
    assert!(!caps.mac_only, "the candle route is not mac-only");

    // sc-24112: both affine tiers are installable on this backend, so the descriptor says so.
    // Advertising them is what lets the worker's A-B tier toggle reach the candle route at all.
    assert_eq!(caps.supported_quants, &[Quant::Q4, Quant::Q8]);
    // ...and a tier is a WHOLE-PIPELINE contract: every packable component runs the width the
    // caller selected, so there is no component precision floor to declare. Both backends agree.
    //
    // *Mutation that reds this:* reinstating the withdrawn Q4 -> Q8 text-encoder floor.
    assert!(caps.component_precision_floors.is_empty());
}

/// The seven presets and the size grid are the SAME numbers the MLX crate's `config.rs` carries —
/// pinned here so a divergence in either backend's preset table is a test failure, not a silently
/// different render surface.
#[test]
fn presets_match_the_upstream_table() {
    let presets: Vec<(&str, u32, u32)> = candle_gen_qwen_image_2_1::PRESETS
        .iter()
        .map(|p| (p.ratio, p.width, p.height))
        .collect();
    assert_eq!(
        presets,
        vec![
            ("1:1", 2048, 2048),
            ("4:3", 2400, 1792),
            ("3:4", 1792, 2400),
            ("3:2", 2528, 1696),
            ("2:3", 1696, 2528),
            ("16:9", 2752, 1536),
            ("9:16", 1536, 2752),
        ]
    );
    for p in candle_gen_qwen_image_2_1::PRESETS {
        assert_eq!(p.width % candle_gen_qwen_image_2_1::SIZE_MULTIPLE, 0);
        assert_eq!(p.height % candle_gen_qwen_image_2_1::SIZE_MULTIPLE, 0);
    }
}

#[test]
fn frozen_upstream_revisions_are_recorded() {
    assert_eq!(
        candle_gen_qwen_image_2_1::UPSTREAM_HF_REVISION,
        "790c92633540aa0cb11d9abf19eb46d861714758"
    );
    assert_eq!(
        candle_gen_qwen_image_2_1::UPSTREAM_GITHUB_REVISION,
        "fb7ae1d1f9611cd91524d03c53c5246b36ac8577"
    );
    assert_eq!(
        candle_gen_qwen_image_2_1::UPSTREAM_DIFFUSERS_REVISION,
        "8b3c707ebd3ec4881f4190cf42931da07eaf3b65"
    );
    assert!(candle_gen_qwen_image_2_1::UPSTREAM_LICENSE.contains("Qwen Research License"));
    let notice = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/NOTICE")).unwrap();
    assert!(notice.contains(candle_gen_qwen_image_2_1::UPSTREAM_LICENSE_NOTICE));
    let upstream =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/UPSTREAM.md")).unwrap();
    for sha in [
        candle_gen_qwen_image_2_1::UPSTREAM_HF_REVISION,
        candle_gen_qwen_image_2_1::UPSTREAM_GITHUB_REVISION,
        candle_gen_qwen_image_2_1::UPSTREAM_DIFFUSERS_REVISION,
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
    let vision = candle_gen_qwen_image_2_1::VisionConfig::from_value(
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
        candle_gen_qwen_image_2_1::OUTPUT_RESOLUTION,
        "the released budget must derive upstream's own `output_resolution` default"
    );
    assert_eq!(candle_gen_qwen_image_2_1::OUTPUT_RESOLUTION, 1024);
}
