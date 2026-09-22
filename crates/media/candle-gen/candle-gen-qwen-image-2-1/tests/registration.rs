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

/// The descriptor axes the two backends must agree on field-for-field. The two deliberate
/// differences are `backend`/`mac_only` (this is the candle route) and `supported_quants`: MLX
/// quantizes the DiT's Linears at load, candle has no affine-quantize-at-load path and instead
/// loads an already-packed snapshot through its packed-detect Linears, so it advertises none.
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

    // The two deliberate differences, asserted rather than assumed.
    assert!(!caps.mac_only, "the candle route is not mac-only");
    assert_eq!(
        caps.supported_quants,
        &[] as &[Quant],
        "candle has no on-the-fly Q4/Q8 for this family"
    );
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
