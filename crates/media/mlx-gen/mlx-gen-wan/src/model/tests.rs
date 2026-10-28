use super::*;

fn req(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "x".into(),
        width,
        height,
        ..Default::default()
    }
}

/// A tiny solid-color RGB conditioning frame.
fn dummy_image() -> Image {
    Image {
        width: 64,
        height: 64,
        pixels: vec![128u8; 64 * 64 * 3],
    }
}

fn keyframe(frame_idx: i32, strength: f32) -> Conditioning {
    Conditioning::Keyframe {
        image: dummy_image(),
        frame_idx,
        strength,
    }
}

/// **The seam assertion** (sc-19571). `resolve_ti2v_pins` is the single hop between the request
/// and [`build_ti2v_mask`], so this pins what the mask builder is actually handed — with
/// **non-default** strengths on both ends of a first_last_frame pair, because the defect it
/// replaces produced a full 1.0 pin for every keyframe and would satisfy any assertion written
/// against the default.
///
/// Mutation guard: change the `pins.push((idx as usize, kf.strength))` in `resolve_ti2v_pins` to
/// `(idx as usize, 1.0)` — i.e. re-introduce the exact defect — and the strength asserts below
/// go red while every index assert still passes. That asymmetry is the point: indices were never
/// the broken half.
#[test]
fn ti2v_pins_carry_each_keyframes_own_strength_to_the_mask_builder() {
    // t_lat = 4 latent frames. first_last_frame with two DIFFERENT, non-default strengths, and
    // the last frame addressed negative-from-end (the shape SceneWorks actually sends).
    let mut r = req(704, 480);
    // 0.25 / 0.75 are exactly representable in f32, so `1 − strength` is exact and the mask
    // assertion below can be `==` rather than an epsilon that could hide a wrong-but-close pin.
    r.conditioning = vec![keyframe(0, 0.25), keyframe(-1, 0.75)];
    let pins = resolve_ti2v_pins(&r, 4).expect("pins");
    assert_eq!(
        pins,
        vec![(0usize, 0.25f32), (3usize, 0.75f32)],
        "each keyframe's own strength must reach the mask builder, in request order"
    );

    // …and the mask the builder derives from them carries `1 − strength` at each pinned frame.
    // z=1, t_lat=4, h=w=1, patch (1,1,1) → one token per latent frame.
    let (mask, tokens) = build_ti2v_mask(&pins, 1, 4, 1, 1, (1, 1, 1));
    assert_eq!(mask.as_slice::<f32>(), &[0.75, 1.0, 1.0, 0.25]);
    assert_eq!(tokens.as_slice::<f32>(), &[0.75, 1.0, 1.0, 0.25]);
}

/// The `Reference` half of the same seam: a single conditioning image pins latent frame 0 at the
/// strength the request resolves — the per-`Reference` value first, then the request-level
/// img2img `strength`, then a full pin. Non-default at every rung.
#[test]
fn ti2v_pins_resolve_the_reference_strength_precedence() {
    let mut r = req(704, 480);
    // Per-reference strength wins.
    r.conditioning = vec![Conditioning::Reference {
        image: dummy_image(),
        strength: Some(0.4),
    }];
    r.strength = Some(0.9);
    assert_eq!(resolve_ti2v_pins(&r, 4).unwrap(), vec![(0usize, 0.4f32)]);
    // No per-reference strength → the request-level img2img strength.
    r.conditioning = vec![Conditioning::Reference {
        image: dummy_image(),
        strength: None,
    }];
    assert_eq!(resolve_ti2v_pins(&r, 4).unwrap(), vec![(0usize, 0.9f32)]);
    // Neither → a full pin (the historical hard mask).
    r.strength = None;
    assert_eq!(resolve_ti2v_pins(&r, 4).unwrap(), vec![(0usize, 1.0f32)]);
    // No image conditioning at all → no pins, i.e. pure-noise T2V.
    r.conditioning = vec![];
    assert!(resolve_ti2v_pins(&r, 4).unwrap().is_empty());
}

/// Out-of-bounds latent indices still reject (this moved into `resolve_ti2v_pins` with the
/// strengths — the guard must not have been lost in the move).
#[test]
fn ti2v_pins_reject_out_of_bounds_latent_indices() {
    let mut r = req(704, 480);
    r.conditioning = vec![keyframe(4, 1.0)];
    let err = resolve_ti2v_pins(&r, 4).unwrap_err().to_string();
    assert!(err.contains("out of bounds"), "unexpected: {err}");
    r.conditioning = vec![keyframe(-5, 1.0)];
    assert!(resolve_ti2v_pins(&r, 4).is_err());
}

/// sc-19571 — the strengths now reach `1 − strength`, so a value outside `[0,1]` would build a
/// NEGATIVE mask: negative per-token timesteps and an extrapolating blend, i.e. silent garbage.
/// Mirrors candle's `validate_rejects_non_finite_or_out_of_range_ti2v_strengths`.
///
/// Two gates, deliberately asserted apart: **non-finite** is the shared gen-core floor
/// (`ensure_finite_floats`, which sc-19571 also taught about `Keyframe.strength`), **finite but
/// out of range** is this provider's own gate. Mutation guard: delete the
/// `reject_out_of_range_strengths` call in `validate_impl` and the `[0, 1]` asserts go red while
/// the finiteness ones still pass — which is how you can tell the two gates are distinct and
/// neither is carrying the other.
#[test]
fn validate_rejects_out_of_range_conditioning_strengths_5b() {
    let wan = wan_5b();
    for bad in [1.5f32, -0.1] {
        let mut r = req(704, 480);
        r.conditioning = vec![keyframe(0, bad)];
        let err = wan.validate_impl(&r).unwrap_err().to_string();
        assert!(
            err.contains("Keyframe strength must be finite and in [0, 1]"),
            "keyframe {bad}: {err}"
        );

        let mut r = req(704, 480);
        r.conditioning = vec![Conditioning::Reference {
            image: dummy_image(),
            strength: Some(bad),
        }];
        let err = wan.validate_impl(&r).unwrap_err().to_string();
        assert!(
            err.contains("Reference strength must be finite and in [0, 1]"),
            "reference {bad}: {err}"
        );

        let mut r = req(704, 480);
        r.strength = Some(bad);
        let err = wan.validate_impl(&r).unwrap_err().to_string();
        assert!(
            err.contains("img2img strength must be finite and in [0, 1]"),
            "img2img {bad}: {err}"
        );
    }
    // Non-finite is caught one layer up, by the shared finiteness floor.
    for bad in [f32::NAN, f32::INFINITY] {
        let mut r = req(704, 480);
        r.conditioning = vec![keyframe(0, bad)];
        let err = wan.validate_impl(&r).unwrap_err().to_string();
        assert!(
            err.contains("conditioning.keyframe.strength") && err.contains("must be finite"),
            "keyframe {bad}: {err}"
        );
    }
    // The in-range partial pins this story exists to enable still validate.
    let mut ok = req(704, 480);
    ok.conditioning = vec![keyframe(0, 0.35), keyframe(-1, 0.0)];
    assert!(wan.validate_impl(&ok).is_ok());
}

/// sc-19571 — **refuse, do not ignore.** The A14B I2V is channel-concat and has no denoise mask,
/// so a conditioning strength cannot be honored there; it must be rejected with a message naming
/// the mechanism rather than accepted and dropped. Mutation guard: delete the `strength != 1.0`
/// branch in `Wan14b::validate_impl` and the `unwrap_err`s below panic.
#[test]
fn validate_refuses_conditioning_strength_on_a14b_i2v() {
    let wan = Wan14b {
        descriptor: descriptor_i2v_14b(),
        config: WanModelConfig::wan22_i2v_14b(),
        root: PathBuf::new(),
        adapters: vec![],
        quant: None,
        offload_policy: OffloadPolicy::Resident,
        i2v_memory: None,
    };
    let reference = |strength: Option<f32>| Conditioning::Reference {
        image: dummy_image(),
        strength,
    };
    // A full pin — the only thing channel-concat can express — validates.
    let mut ok = req(704, 480);
    ok.conditioning = vec![reference(Some(1.0))];
    assert!(wan.validate_impl(&ok).is_ok(), "strength 1.0 must validate");
    let mut ok = req(704, 480);
    ok.conditioning = vec![reference(None)];
    assert!(wan.validate_impl(&ok).is_ok(), "unset must validate");

    // Anything else is refused, on either carrier.
    let mut bad = req(704, 480);
    bad.conditioning = vec![reference(Some(0.5))];
    let err = wan.validate_impl(&bad).unwrap_err().to_string();
    assert!(
        err.contains("conditioning strength is not supported")
            && err.contains("channel-concatenated"),
        "unexpected: {err}"
    );
    let mut bad = req(704, 480);
    bad.conditioning = vec![reference(None)];
    bad.strength = Some(0.5);
    assert!(
        wan.validate_impl(&bad).is_err(),
        "the request-level img2img strength resolves into the same pin and must be refused too"
    );
}

#[test]
fn resolve_capped_dims_aligns_down_only() {
    // 14B aligns to patch.{1,2}·vae_stride.{1,2} = 2·8 = 16, so 130 → 128 on both axes.
    let cfg = WanModelConfig::wan22_t2v_14b();
    assert_eq!(
        resolve_capped_dims(&req(130, 130), &cfg, A14bProviderVae::VAE_TILING),
        (128, 128)
    );
    // Already on-grid → unchanged.
    assert_eq!(
        resolve_capped_dims(&req(128, 256), &cfg, A14bProviderVae::VAE_TILING),
        (128, 256)
    );
    // The 5B's z48 VAE gives a 32-px grid instead.
    let five_b = WanModelConfig::wan22_ti2v_5b();
    assert_eq!(
        resolve_capped_dims(&req(720, 720), &five_b, Ti2vProviderVae::VAE_TILING),
        (704, 704)
    );
    assert_eq!(
        resolve_capped_dims(&req(512, 512), &five_b, Ti2vProviderVae::VAE_TILING),
        (512, 512)
    );
}

#[test]
fn benchmark_scope_forces_dense_wan_spatial_and_temporal_tiles() {
    let scope = mlx_gen::diagnostics::begin_benchmark_request(
        "wan-fixed-tile",
        "wan_video",
        &[],
        Some(mlx_gen::diagnostics::BenchmarkDecodeControl {
            spatial_tile_px: 256,
            spatial_overlap_px: 64,
            temporal_tile_frames: Some(32),
            temporal_overlap_frames: Some(8),
        }),
        |_| {},
    )
    .unwrap();
    let tiling = dense_decode_tiling(&GenerationRequest::default(), 832, 480, 81)
        .unwrap()
        .unwrap();
    let spatial = tiling.spatial.unwrap();
    let temporal = tiling.temporal.unwrap();
    assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));
    assert_eq!((temporal.tile_frames, temporal.overlap_frames), (32, 8));
    scope.finish();
    assert_eq!(mlx_gen::diagnostics::benchmark_decode_control(), None);
}

#[test]
fn effective_te_quant_floors_at_q8_when_dit_is_quantized() {
    // sc-12831: the UMT5 TE packs to Q8 (the near-lossless floor) whenever the DiT is on an
    // MLX-affine quantized tier, and stays dense (None) otherwise. Pins the NON-default width (Q8,
    // not the DiT's) so tier-matching the DiT to Q4 — or dropping the Nvfp4 exclusion — goes red.
    let expect_q8 = Some(WanQuant {
        bits: 8,
        group_size: mlx_gen::quant::DEFAULT_GROUP_SIZE,
    });
    let mut dense = WanModelConfig::wan22_ti2v_5b();
    dense.quantization = None; // a dense bf16 snapshot

    // bf16 tier (dense snapshot, no load-time quant) → TE stays dense.
    assert_eq!(effective_te_quant(&dense, None), None);
    // Dense snapshot + a load-time Q4/Q8 `spec.quantize` → TE packs, ALWAYS at Q8 (not the DiT's 4).
    assert_eq!(effective_te_quant(&dense, Some(Quant::Q4)), expect_q8);
    assert_eq!(effective_te_quant(&dense, Some(Quant::Q8)), expect_q8);
    // Nvfp4 is candle-only (never an MLX affine width) → excluded, TE stays dense.
    assert_eq!(effective_te_quant(&dense, Some(Quant::Nvfp4)), None);

    // A pre-quantized q4 snapshot (`config.quantization` = bits 4) → the TE STILL floors at Q8, not 4.
    let mut prequant_q4 = dense.clone();
    prequant_q4.quantization = Some(WanQuant {
        bits: 4,
        group_size: 64,
    });
    assert_eq!(effective_te_quant(&prequant_q4, None), expect_q8);
}

/// sc-12308 — this replaces `resolve_capped_dims_enforces_max_area_cap`, which asserted that an
/// over-area request SHRINKS. It no longer does: the cap is enforced by rejecting in
/// `validate_impl`, so this function only aligns. Alignment rounds *down*, so it can never push
/// a validated request back over budget — which is why the cap check can live at validate time.
#[test]
fn resolve_capped_dims_never_refits_over_area() {
    let cfg = WanModelConfig::wan22_ti2v_5b();
    // The geometry `validate_impl` would have rejected passes through UNCHANGED (bar alignment)
    // rather than being silently refit to something the caller never asked for.
    assert_eq!(
        resolve_capped_dims(&req(2048, 2048), &cfg, Ti2vProviderVae::VAE_TILING),
        (2048, 2048)
    );
    assert!(
        2048 * 2048 > cfg.max_area,
        "the guard belongs to validate, not to this function"
    );

    // The 14B family's canonical 720p is at its cap and is a fixed point of the alignment.
    let a14b = WanModelConfig::wan22_i2v_14b();
    assert_eq!(
        resolve_capped_dims(&req(1280, 720), &a14b, A14bProviderVae::VAE_TILING),
        (1280, 720)
    );
    assert_eq!(1280 * 720, a14b.max_area);
}

/// sc-12736 + sc-12796 (epic 12732): ALL three Wan2.2 MLX variants advertise
/// `supports_sequential_offload` so the worker's fit-gate can tell "bounds footprint under
/// `Sequential`" from a no-op fallback — the two A14B MoE variants via the expert swap (sc-12736),
/// the dense TI2V-5B via the staged-component `clear_cache` flush (sc-12796). Mutation guard:
/// flipping ANY of these back to `false` regresses the advertisement its offload path relies on.
#[test]
fn all_wan_mlx_variants_advertise_sequential_offload() {
    assert!(
        descriptor_t2v_14b()
            .capabilities
            .supports_sequential_offload,
        "T2V-A14B must advertise the expert-swap sequential offload"
    );
    assert!(
        descriptor_i2v_14b()
            .capabilities
            .supports_sequential_offload,
        "I2V-A14B must advertise the expert-swap sequential offload"
    );
    assert!(
        descriptor().capabilities.supports_sequential_offload,
        "the dense TI2V-5B must advertise sequential offload (sc-12796) — the staged TE/VAE \
             clear_cache flush bounds its unified-memory footprint"
    );
}

#[test]
fn all_registered_base_wan_variants_also_declare_unconditional_phase_staging() {
    for descriptor in [descriptor(), descriptor_t2v_14b(), descriptor_i2v_14b()] {
        assert_eq!(
            descriptor.capabilities.staged_residency_availability(),
            mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged,
            "{} must distinguish its always-staged physical path from the stronger selectable \
                 Sequential controls it also supports",
            descriptor.id
        );
        assert!(descriptor.capabilities.supports_sequential_offload);
    }
}

/// sc-12736: the sc-4986 pre-flight denoise guard's expert-byte accounting must track the ACTUAL
/// resident-expert count of the path it guards — one under every `Sequential` swap and two under
/// `Resident`. Getting this wrong either
/// false-rejects the swappable job on a one-expert machine (summing when it should be one) or waves
/// through a both-resident job that then OOMs (one when it should be two). Distinct per-
/// expert sizes (`lo < hi`) so `max` vs `sum` can't alias.
#[test]
fn preflight_guard_budgets_one_expert_for_every_sequential_sampler() {
    use OffloadPolicy::{Resident, Sequential};
    let (lo, hi) = (8_000_000_000u64, 9_000_000_000u64);

    // Resident → BOTH experts, whatever the sampler (native or curated).
    for s in [Some("uni_pc"), Some("heun"), None] {
        assert_eq!(
            wan14b_denoise_resident_bytes(Resident, s, lo, hi),
            lo + hi,
            "Resident must budget both experts for sampler {s:?}"
        );
    }

    // Sequential → ONE expert (the larger) for native and curated per-eval swap paths.
    for s in [
        Some("uni_pc"),
        Some("euler"),
        Some("dpmpp_2m"),
        Some("unipc"),
        Some("dpmpp2m"),
        Some("euler_ancestral"),
        Some("heun"),
        Some("dpmpp_sde"),
        Some("ddim"),
        None,
    ] {
        assert_eq!(
            wan14b_denoise_resident_bytes(Sequential, s, lo, hi),
            hi,
            "Sequential must budget ONE expert for sampler {s:?}"
        );
    }

    // Discriminator: the swap budgets strictly less than the both-resident path (the whole point).
    assert!(
        wan14b_denoise_resident_bytes(Sequential, Some("uni_pc"), lo, hi)
            < wan14b_denoise_resident_bytes(Resident, Some("uni_pc"), lo, hi),
        "the swap must budget strictly less than both-resident"
    );
}

#[test]
fn wan14b_adapter_residency_tracks_packed_folded_and_expert_routing() {
    let root_tmp = tempfile::tempdir().unwrap();
    let root = root_tmp.path().to_path_buf();
    let shared = root.join("shared.safetensors");
    let low = root.join("low.safetensors");
    let high = root.join("high.safetensors");
    std::fs::write(&shared, vec![0_u8; 10]).unwrap();
    std::fs::write(&low, vec![0_u8; 20]).unwrap();
    std::fs::write(&high, vec![0_u8; 30]).unwrap();
    let adapters = vec![
        AdapterSpec::new(shared, 1.0, mlx_gen::AdapterKind::Lora),
        AdapterSpec::new(low, 1.0, mlx_gen::AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
        AdapterSpec::new(high, 1.0, mlx_gen::AdapterKind::Lora).with_moe_expert(MoeExpert::High),
    ];

    assert_eq!(
        wan14b_adapter_bytes_per_expert(&adapters, AdapterResidencyMode::Folded),
        Some((0, 0)),
        "dense factors are folded into the base experts"
    );
    assert_eq!(
        wan14b_adapter_bytes_per_expert(&adapters, AdapterResidencyMode::Additive),
        Some((30, 40)),
        "packed factors remain resident on their routed experts"
    );
    assert_eq!(
        wan14b_denoise_resident_bytes(OffloadPolicy::Resident, None, 100 + 30, 200 + 40),
        370
    );
    assert_eq!(
        wan14b_denoise_resident_bytes(OffloadPolicy::Sequential, None, 100 + 30, 200 + 40),
        240
    );

    let missing = vec![AdapterSpec::new(
        root.join("missing.safetensors"),
        1.0,
        mlx_gen::AdapterKind::Lora,
    )];
    assert_eq!(
        wan14b_adapter_bytes_per_expert(&missing, AdapterResidencyMode::Additive),
        None
    );
}

fn wan_5b() -> Wan {
    Wan {
        descriptor: descriptor(),
        config: WanModelConfig::wan22_ti2v_5b(),
        root: PathBuf::new(),
        adapters: vec![],
        quant: None,
        offload_policy: OffloadPolicy::Resident,
        memory_strategy: None,
        memory_tier: None,
        i2v_memory: None,
    }
}

fn wan_t2v_14b() -> Wan14b {
    Wan14b {
        descriptor: descriptor_t2v_14b(),
        config: WanModelConfig::wan22_t2v_14b(),
        root: PathBuf::new(),
        adapters: vec![],
        quant: None,
        offload_policy: OffloadPolicy::Resident,
        i2v_memory: None,
    }
}

/// sc-12607 — the 5B's z48 VAE renders on a 32-px grid; candle rejects an off-32 request, so mlx
/// must too (it used to silently align it *down* to the nearest tile). Mutation guard: delete the
/// `reject_off_grid` call and `req(720, 480)` validates (720 is in range, its aligned area fits) —
/// this `unwrap_err` then panics.
#[test]
fn validate_rejects_off_grid_size_5b() {
    let wan = wan_5b();
    // On-grid, in range, under the 901 120 px cap → validates.
    assert!(wan.validate_impl(&req(704, 480)).is_ok());
    // 720 is off the 32-px grid (720 = 22.5·32) — rejected, not snapped to 704.
    let err = wan.validate_impl(&req(720, 480)).unwrap_err().to_string();
    assert!(err.contains("multiples of 32"), "unexpected: {err}");
    // Off-grid on the height axis is rejected the same way.
    assert!(wan.validate_impl(&req(704, 500)).is_err());
}

/// sc-12636 — the 5B's z48 vae22 renders rainbow garbage below a 15×15 latent grid (`< 480` px per
/// side), so candle floors its descriptor `min_size` at [`MIN_SIZE`] = 480 and mlx must match, or an
/// on-grid sub-480 request that candle *rejects* is instead *rendered as garbage* on mlx. This is a
/// SEPARATE axis from the 32-px grid stride above: the sizes here are all on-grid (multiples of 32),
/// so only the `min_size` range check (`gen_core::Capabilities::validate_request`) can reject them.
/// Mutation guard: revert the descriptor `min_size` to 32 (or drop [`MIN_SIZE`] below 480) and
/// `req(384, 384)` / `req(480, 320)` validate — the `unwrap_err` / `is_err` below then panic.
#[test]
fn validate_rejects_below_min_size_floor_5b() {
    use crate::config::{MIN_SIZE, SIZE_MULTIPLE};
    let wan = wan_5b();
    // At the floor: 480 = 15·32, on-grid, well under the 901 120 px cap → validates.
    assert!(wan.validate_impl(&req(480, 480)).is_ok());
    // 384 = 12·32 is on-grid but below the 480 coherence floor → rejected by the range check.
    let err = wan.validate_impl(&req(384, 384)).unwrap_err().to_string();
    assert!(
        err.contains("outside supported range") && err.contains(&MIN_SIZE.to_string()),
        "expected a min_size range rejection naming {MIN_SIZE}, got: {err}"
    );
    // Sub-floor on a single axis (320 = 10·32) is rejected too — both sides must clear the floor.
    assert!(wan.validate_impl(&req(480, 320)).is_err());
    // The rejected sizes are on-grid, so it is the floor — not the stride — doing the rejecting.
    assert_eq!(384 % SIZE_MULTIPLE, 0);
    assert_eq!(320 % SIZE_MULTIPLE, 0);
}

/// sc-12607 — the 14B family renders on a 16-px grid (`SIZE_MULTIPLE_14B`); same reject-not-refit
/// contract as candle. Exercised on the T2V variant (no Reference required).
#[test]
fn validate_rejects_off_grid_size_14b() {
    let wan = wan_t2v_14b();
    // On-grid (240 = 15·16, 176 = 11·16), well under the 921 600 px cap → validates.
    assert!(wan.validate_impl(&req(240, 176)).is_ok());
    // 170 is off the 16-px grid (170 = 10.625·16) — rejected.
    let err = wan.validate_impl(&req(240, 170)).unwrap_err().to_string();
    assert!(err.contains("multiples of 16"), "unexpected: {err}");
    // Off-grid on the width axis too.
    assert!(wan.validate_impl(&req(250, 176)).is_err());
}

/// sc-12607/sc-12409/sc-12587 — the worker's `pinned_engine_geometry` tie reads these `pub const`s
/// to assert each model's manifest stride. They must equal the lattice `validate`/`resolve_capped_dims`
/// actually enforce (the config-derived `grid`), or the tie would anchor a stride the engine doesn't
/// use. Mutation guard: change `SIZE_MULTIPLE` to 16 (or the config's `vae_stride`) and this fails.
#[test]
fn pinned_stride_consts_match_the_enforced_lattice() {
    use crate::config::{SIZE_MULTIPLE, SIZE_MULTIPLE_14B};
    assert_eq!(
        grid(
            &WanModelConfig::wan22_ti2v_5b(),
            Ti2vProviderVae::VAE_TILING
        ),
        (SIZE_MULTIPLE, SIZE_MULTIPLE)
    );
    assert_eq!(
        grid(
            &WanModelConfig::wan22_t2v_14b(),
            A14bProviderVae::VAE_TILING
        ),
        (SIZE_MULTIPLE_14B, SIZE_MULTIPLE_14B)
    );
    assert_eq!(
        grid(
            &WanModelConfig::wan22_i2v_14b(),
            A14bProviderVae::VAE_TILING
        ),
        (SIZE_MULTIPLE_14B, SIZE_MULTIPLE_14B)
    );
}

// ---- sc-12459: `dit_resident_bytes` shard-dir sizing must match `Weights::from_dir` ----------

/// A fresh, empty scratch dir for one sizing test (recreated per run).
///
/// Keyed by pid as well as `name`: `RUST_TEST_THREADS=1` only serializes tests *within* one
/// invocation, so it rules out cross-*test* races but not cross-*process* ones — two concurrent
/// `cargo test` runs on the same machine share `$TMPDIR`, and the `remove_dir_all` below would
/// then delete the other run's fixtures mid-test.
fn sizing_dir(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
    let d = tmp.path().join(name);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Pins the symlink traversal (adversarial-review issue on sc-12459): the `link == true`
/// assembled snapshots (convert.rs, the default local-dev / gated-e2e layout) and HF-hub caches
/// store each shard as a symlink into `blobs/`. `DirEntry::metadata` does NOT traverse the link
/// (a symlinked shard would sum as ~14 bytes → the preflight sees none of the expert weights and
/// the load dies by SIGKILL, the original F-008 failure); `std::fs::metadata` must size the
/// blob target instead.
#[test]
fn dit_resident_bytes_dir_sums_symlinked_shards_at_target_size() {
    let tmp = tempfile::tempdir().unwrap();
    let root = sizing_dir(&tmp, "symlinked_shards");
    let blobs = root.join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    let blob = blobs.join("blob0");
    std::fs::write(&blob, vec![0u8; 4096]).unwrap();

    let shard_dir = root.join("transformer");
    std::fs::create_dir_all(&shard_dir).unwrap();
    // One symlinked shard (the linked-snapshot / HF-cache layout) + one regular shard.
    std::os::unix::fs::symlink(&blob, shard_dir.join("model-00001-of-00002.safetensors")).unwrap();
    std::fs::write(
        shard_dir.join("model-00002-of-00002.safetensors"),
        vec![0u8; 2048],
    )
    .unwrap();

    assert_eq!(
        dit_resident_bytes(&[shard_dir], None),
        4096 + 2048,
        "symlinked shard must size as its blob target, not the link"
    );
}

/// The dir branch applies the exact `Weights::from_dir` selection: hidden AppleDouble sidecars
/// (`._model.safetensors` — extension-only filters admit them) and non-`.safetensors` files
/// contribute nothing, so the sizing matches what the loader will actually map.
#[test]
fn dit_resident_bytes_dir_skips_hidden_sidecars_and_non_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let shard_dir = sizing_dir(&tmp, "hidden_sidecars");
    std::fs::write(shard_dir.join("model.safetensors"), vec![0u8; 1024]).unwrap();
    std::fs::write(shard_dir.join("._model.safetensors"), vec![0u8; 999]).unwrap();
    std::fs::write(shard_dir.join("config.json"), b"{}").unwrap();

    assert_eq!(
        dit_resident_bytes(&[shard_dir], None),
        1024,
        "only the real shard `Weights::from_dir` maps may be counted"
    );
}

// ---- sc-10045: adapter routing at the two merge_adapters SITES (5B `Wan` + A14B `Wan14b`) ------
//
// These drive the real private `install_adapters_additive` / `merge_adapters` methods that the
// generate paths call — with a *tiny* synthetic `WanTransformer` (in = 64 so the block Linears
// quantize at group_size 64), never the ~7GB checkpoint. They prove: (a) a plain LoRA AND a LoKr
// install on a PACKED base with NO error (the old model.rs:614 rejection is gone; LoKr via the
// structured deferred-Kronecker path, sc-10050) with the base STILL packed; (b) LoHa on a packed
// base is the explicit typed error (deferred to sc-10051); (c) the dense fold path is untouched.

use crate::config::WanModelConfig;
use crate::transformer::WanTransformer;
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterKind;
use mlx_rs::ops::array_eq;
use mlx_rs::{Array, Dtype};
use std::collections::HashMap;
use std::path::PathBuf;

/// A tiny 1-layer Wan config: dim 64 (in = 64 is a group_size multiple, so the block Linears pack
/// to Q4/Q8), 2 heads, ffn 128. Not run through forward — only built + adapter-installed.
fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan21_t2v_1_3b();
    c.dim = 64;
    c.ffn_dim = 128;
    c.num_heads = 2;
    c.num_layers = 1;
    c.freq_dim = 16;
    c.text_dim = 64;
    c.in_dim = 4;
    c.out_dim = 4;
    c.quantization = None; // built dense; the test packs the linears after build
    c
}

fn tmp_dir(tmp: &tempfile::TempDir) -> PathBuf {
    // Per-process scratch dir — a fixed `$TMPDIR` name races a second concurrent `cargo test`.
    let d = tmp.path().join("mlx_gen_wan_model_site_test");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn bf16(vals: impl Iterator<Item = f32>, shape: &[i32]) -> Array {
    Array::from_slice(&vals.collect::<Vec<_>>(), shape)
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// Build a tiny dense [`WanTransformer`] from synthesized weights matching [`tiny_cfg`]. Every
/// tensor `from_weights` + `Block::load` requires is present; the per-block attn/ffn Linears carry
/// `in = 64`/`128` so a subsequent `.quantize(bits)` packs them (group 64).
fn tiny_transformer(tmp: &tempfile::TempDir, cfg: &WanModelConfig) -> WanTransformer {
    let dim = cfg.dim as i32;
    let ffn = cfg.ffn_dim as i32;
    let mut entries: Vec<(String, Array)> = Vec::new();
    fn lin(entries: &mut Vec<(String, Array)>, name: &str, out: i32, inp: i32, seed: f32) {
        entries.push((
            format!("{name}.weight"),
            bf16(
                (0..out * inp).map(|i| (i as f32 * 0.001 + seed).sin() * 0.05),
                &[out, inp],
            ),
        ));
        entries.push((
            format!("{name}.bias"),
            bf16((0..out).map(|i| i as f32 * 0.001), &[out]),
        ));
    }
    // Embeddings + head (shapes plausible; not exercised by forward here).
    lin(
        &mut entries,
        "patch_embedding_proj",
        dim,
        cfg.in_dim as i32,
        0.1,
    );
    lin(
        &mut entries,
        "text_embedding_0",
        dim,
        cfg.text_dim as i32,
        0.2,
    );
    lin(&mut entries, "text_embedding_1", dim, dim, 0.3);
    lin(
        &mut entries,
        "time_embedding_0",
        dim,
        cfg.freq_dim as i32,
        0.4,
    );
    lin(&mut entries, "time_embedding_1", dim, dim, 0.5);
    lin(&mut entries, "time_projection", dim * 6, dim, 0.6);
    lin(&mut entries, "head.head", cfg.out_dim as i32, dim, 0.7);
    entries.push((
        "head.modulation".into(),
        bf16((0..2 * dim).map(|i| i as f32 * 0.001), &[1, 2, dim]),
    ));
    // Block 0.
    entries.push((
        "blocks.0.modulation".into(),
        bf16((0..6 * dim).map(|i| i as f32 * 0.001), &[1, 6, dim]),
    ));
    for attn in ["self_attn", "cross_attn"] {
        for (p, seed) in [("q", 1.1), ("k", 1.2), ("v", 1.3), ("o", 1.4)] {
            lin(
                &mut entries,
                &format!("blocks.0.{attn}.{p}"),
                dim,
                dim,
                seed,
            );
        }
        entries.push((
            format!("blocks.0.{attn}.norm_q.weight"),
            bf16((0..dim).map(|_| 1.0), &[dim]),
        ));
        entries.push((
            format!("blocks.0.{attn}.norm_k.weight"),
            bf16((0..dim).map(|_| 1.0), &[dim]),
        ));
    }
    entries.push((
        "blocks.0.norm3.weight".into(),
        bf16((0..dim).map(|_| 1.0), &[dim]),
    ));
    entries.push((
        "blocks.0.norm3.bias".into(),
        bf16((0..dim).map(|_| 0.0), &[dim]),
    ));
    lin(&mut entries, "blocks.0.ffn.fc1", ffn, dim, 1.5);
    lin(&mut entries, "blocks.0.ffn.fc2", dim, ffn, 1.6);

    let path = tmp_dir(tmp).join("tiny_wan.safetensors");
    let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &path).unwrap();
    let w = Weights::from_file(&path).unwrap();
    WanTransformer::from_weights(&w, cfg).unwrap()
}

use crate::transformer::WAN_BLOCK_NORM_DIFF_PATCH_TARGETS;

/// sc-15326 — the norm diff-patch surface routes each `(module, part)` to the **field the forward
/// actually reads**.
///
/// A step-distill LoRA patches `norm3` on both halves (`norm3.diff` + `norm3.diff_b`), so an
/// aliased or swapped routing would fold the bias delta onto the LayerNorm *gain* — a wrong render
/// on all 40 blocks that still reports a clean, fully-applied install.
///
/// **Written through the accessor, read back off the fields.** Stamping a marker through
/// `norm_param_mut` and reading it back through the same accessor is non-discriminating: any
/// consistent bijection over the six targets passes, including swapping `norm3`'s `Weight` and
/// `Bias` arms (the γ and β of `layer_norm(x, γ, β, eps)`) or `norm_q`↔`norm_k`. So the read-back
/// side uses `WanTransformer::block_norm_diff_patch_fields`, which names `norm_q` / `norm_k` /
/// `norm3_w` / `norm3_b` **directly** and shares no code with the routing match arms.
#[test]
fn wan_norm_diff_patch_targets_route_to_distinct_parameters() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_cfg();
    let mut dit = tiny_transformer(&tmp, &cfg);
    assert_eq!(
        WAN_BLOCK_NORM_DIFF_PATCH_TARGETS.len(),
        6,
        "4 qk-RMSNorm gains + norm3 weight + norm3 bias"
    );

    // Stamp a unique constant into every target.
    for (i, (suffix, part)) in WAN_BLOCK_NORM_DIFF_PATCH_TARGETS.iter().enumerate() {
        let dotted = format!("blocks.0.{suffix}");
        let segs: Vec<&str> = dotted.split('.').collect();
        let p = dit
            .norm_param_mut(&segs, *part)
            .unwrap_or_else(|| panic!("`{dotted}` / {part:?} is advertised but does not route"));
        let marker = 100.0 + i as f32;
        *p = Array::full::<f32>(p.shape(), &Array::from_f32(marker)).unwrap();
    }

    // Read them back OFF THE FIELDS, in the advertised order. Target `i` must have landed on
    // field `i` — the parameter the block's forward passes to `rms_norm` / `layer_norm`. An
    // aliased, swapped, or wrong-part arm fails here even though it round-trips through the
    // accessor cleanly.
    let fields = dit.block_norm_diff_patch_fields(0);
    for (i, (suffix, part)) in WAN_BLOCK_NORM_DIFF_PATCH_TARGETS.iter().enumerate() {
        let got = mlx_rs::ops::max(fields[i].as_dtype(mlx_rs::Dtype::Float32).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert_eq!(
            got,
            100.0 + i as f32,
            "`blocks.0.{suffix}` / {part:?} did not land on the field the forward reads \
                 (aliased, swapped, or routed to the wrong part)"
        );
    }

    // The qk-RMSNorms are weight-only: a `.diff_b` on one has nowhere to land and must NOT be
    // quietly absorbed by the gain.
    for suffix in ["self_attn.norm_q", "cross_attn.norm_k"] {
        let dotted = format!("blocks.0.{suffix}");
        let segs: Vec<&str> = dotted.split('.').collect();
        assert!(
            dit.norm_param_mut(&segs, mlx_gen::adapters::DiffPatchPart::Bias)
                .is_none(),
            "`{dotted}` is an RMSNorm and has no bias channel"
        );
    }
    // An out-of-surface stem routes nowhere (reported unmatched by the fold, never dropped).
    assert!(dit
        .norm_param_mut(
            &["blocks", "0", "cross_attn", "norm_k_img"],
            mlx_gen::adapters::DiffPatchPart::Weight
        )
        .is_none());
}

/// A PEFT LoRA file targeting `blocks.0.self_attn.q` ([dim,dim]) — `diffusion_model.`-prefixed,
/// A `[rank,dim]`, B `[dim,rank]`, no alpha.
fn tiny_lora(tmp: &tempfile::TempDir, name: &str, dim: i32, rank: i32) -> PathBuf {
    let a = bf16(
        (0..rank * dim).map(|i| (i as f32 * 0.01).sin() * 0.03),
        &[rank, dim],
    );
    let b = bf16(
        (0..dim * rank).map(|i| (i as f32 * 0.007).cos() * 0.03),
        &[dim, rank],
    );
    let path = tmp_dir(tmp).join(name);
    Array::save_safetensors(
        vec![
            ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", &a),
            ("diffusion_model.blocks.0.self_attn.q.lora_B.weight", &b),
        ],
        None,
        &path,
    )
    .unwrap();
    path
}

/// A peft LoKr file (networkType=lokr) targeting `blocks.0.self_attn.q` ([64,64] = kron([8,8],[8,8])).
fn tiny_lokr(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
    let w1 = Array::from_slice(
        &(0..64)
            .map(|i| (i as f32 * 0.03).sin() * 0.1)
            .collect::<Vec<_>>(),
        &[8, 8],
    );
    let w2 = Array::from_slice(
        &(0..64)
            .map(|i| (i as f32 * 0.05).cos() * 0.1)
            .collect::<Vec<_>>(),
        &[8, 8],
    );
    let meta = HashMap::from([
        ("networkType".to_string(), "lokr".to_string()),
        ("alpha".to_string(), "8".to_string()),
        ("rank".to_string(), "8".to_string()),
    ]);
    let path = tmp_dir(tmp).join(name);
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.lokr_w1", &w1),
            ("blocks.0.self_attn.q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    path
}

fn lora_spec(path: PathBuf) -> AdapterSpec {
    AdapterSpec {
        path,
        scale: 0.8,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }
}

/// Probe the built transformer's `blocks.0.self_attn.q` linear's quant state, for the "stays
/// packed" assertions (the site methods must never dequantize the base).
fn q_is_quantized(dit: &mut WanTransformer) -> bool {
    dit.adaptable_mut(&["blocks", "0", "self_attn", "q"])
        .unwrap()
        .is_quantized()
}

/// Construct a bare pre-quantized `Wan` (5B) with a config that reports pre-quantized, so the
/// generate path would route to the additive install. `bits` mirrors a Q4/Q8 snapshot.
fn wan_5b_prequant(tmp: &tempfile::TempDir, adapters: Vec<AdapterSpec>) -> Wan {
    let mut cfg = tiny_cfg();
    cfg.quantization = Some(crate::config::WanQuant {
        bits: 8,
        group_size: 64,
    });
    Wan {
        descriptor: descriptor(),
        config: cfg,
        root: tmp_dir(tmp),
        adapters,
        quant: None,
        offload_policy: OffloadPolicy::Resident,
        memory_strategy: None,
        memory_tier: None,
        i2v_memory: None,
    }
}

fn wan_14b_prequant(tmp: &tempfile::TempDir, adapters: Vec<AdapterSpec>) -> Wan14b {
    let mut cfg = tiny_cfg();
    cfg.dual_model = true;
    cfg.quantization = Some(crate::config::WanQuant {
        bits: 8,
        group_size: 64,
    });
    Wan14b {
        descriptor: descriptor_t2v_14b(),
        config: cfg,
        root: tmp_dir(tmp),
        adapters,
        quant: None,
        offload_policy: OffloadPolicy::Resident,
        i2v_memory: None,
    }
}

#[test]
fn site_5b_packed_lora_installs_additively_no_error() {
    let tmp = tempfile::tempdir().unwrap();
    // 5B `Wan::install_adapters_additive`: a plain LoRA installs on a PACKED base with no error;
    // the base stays packed and the q forward-linear gains a residual.
    let cfg = tiny_cfg();
    let mut dit = tiny_transformer(&tmp, &cfg);
    dit.quantize(8, None).unwrap();
    assert!(q_is_quantized(&mut dit), "base packed before install");

    let model = wan_5b_prequant(
        &tmp,
        vec![lora_spec(tiny_lora(
            &tmp,
            "site5b.safetensors",
            cfg.dim as i32,
            8,
        ))],
    );
    model
        .install_adapters_additive(&mut dit)
        .expect("plain LoRA on a packed 5B must install with no error");
    assert!(
        q_is_quantized(&mut dit),
        "base must STAY packed after install (no dequant)"
    );
}

#[test]
fn site_5b_packed_lokr_installs_structurally_no_error() {
    let tmp = tempfile::tempdir().unwrap();
    // 5B (sc-10050): LoKr on a packed base now installs via the structured deferred-Kronecker path
    // with NO error (the sc-10045 interim rejection is gone), and the base STAYS packed.
    let cfg = tiny_cfg();
    let mut dit = tiny_transformer(&tmp, &cfg);
    dit.quantize(8, None).unwrap();
    assert!(q_is_quantized(&mut dit), "base packed before install");

    let mut spec = lora_spec(tiny_lokr(&tmp, "site5b_lokr.safetensors"));
    spec.kind = AdapterKind::Lokr;
    let model = wan_5b_prequant(&tmp, vec![spec]);
    model
        .install_adapters_additive(&mut dit)
        .expect("LoKr on a packed 5B must install structurally with no error (sc-10050)");
    assert!(
        q_is_quantized(&mut dit),
        "base must STAY packed after structured LoKr install (no dequant)"
    );
}

#[test]
fn site_14b_packed_lora_installs_on_both_experts() {
    let tmp = tempfile::tempdir().unwrap();
    // A14B `Wan14b::install_adapters_additive`: a shared plain LoRA installs onto BOTH packed
    // experts with no error; both bases stay packed.
    let cfg = tiny_cfg();
    let mut low = tiny_transformer(&tmp, &cfg);
    let mut high = tiny_transformer(&tmp, &cfg);
    low.quantize(4, None).unwrap();
    high.quantize(4, None).unwrap();

    let model = wan_14b_prequant(
        &tmp,
        vec![lora_spec(tiny_lora(
            &tmp,
            "site14b.safetensors",
            cfg.dim as i32,
            8,
        ))],
    );
    model
        .install_adapters_additive(&mut low, &mut high)
        .expect("plain LoRA on packed A14B experts must install with no error");
    assert!(
        q_is_quantized(&mut low) && q_is_quantized(&mut high),
        "both experts stay packed"
    );
}

#[test]
fn site_14b_packed_lokr_installs_structurally_no_error() {
    let tmp = tempfile::tempdir().unwrap();
    // A14B (sc-10050): a shared LoKr installs onto BOTH packed experts via the structured
    // deferred-Kronecker path with NO error; both bases stay packed.
    let cfg = tiny_cfg();
    let mut low = tiny_transformer(&tmp, &cfg);
    let mut high = tiny_transformer(&tmp, &cfg);
    low.quantize(4, None).unwrap();
    high.quantize(4, None).unwrap();

    let mut spec = lora_spec(tiny_lokr(&tmp, "site14b_lokr.safetensors"));
    spec.kind = AdapterKind::Lokr;
    let model = wan_14b_prequant(&tmp, vec![spec]);
    model
        .install_adapters_additive(&mut low, &mut high)
        .expect("LoKr on packed A14B experts must install structurally with no error (sc-10050)");
    assert!(
        q_is_quantized(&mut low) && q_is_quantized(&mut high),
        "both experts stay packed after structured LoKr install"
    );
}

#[test]
fn site_5b_packed_loha_is_explicit_typed_error() {
    let tmp = tempfile::tempdir().unwrap();
    // 5B: LoHa on a packed base is still rejected (LoHa is sc-10051, out of scope) — the actionable
    // error points at the bf16 tier + sc-10051, not a panic, not the old "not yet wired".
    let cfg = tiny_cfg();
    let mut dit = tiny_transformer(&tmp, &cfg);
    dit.quantize(8, None).unwrap();

    // A committed third-party LoHa fixture (detected by `hada_*` keys).
    let loha = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests/fixtures/sc3643_loha/linear.safetensors");
    let model = wan_5b_prequant(&tmp, vec![lora_spec(loha)]);
    let err = model
        .install_adapters_additive(&mut dit)
        .expect_err("LoHa on a packed 5B must be rejected (sc-10051)");
    let msg = err.to_string();
    assert!(
        msg.contains("bf16") && msg.contains("sc-10051") && msg.contains("LoHa"),
        "actionable LoHa-deferral msg, got: {msg}"
    );
    assert!(
        !msg.contains("not yet wired"),
        "not the old generic message, got: {msg}"
    );
}

#[test]
fn site_14b_dense_fold_still_works() {
    let tmp = tempfile::tempdir().unwrap();
    // Dense-path regression: `Wan14b::merge_adapters` (the fold path) still folds a plain LoRA onto
    // both dense expert weight maps, unchanged by sc-10045.
    let cfg = tiny_cfg();
    let lora = tiny_lora(&tmp, "site14b_dense.safetensors", cfg.dim as i32, 8);
    // Build the two dense expert weight maps from the tiny synthetic base.
    let base_path = {
        let mut c = cfg.clone();
        c.dual_model = true;
        let _ = tiny_transformer(&tmp, &c); // side-effect: writes tiny_wan.safetensors
        tmp_dir(&tmp).join("tiny_wan.safetensors")
    };
    let mut low_w = Weights::from_file(&base_path).unwrap();
    let mut high_w = Weights::from_file(&base_path).unwrap();
    let before = low_w
        .require("blocks.0.self_attn.q.weight")
        .unwrap()
        .clone();

    let model = wan_14b_prequant(&tmp, vec![lora_spec(lora)]);
    // Use a DENSE config for the fold path (quantization None).
    let dense = Wan14b {
        config: {
            let mut c = model.config.clone();
            c.quantization = None;
            c
        },
        ..model
    };
    dense
        .merge_adapters(&mut low_w, &mut high_w)
        .expect("dense fold must apply");
    let after = low_w.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        !array_eq(after, &before, false).unwrap().item::<bool>(),
        "dense fold must change the q weight"
    );
}
