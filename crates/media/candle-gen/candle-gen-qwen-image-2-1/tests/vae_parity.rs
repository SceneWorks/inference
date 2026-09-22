//! RGBA VAE parity vs the frozen `AutoencoderKLQwenImage21` on the committed miniature snapshot
//! (`tools/dump_qwen21_vae.py`): every encoder stage, the moments and posterior mode, every
//! decoder stage and the decoded RGBA image, for a square and a non-square geometry.
//!
//! Same fixture and same oracle as `mlx-gen-qwen-image-2-1`'s own `tests/vae_parity.rs`, reached
//! across the backend boundary by relative path. The MLX twin budgets **1e-3 × peak** on the
//! encoder and **1e-2 × peak** on the decoder for Metal's reduced-precision f32 matmul; candle's
//! CPU f32 runs the same arithmetic torch did, so the bars here are the MEASURED maxima rounded up
//! (see the constants). Every stage prints its own numbers.

use candle_core::Device;
use candle_gen_qwen_image_2_1::{load_vae, rgba_to_rgb_over_white};

use crate::common::{assert_close, errors, tiny_snapshot, Fixture};

/// Encoder bar, held on the regenerated non-uniform fixture (`dim_mult [1, 1, 2, 4, 4]`, decoder
/// width 6 vs encoder 4 — so `DupUp`'s channel-duplication index and the residual `conv_shortcut`
/// are both live). The MLX twin had to widen to 1e-2 because MLX dispatches the width-changing
/// 1×1 `conv_shortcut` as a reduced-precision f32 GEMM; candle CPU f32 runs the same arithmetic
/// torch did, so this lane stays three orders tighter.
const ENCODER_TOL: f32 = 1e-5;
/// Decoder bar, same fixture and same reasoning (the MLX twin budgets 1e-2 for Metal's 16×
/// upsampling stack).
///
/// This bar is mutation-checked, not merely passing: dropping `DupUp`'s `first_chunk` temporal
/// slot (`(o·ft + (ft−1))` → `(o·ft)`) turns it RED at `square/decoder/up_block_2` with
/// `max|Δ| = 4.9e1`, and transposing its row/column offsets turns it RED at
/// `square/decoder/up_block_3` with `max|Δ| = 1.3e2`.
const DECODER_TOL: f32 = 1e-5;

#[test]
fn every_stage_of_encode_and_decode_matches_upstream() {
    let w = Fixture::open("qwen21_vae.safetensors");
    let vae = load_vae(&tiny_snapshot(), &Device::Cpu).unwrap();
    let want = |key: &str| {
        // Fixtures are 5-D `[B, C, T=1, H, W]`; the port is single-frame NCHW.
        w.tensor(key).squeeze(2).unwrap()
    };
    for case in ["square", "tall"] {
        let image = want(&format!("{case}/image"));
        let (moments, trace) = vae.encode_moments_traced(&image).unwrap();
        assert_eq!(trace.len(), 4 + vae.config().dim_mult.len());
        for (stage, got) in &trace {
            assert_close(
                &format!("{case}/{stage}"),
                got,
                &want(&format!("{case}/trace/{stage}")),
                ENCODER_TOL,
            );
        }
        assert_close(
            &format!("{case}/moments"),
            &moments,
            &want(&format!("{case}/moments")),
            ENCODER_TOL,
        );
        let mode = vae.encode_mode(&image).unwrap();
        assert_close(
            &format!("{case}/mode"),
            &mode,
            &want(&format!("{case}/mode")),
            ENCODER_TOL,
        );

        let z = want(&format!("{case}/z"));
        let (decoded, trace) = vae.decode_rgba_traced(&z).unwrap();
        assert_eq!(trace.len(), 4 + vae.config().dim_mult.len());
        for (stage, got) in &trace {
            assert_close(
                &format!("{case}/{stage}"),
                got,
                &want(&format!("{case}/trace/{stage}")),
                DECODER_TOL,
            );
        }
        assert_eq!(decoded.dims()[1], 4, "four RGBA channels");
        assert_close(
            &format!("{case}/decoded"),
            &decoded,
            &want(&format!("{case}/decoded")),
            DECODER_TOL,
        );

        // Composited RGB keeps three channels and the spatial size.
        let rgb = rgba_to_rgb_over_white(&decoded).unwrap();
        assert_eq!(rgb.dims()[1], 3);
        assert_eq!(rgb.dims()[2], decoded.dims()[2]);
        assert_eq!(rgb.dims()[3], decoded.dims()[3]);

        // Head + tail is the single pass, exactly.
        let split = vae.decode_tail(&vae.decode_head(&z).unwrap()).unwrap();
        assert_close(&format!("{case}/head_then_tail"), &split, &decoded, 0.0);
    }
}

/// The bounded decode tiles only the spatially local tail (the head with its global attention runs
/// once), so it must land on the single pass up to the conv-halo seam term the overlap attenuates.
/// The candle twin of `mlx-gen-qwen-image-2-1`'s test of the same name, over the SAME
/// [`VaeTiling::QWEN_IMAGE_2_1`] geometry declared once in gen-core.
///
/// Geometry: the `tall` fixture is a 6×4 latent (96×64 px); a 64 px tile forces overlapping row
/// tiles (the plan asserts it). Two overlaps are measured so the seam term's dependence on overlap
/// is visible: 32 px (2 latent rows — narrower than the tail's receptive field) and 48 px
/// (3 latent rows). The tail's five up-stages of 3×3 convolutions reach ~4 latent rows past a tile
/// edge, so an overlap below that leaves a seam term the trapezoidal blend attenuates rather than
/// eliminates — it is real, it is the same term the sibling VAEs document, and it must never be
/// asserted as zero.
///
/// Bounds are absolute on the `[-1, 1]` image. This fixture's random 0.2-scaled weights drive the
/// tail's activations to ~1e2, which magnifies any halo far beyond a trained VAE's; the constants
/// below carry ≥ 2× headroom over the measured values printed by the test.
///
/// MEASURED (sc-24109, candle CPU f32, this fixture): overlap 48 px (3 row tiles) → `max|Δ|`
/// 9.526e-2, `mean|Δ|` 4.028e-3; overlap 32 px (2 row tiles) → `max|Δ|` 1.312e-1, `mean|Δ|`
/// 4.085e-3. The MLX twin measures 9.5e-2 / 4.0e-3 and 1.3e-1 / 4.1e-3 on the same fixture, so the
/// two engines' seam terms agree to the printed precision — the halo is the tiling geometry's, not
/// either backend's.
#[test]
fn tiled_decode_matches_the_single_pass_up_to_the_seam_term() {
    use candle_gen::gen_core::tiling::{TilingConfig, VaeTiling};
    use candle_gen::gen_core::CancelFlag;
    use candle_gen::{CandleError, LatentDecoder};

    let w = Fixture::open("qwen21_vae.safetensors");
    let vae = load_vae(&tiny_snapshot(), &Device::Cpu).unwrap();
    let z = w.tensor("tall/z").squeeze(2).unwrap(); // [1, z_dim, 6, 4]
    let single = vae.decode_rgba(&z).unwrap();

    for (overlap_px, max_bound, mean_bound) in [(48i32, 2e-1f32, 1e-2f32), (32, 3e-1, 2e-2)] {
        let cfg = TilingConfig::spatial_only(64, overlap_px);
        assert!(cfg.needs_tiling(VaeTiling::QWEN_IMAGE_2_1, 1, 6, 4));
        let plan = cfg.plan(VaeTiling::QWEN_IMAGE_2_1, 1, 6, 4);
        assert!(
            plan.h.len() >= 2,
            "the geometry must actually tile: {} row tiles",
            plan.h.len()
        );
        assert_eq!(plan.w.len(), 1);
        let tiled = vae.decode_rgba_tiled(&z, &cfg, None).unwrap();
        assert_eq!(tiled.dims(), single.dims());
        let (max_abs, _, mean) = errors(&tiled, &single);
        eprintln!(
            "tall/tiled_vs_single overlap={overlap_px}px rows={}: max|Δ|={max_abs:.3e} mean|Δ|={mean:.3e}",
            plan.h.len()
        );
        assert!(
            max_abs <= max_bound && mean <= mean_bound,
            "overlap {overlap_px}px: max {max_abs:.3e} (bound {max_bound:.0e}) mean {mean:.3e} (bound {mean_bound:.0e})"
        );
    }

    // Below the threshold the tiled entry point IS the single pass.
    let loose = TilingConfig::spatial_only(512, 64);
    assert!(!loose.needs_tiling(VaeTiling::QWEN_IMAGE_2_1, 1, 6, 4));
    let same = vae.decode_rgba_tiled(&z, &loose, None).unwrap();
    assert_close("tall/untiled_passthrough", &same, &single, 0.0);

    // The trait seam composites the denormalised, tiled decode to RGB and honours a pre-tripped
    // cancel with the TYPED variant (never a stringified Msg).
    let cfg = TilingConfig::spatial_only(64, 48);
    let normalised = vae.normalize(&z).unwrap();
    let via_trait = LatentDecoder::decode_tiled(&vae, &normalised, &cfg, None).unwrap();
    assert_eq!(via_trait.dims()[1], 3);
    let cancel = CancelFlag::new();
    cancel.cancel();
    assert!(matches!(
        LatentDecoder::decode_tiled(&vae, &normalised, &cfg, Some(&cancel)),
        Err(CandleError::Canceled)
    ));
}

/// `GenerationMemory::tile_vae_decode` is what selects the bounded decode at render time, and the
/// request's edge/overlap override the family defaults.
#[test]
fn a_request_selects_the_bounded_decode_and_its_geometry() {
    use candle_gen::gen_core::{GenerationMemory, GenerationRequest};
    use candle_gen_qwen_image_2_1::{decode_tiling, DECODE_OVERLAP, DECODE_TILE_EDGE};

    let base = GenerationRequest {
        prompt: "a red fox".into(),
        width: 2048,
        height: 2048,
        ..Default::default()
    };
    assert!(
        decode_tiling(&base).is_none(),
        "no memory options means the single pass"
    );

    let mut req = base.clone();
    req.memory = Some(GenerationMemory {
        tile_vae_decode: true,
        ..Default::default()
    });
    let cfg = decode_tiling(&req).expect("tile_vae_decode selects bounded decode");
    let spatial = cfg.spatial.expect("a spatial-only tiling");
    assert_eq!(
        (spatial.tile_px, spatial.overlap_px),
        (DECODE_TILE_EDGE as i32, DECODE_OVERLAP as i32)
    );
    assert!(cfg.temporal.is_none(), "a still image never tiles in time");

    req.memory = Some(GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: Some(256),
        decode_overlap: Some(32),
        ..Default::default()
    });
    let spatial = decode_tiling(&req).unwrap().spatial.expect("spatial");
    assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 32));
}
