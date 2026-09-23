//! RGBA VAE parity vs the frozen `AutoencoderKLQwenImage21` on the committed miniature snapshot
//! (`tools/dump_qwen21_vae.py`): every encoder stage, the moments and posterior mode, every
//! decoder stage and the decoded RGBA image, for a square and a non-square geometry.
//!
//! Tolerances: **1e-2 × peak** on every stage, set from the measured drift with ≥ 5× headroom:
//! encoder ≤ **1.4e-3 × peak** (the 3×3 stages agree to ~1e-6; the drift enters at the first
//! width-changing stage's 1×1 `conv_shortcut`, which MLX dispatches as a reduced-precision f32
//! GEMM — a bare 16×8·8×8 matmul reproduces 1.2e-3 while an f64 host reference matches the fixture
//! to 1e-7), decoder ≤ **1.9e-3 × peak**, decoded RGBA image ≤ **9e-4** absolute. Every stage
//! prints its own numbers. The tiny VAE is deliberately non-uniform (`dim_mult [1, 1, 2, 4, 4]`,
//! decoder width 6 vs encoder 4) so the `DupUp` channel-duplication index and the residual
//! `conv_shortcut` are live: dropping `DupUp`'s `first_chunk` slot or transposing its row/column
//! offsets each turn this suite RED (measured max|Δ| 49 and 128 on `up_block_2` / `up_block_3`).

use mlx_gen_qwen_image_2_1::load_vae;
use mlx_rs::ops::indexing::IndexOp;

use crate::common::{assert_close, fixture, tiny_snapshot};

const ENCODER_TOL: f32 = 1e-2;
const DECODER_TOL: f32 = 1e-2;

#[test]
fn every_stage_of_encode_and_decode_matches_upstream() {
    let w = fixture("qwen21_vae.safetensors");
    let vae = load_vae(&tiny_snapshot()).unwrap();
    let want = |key: &str| {
        // Fixtures are 5-D `[B, C, T=1, H, W]`; the port is single-frame NCHW.
        w.require(key)
            .unwrap_or_else(|_| panic!("fixture lacks {key}"))
            .squeeze_axes(&[2])
            .unwrap()
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
        assert_eq!(decoded.shape()[1], 4, "four RGBA channels");
        assert_close(
            &format!("{case}/decoded"),
            &decoded,
            &want(&format!("{case}/decoded")),
            DECODER_TOL,
        );

        // Composited RGB keeps three channels and the spatial size.
        let rgb = mlx_gen_qwen_image_2_1::rgba_to_rgb_over_white(&decoded).unwrap();
        assert_eq!(rgb.shape()[1], 3);
        assert_eq!(rgb.index((.., 0..1, .., ..)).shape()[2], decoded.shape()[2]);

        // Head + tail is the single pass, exactly.
        let split = vae.decode_tail(&vae.decode_head(&z).unwrap()).unwrap();
        assert_close(&format!("{case}/head_then_tail"), &split, &decoded, 0.0);
    }
}

/// The bounded decode tiles only the spatially local tail (the head with its global attention runs
/// once), so it must land on the single pass up to the conv-halo seam term the overlap attenuates.
///
/// Geometry: the `tall` fixture is a 6×4 latent (96×64 px); a 64 px tile forces overlapping row
/// tiles (the plan asserts it). Two overlaps are measured so the seam term's dependence on overlap
/// is visible: 32 px (2 latent rows — narrower than the tail's receptive field) and 48 px
/// (3 latent rows). The tail's five up-stages of 3×3 convolutions reach ~4 latent rows past a tile
/// edge, so an overlap below that leaves a seam term the trapezoidal blend attenuates rather than
/// eliminates — it is real, it is the same term the sibling VAEs document, and it must never be
/// asserted as zero.
///
/// Bounds (absolute, on the `[-1, 1]` image; this fixture's random 0.2-scaled weights drive the
/// tail's activations to ~1e2, which magnifies any halo far beyond a trained VAE's): the 48 px
/// overlap is held to **max 2e-1 / mean 1e-2**, the 32 px overlap to **max 3e-1 / mean 2e-2**.
/// Measured on this fixture: see the printed lines (the `MEASURED` comment below records them).
#[test]
fn tiled_decode_matches_the_single_pass_up_to_the_seam_term() {
    use mlx_gen::gen_core::tiling::VaeTiling;
    use mlx_gen::tiling::TilingConfig;
    use mlx_gen::{CancelFlag, LatentDecoder};

    let w = fixture("qwen21_vae.safetensors");
    let vae = load_vae(&tiny_snapshot()).unwrap();
    let z = w.require("tall/z").unwrap().squeeze_axes(&[2]).unwrap(); // [1, 8, 6, 4]
    let single = vae.decode_rgba(&z).unwrap();

    // MEASURED (sc-24108 fix pass, this fixture): overlap 48 px (3 row tiles) → max|Δ| 9.5e-2,
    // mean|Δ| 4.0e-3; overlap 32 px (2 row tiles) → max|Δ| 1.3e-1, mean|Δ| 4.1e-3. The bounds
    // below carry ≥ 2× headroom on max and ≥ 2.5× on mean over those measurements.
    for (overlap_px, max_bound, mean_bound) in [(48, 2e-1, 1e-2), (32, 3e-1, 2e-2)] {
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
        assert_eq!(tiled.shape(), single.shape());
        let (max_abs, _, mean) = crate::common::errors(&tiled, &single);
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

    // The trait seam composites the denormalised, tiled decode to RGB and honours a pre-tripped cancel.
    let cfg = TilingConfig::spatial_only(64, 48);
    let normalised = vae.normalize(&z).unwrap();
    let via_trait = LatentDecoder::decode_tiled(&vae, &normalised, &cfg, None).unwrap();
    assert_eq!(via_trait.shape()[1], 3);
    let cancel = CancelFlag::new();
    cancel.cancel();
    assert!(matches!(
        LatentDecoder::decode_tiled(&vae, &normalised, &cfg, Some(&cancel)),
        Err(mlx_gen::Error::Canceled)
    ));
}
