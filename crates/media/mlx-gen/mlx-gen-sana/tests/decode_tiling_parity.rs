//! Real-weight parity gate for the **tiled** DC-AE decode (E5, iOS memory work).
//!
//! Tiling is a memory lever, and a memory lever that changes pixels is a bug wearing a good number.
//! At 1024² it takes SANA's sequential peak from 9177 MiB to 3465 MiB (`docs/ios-epics.md`, E5) —
//! which is only worth having if the image is the same image.
//!
//! What this actually guards is the **layout reconciliation** in `pipeline::decode_tiled`. The shared
//! [`mlx_gen::vae_tiling::tiled_decode`] is 5-D and slices the latent and shapes the decoded tile
//! through one `[t, h, w]` axis triple, but SANA's latent is NCHW and DC-AE emits NHWC — so the two
//! are bridged through a channels-last NTHWC lift. A wrong axis there does not error: it silently
//! decodes transposed or spatially scrambled tiles and blends them into a plausible-looking mess.
//! The first version of this code did exactly that (it passed the 4-D latent straight in, and only
//! failed because a *channel* count mismatched). A number-only check would not have caught the
//! variants that do not raise.
//!
//! Run:
//!   SANA_DCAE_WEIGHTS=/path/vae/diffusion_pytorch_model.safetensors \
//!   cargo test -p mlx-gen-sana --test decode_tiling_parity -- --ignored --nocapture

use mlx_rs::random::normal;

use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen_sana::{pipeline, DcAeConfig, DcAeDecoder};

/// Ceiling on the mean tiled-vs-whole pixel difference, set from the measured sweep rather than
/// picked. See the module docs for why this is a wide band and not a parity bound.
///
/// Measured (random-normal latent, 1024²): 1.9-4.3. Measured on a real render
/// (`mlx-gen-ios-catalog`'s `tiling_fidelity`): 2.4 at 512 px tiles rising to 6.2 at 128 px. A
/// *layout* error moves pixels wholesale and lands an order of magnitude above that, so 12.0
/// separates the two cleanly while admitting every legitimate tiling.
const MAX_MEAN_DIFF: f64 = 12.0;

/// Decode one latent whole-image and tiled, and check the difference is the *kind* tiling causes.
///
/// # This is not a parity gate, and cannot be
///
/// DC-AE's decoder is `EfficientViTBlock` = `SanaMultiscaleLinearAttention → GLUMBConv`, and that
/// attention normalizes by `1/(Σ + eps)` **summed over every spatial position** it is given. A tile
/// sees only its own pixels, so its normalizer differs from the whole image's — which is a global
/// operation, and therefore something *no* amount of overlap repairs. The measured sweep says so
/// directly: doubling the overlap at 512 px moved the mean 2.41 → 1.89, while halving the tile made
/// it worse (4.29 at 256 px), the opposite of how a boundary artifact behaves.
///
/// So tiled SANA output is a slightly different render, not the same one. Looked at
/// (`mlx-gen-ios-catalog`'s `tiling_fidelity` writes both PNGs) it is seam-free and equally valid —
/// the tone shifts smoothly because the trapezoidal blend spreads the per-tile difference out. That
/// is a trade worth making on a phone, where the alternative at 1024² is not rendering at all.
///
/// What this test *does* gate is the layout bridge, where a mistake is silent and fatal.
#[test]
#[ignore = "needs dc-ae-f32c32-sana-1.0 safetensors (~1.25 GB); set SANA_DCAE_WEIGHTS"]
fn tiled_decode_matches_whole_image_decode() {
    let path = std::env::var("SANA_DCAE_WEIGHTS").expect("set SANA_DCAE_WEIGHTS");
    let weights = Weights::from_file(&path).expect("load weights");
    let cfg = DcAeConfig::sana_f32c32();
    let decoder = DcAeDecoder::from_weights(&weights, cfg.clone()).expect("build");

    // [1, 32, 32, 32] → a 1024² image. Two tile edges: 512 splits into 2×2, 256 into 4×4, so both
    // the "few large tiles" and "many small tiles" plans are exercised, and 4×4 has interior tiles
    // that overlap on all four sides (the case a corner-only bug survives).
    let key = mlx_rs::random::key(0).unwrap();
    let latent = normal::<f32>(&[1, 32, 32, 32], None, None, &key).expect("latent");

    let whole = pipeline::decode_to_image(&decoder, &cfg, &latent, &Default::default(), None)
        .expect("whole-image decode");
    assert_eq!((whole.width, whole.height), (1024, 1024));

    // (tile edge, overlap) in OUTPUT pixels; `plan` divides both by DC-AE's ×32 spatial scale, so an
    // overlap below 32 px is zero latent cells and no overlap at all.
    let cases: &[(i32, i32)] = &[
        (1024, 256), // one tile: no blending, so this isolates the layout bridge from tiling artifacts
        (512, 128),
        (512, 256),
        (256, 64),
        (256, 128),
    ];
    let mut results: Vec<(i32, i32, u8, f64)> = Vec::new();
    for &(tile_px, overlap_px) in cases {
        let tiling = TilingConfig::spatial_only(tile_px, overlap_px);
        let tiled =
            pipeline::decode_to_image(&decoder, &cfg, &latent, &Default::default(), Some(&tiling))
                .expect("tiled decode");

        assert_eq!(
            (tiled.width, tiled.height),
            (whole.width, whole.height),
            "{tile_px}/{overlap_px}: dimensions changed"
        );
        assert_eq!(
            tiled.pixels.len(),
            whole.pixels.len(),
            "{tile_px}/{overlap_px}: pixel buffer length changed"
        );

        let (mut max_diff, mut sum_diff) = (0u8, 0u64);
        for (a, b) in whole.pixels.iter().zip(&tiled.pixels) {
            let d = a.abs_diff(*b);
            max_diff = max_diff.max(d);
            sum_diff += d as u64;
        }
        let mean_diff = sum_diff as f64 / whole.pixels.len() as f64;
        println!(
            "tile {tile_px:>5}px overlap {overlap_px:>4}px: max |Δ| = {max_diff:>3}, \
             mean |Δ| = {mean_diff:.4}"
        );
        results.push((tile_px, overlap_px, max_diff, mean_diff));
    }

    // Assert after the whole sweep, so one bad cell still reports every other cell's number — the
    // point of this test is to *choose* the production overlap, which needs the full table.
    let (_, _, single_tile_max, _) = results[0];
    assert_eq!(
        single_tile_max, 0,
        "a single tile covering the whole image must reproduce the whole-image decode exactly; a \
         non-zero difference here is the NCHW↔NTHWC layout bridge, not a tiling artifact"
    );
    for &(tile_px, overlap_px, max_diff, mean_diff) in &results {
        assert!(
            mean_diff < MAX_MEAN_DIFF,
            "{tile_px}/{overlap_px}: mean pixel difference {mean_diff:.4} exceeds \
             {MAX_MEAN_DIFF} — that is the scale of a layout error, not of DC-AE's per-tile \
             attention scope"
        );
        // `max_diff` is deliberately NOT asserted. It reaches ~226/255 on a perfectly good render:
        // a handful of pixels on the highest-contrast edges shift, and one outlier says nothing
        // about whether the image is right. It is printed because the trend across the sweep is
        // informative, and it is not a gate because it does not discriminate.
        let _ = max_diff;
    }
}
