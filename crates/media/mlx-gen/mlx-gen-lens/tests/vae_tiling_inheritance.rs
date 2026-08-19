//! sc-19753 — Lens **inherits** the layer-wise, normalization-correct bounded VAE decode.
//!
//! Lens does not own a VAE: [`mlx_gen_lens::vae::decode_with_tiling`] reshapes the DiT output into
//! the packed grid and hands it to [`mlx_gen_flux2::Flux2Vae::decode_packed_latents_tiled`]
//! (`mlx-gen-lens/src/vae.rs`). Nothing was re-implemented on the Lens side for sc-19753, so the
//! claim that Lens is covered is only worth what a test proves. These run the real Lens seam —
//! reshape → bn de-normalize → 2×2 unpatchify → post-quant → head → bounded tail — on the shared
//! weights-free FLUX.2 fixture, and assert the bounded result tracks the dense decode.
//!
//! Weights-free and unit-scale: the fixture is a structurally faithful FLUX.2 VAE (real block and
//! resnet counts) at a 32-channel width, and the latent is a 4×5 packed grid.

use mlx_gen_flux2::vae::tiling_fixture::{max_abs_delta, packed_latents, weights};
use mlx_gen_flux2::Flux2Vae;
use mlx_gen_lens::vae::{decode, decode_with_tiling};
use mlx_rs::Array;

const GRID_H: i32 = 4;
const GRID_W: i32 = 5;

/// `[1, h, w, 128]` packed grid → the `[1, h·w, 128]` DiT output layout Lens's decode consumes.
fn dit_output() -> Array {
    let packed = packed_latents(GRID_H, GRID_W);
    packed.reshape(&[1, GRID_H * GRID_W, 128]).unwrap()
}

fn vae() -> Flux2Vae {
    Flux2Vae::from_weights(&weights()).unwrap()
}

/// The Lens bounded decode must track the Lens dense decode. This is the inheritance proof: it
/// fails if `decode_with_tiling` stops routing to the fixed FLUX.2 tail (or if that tail regresses
/// to whole-tail tiling, which moves the same comparison by ~2e0 against this 3e-3 bound).
#[test]
fn lens_bounded_decode_tracks_the_dense_decode() {
    let vae = vae();
    let dit_out = dit_output();
    let dense = decode(&vae, &dit_out, GRID_H as usize, GRID_W as usize, None).unwrap();
    let bounded = decode_with_tiling(
        &vae,
        &dit_out,
        GRID_H as usize,
        GRID_W as usize,
        None,
        Some(&mlx_gen::tiling::TilingConfig::spatial_only(16, 0)),
        None,
    )
    .unwrap();
    dense.eval().unwrap();
    bounded.eval().unwrap();
    assert_eq!(bounded.shape(), dense.shape());
    let delta = max_abs_delta(&dense, &bounded);
    assert!(
        delta < 3e-3,
        "Lens bounded decode diverged from its dense decode: max|delta|={delta:.3e}"
    );
}

/// The tiling request must physically split this geometry, otherwise the assertion above would pass
/// against an untiled fall-through and prove nothing. A tile edge wide enough to hold the whole
/// output takes that fall-through and is therefore *exactly* equal — the mutation control that
/// distinguishes "bounded and correct" from "never bounded".
#[test]
fn the_bounded_request_actually_tiles_and_the_wide_request_does_not() {
    let vae = vae();
    let dit_out = dit_output();
    let dense = decode(&vae, &dit_out, GRID_H as usize, GRID_W as usize, None).unwrap();
    let untiled = decode_with_tiling(
        &vae,
        &dit_out,
        GRID_H as usize,
        GRID_W as usize,
        None,
        Some(&mlx_gen::tiling::TilingConfig::spatial_only(4096, 128)),
        None,
    )
    .unwrap();
    dense.eval().unwrap();
    untiled.eval().unwrap();
    assert_eq!(
        max_abs_delta(&dense, &untiled),
        0.0,
        "a request too wide to tile must be the exact single-pass decode"
    );

    // The narrow request genuinely partitions the same geometry.
    let plan = mlx_gen::tiling::TilingConfig::spatial_only(16, 0).plan(
        mlx_gen::tiling::VaeTiling {
            spatial_scale: 8,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: 128,
        },
        1,
        GRID_H * 2,
        GRID_W * 2,
    );
    assert!(
        plan.h.len() > 1 && plan.w.len() > 1,
        "the bounded request must split both spatial axes, got {}x{}",
        plan.h.len(),
        plan.w.len()
    );
}

/// A pre-tripped cancel must be observed before any tensor work, on the Lens seam too.
#[test]
fn lens_bounded_decode_honors_a_pretripped_cancel() {
    let vae = vae();
    let dit_out = dit_output();
    let cancel = mlx_gen::CancelFlag::new();
    cancel.cancel();
    let result = decode_with_tiling(
        &vae,
        &dit_out,
        GRID_H as usize,
        GRID_W as usize,
        None,
        Some(&mlx_gen::tiling::TilingConfig::spatial_only(16, 0)),
        Some(&cancel),
    );
    assert!(matches!(result, Err(mlx_gen::Error::Canceled)));
}
