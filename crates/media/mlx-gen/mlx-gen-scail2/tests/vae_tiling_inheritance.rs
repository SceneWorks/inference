//! sc-19753 — SCAIL-2 **inherits** the normalization-correct bounded VAE decode from Wan.
//!
//! SCAIL-2 owns no VAE: [`mlx_gen_scail2::ProviderVae`] is a type alias for
//! [`mlx_gen_wan::WanVae`], and `generate.rs`'s decode phase calls `vae.decode_tiled(..)` on it.
//! sc-19753 split that decoder head/tail — `Decoder3d::forward_middle` (which carries the per-frame
//! spatial softmax self-attention) runs **once** on the full latent, and only
//! `forward_upsample_tail` is tiled — so SCAIL-2 is covered *by inheritance*, a claim only worth what
//! a test proves.
//!
//! This is an **executed decode**, not a type assertion: the committed weights-free tiny z16 VAE
//! (`mlx-gen-wan/tests/fixtures/s2_tiling.safetensors`, `dim=4`, real architecture) is loaded through
//! `ProviderVae::from_weights` and decoded through `ProviderVae::decode_tiled`. The compile-time
//! binding below additionally fails the build if SCAIL-2 ever grows its own VAE type.
//!
//! **Why the claim is relative.** Tiling is not identical to a one-shot decode here for a reason that
//! is not this story's: each tile's causal conv sees zero-pad at its boundary instead of neighbour
//! data, and on a tiny *random*-weight VAE (no learned smoothness) that seam residual dominates any
//! normalization term. So — exactly as `mlx-gen-wan/tests/tiling_parity.rs` does — the proof is that
//! the shipped bounded decode is materially **closer to a dense decode** than the pre-sc-19753 route
//! (whole decoder per tile) is. A tolerance would be measuring the conv seam, not the fix.

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_gen_scail2::{ProviderVae, VAE_TILING};
use mlx_gen_wan::WanVae;
use mlx_rs::Array;

/// **Compile-time inheritance proof.** SCAIL-2's provider VAE constructor must be
/// [`mlx_gen_wan::WanVae`]'s *by identity*. If SCAIL-2 ever re-points `ProviderVae` at a decoder of
/// its own — the only way it could miss a fix landed in Wan's — this binding stops compiling.
const _: fn(&Weights) -> Result<WanVae> = ProviderVae::from_weights;

/// The committed tiny z16 VAE + the reference's tiled IO (`mlx-gen-wan/tools/dump_s2_tiling_fixtures.py`).
/// Referenced across the crate boundary because SCAIL-2 has no VAE fixture of its own — it has no
/// VAE. `mlx-gen-wan` is a first-class dependency of this crate, so the same `CARGO_MANIFEST_DIR`-
/// relative form already used for the `tools/golden` tree reaches it.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mlx-gen-wan/tests/fixtures/s2_tiling.safetensors"
);

/// The dump's tiling config (`s2_tiling.json`): spatial 64 px / 32 overlap, temporal 16 f / 8.
fn bounded_cfg() -> TilingConfig {
    TilingConfig {
        spatial: Some(SpatialTiling {
            tile_px: 64,
            overlap_px: 32,
        }),
        temporal: Some(TemporalTiling {
            tile_frames: 16,
            overlap_frames: 8,
        }),
    }
}

/// The tiny z16 VAE loaded **through SCAIL-2's own provider alias**, plus the golden latent
/// `[1, 16, 6, 12, 12]`.
fn fixture_vae() -> (ProviderVae, Array) {
    let w = Weights::from_file(FIXTURE).unwrap_or_else(|e| {
        panic!("read {FIXTURE}: {e} (run mlx-gen-wan/tools/dump_s2_tiling_fixtures.py)")
    });
    let vae = ProviderVae::from_weights(&w).expect("build the tiny z16 provider VAE");
    let latent = w.require("tiled_in").expect("tiled_in").clone();
    (vae, latent)
}

/// Mean relative deviation against `reference` — the metric `mlx-gen-wan/tests/tiling_parity.rs`
/// reports, so the two gates are directly comparable.
fn mean_rel(got: &Array, reference: &Array) -> f64 {
    got.eval().unwrap();
    reference.eval().unwrap();
    let (mut sum_abs, mut sum_ref) = (0.0_f64, 0.0_f64);
    for (g, e) in got
        .as_slice::<f32>()
        .iter()
        .zip(reference.as_slice::<f32>())
    {
        sum_abs += (g - e).abs() as f64;
        sum_ref += e.abs() as f64;
    }
    sum_abs / sum_ref.max(1e-9)
}

/// The **pre-sc-19753 route**, reconstructed from public API alone: run the *whole* decoder per tile,
/// so every spatial tile's `middle.1` softmax attends over its own crop's token set. Denormalization
/// is a per-channel affine, so doing it inside `WanVae::decode` per tile is algebraically the same as
/// the reference's hoisted denormalize.
fn per_tile_whole_decoder(vae: &ProviderVae, latent: &Array, cfg: &TilingConfig) -> Array {
    let sh = latent.shape();
    let plan = cfg.plan(VaeTiling::WAN, sh[2], sh[3], sh[4]);
    mlx_gen::vae_tiling::tiled_decode(latent, &plan, [2, 3, 4], None, |tile| vae.decode(tile))
        .expect("per-tile whole-decoder decode")
}

/// SCAIL-2's provider geometry must be Wan's z16 geometry, and must resolve for its registered id.
/// A crate that grew its own VAE would have to restate this — and would then be free to restate it
/// wrongly, which is what makes this worth pinning next to the executed decode.
#[test]
fn scail2_provider_geometry_is_the_wan_z16_geometry() {
    assert_eq!(VAE_TILING, WanVae::VAE_TILING);
    assert_eq!(VAE_TILING, VaeTiling::WAN);
    assert_eq!(
        mlx_gen_scail2::vae_tiling(mlx_gen_scail2::pipeline::MODEL_ID),
        Some(VaeTiling::WAN)
    );
    assert_eq!(mlx_gen_scail2::vae_tiling("not_a_scail2_id"), None);
}

/// The inheritance proof, executed. SCAIL-2's bounded decode must track a dense decode materially
/// better than the route sc-19753 replaced. Measured on this fixture: bounded `mean_rel` 1.810e-1 vs
/// per-tile-whole-decoder 3.407e-1 — a 47 % reduction. If SCAIL-2 ever stopped routing to the fixed
/// `WanVae::decode_tiled` (or that head/tail split regressed to whole-tail tiling), the two numbers
/// converge and this fails.
#[test]
fn scail2_bounded_decode_is_closer_to_dense_than_per_tile_normalization() {
    let (vae, latent) = fixture_vae();
    let cfg = bounded_cfg();
    let sh = latent.shape();
    assert!(
        cfg.needs_tiling(VaeTiling::WAN, sh[2], sh[3], sh[4]),
        "the golden latent must actually tile under SCAIL-2's geometry"
    );

    let dense = vae.decode(&latent).expect("single-pass decode");
    let bounded = vae
        .decode_tiled(&latent, &cfg, Some(&CancelFlag::new()))
        .expect("bounded decode");
    let per_tile = per_tile_whole_decoder(&vae, &latent, &cfg);

    assert_eq!(bounded.shape(), dense.shape());
    let bounded_rel = mean_rel(&bounded, &dense);
    let per_tile_rel = mean_rel(&per_tile, &dense);
    println!(
        "[scail2 inheritance] bounded mean_rel={bounded_rel:.3e} \
         per-tile-whole-decoder mean_rel={per_tile_rel:.3e}"
    );
    assert!(
        bounded_rel < per_tile_rel * 0.75,
        "SCAIL-2's bounded decode is not keeping the middle-block attention whole: \
         bounded={bounded_rel:.3e} vs per-tile={per_tile_rel:.3e}"
    );
}

/// The tiling control. A request too wide to tile must be the *exact* single-pass decode — the
/// mutation control that separates "bounded and correct" from "never bounded", which the relative
/// claim above cannot see on its own.
#[test]
fn a_request_too_wide_to_tile_is_the_exact_single_pass_decode() {
    let (vae, latent) = fixture_vae();
    let sh = latent.shape();
    let wide = TilingConfig::spatial_only(4096, 64);
    assert!(!wide.needs_tiling(VaeTiling::WAN, sh[2], sh[3], sh[4]));

    let dense = vae.decode(&latent).expect("single-pass decode");
    let fallthrough = vae.decode_tiled(&latent, &wide, None).expect("fallback");
    dense.eval().unwrap();
    fallthrough.eval().unwrap();
    assert_eq!(
        dense.as_slice::<f32>(),
        fallthrough.as_slice::<f32>(),
        "a request too wide to tile must be the exact single-pass decode"
    );
}

/// A pre-tripped cancel must be observed before any tensor work on the SCAIL-2 seam too — the decode
/// is a dominant fraction of a SCAIL-2 render's wall clock.
#[test]
fn scail2_bounded_decode_honors_a_pretripped_cancel() {
    let (vae, latent) = fixture_vae();
    let cancel = CancelFlag::new();
    cancel.cancel();
    for cfg in [bounded_cfg(), TilingConfig::spatial_only(4096, 64)] {
        assert!(matches!(
            vae.decode_tiled(&latent, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }
}
