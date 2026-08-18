//! sc-19753 — Chroma **inherits** the layer-wise, normalization-correct bounded VAE decode.
//!
//! Chroma owns no VAE. `loader::load_vae` delegates to `mlx_gen_flux::load_vae` (byte-identical
//! FLUX.1 16-ch AutoencoderKL layout) and returns [`mlx_gen_z_image::vae::Vae`]; `Chroma`'s
//! `decode_with_vae` decodes through `LatentDecoder::decode_tiled` on that exact type. sc-19753
//! converted that decoder (mid-block attention hoisted into a dense head; layer-wise
//! `GlobalGroupNorm` + `tiled_conv2d_3x3_nhwc` in the tail), so Chroma is covered *by inheritance* —
//! a claim that is only worth what a test proves.
//!
//! These are executed decodes, not a shape check: the weights-free z-image decoder fixture is driven
//! through the same seam body the production render takes (`mlx_gen_flux::unpack_latents` →
//! `ensure_decoder_compatible` → `LatentDecoder::decode_tiled`), and the bounded result must track the
//! dense decode. The compile-time bindings below fail the build if Chroma ever grows its own VAE type;
//! the per-crop control fails if the shared decoder ever goes back to per-tile normalization.

use std::path::Path;

use mlx_gen::gen_core::FLUX1_LATENT_SPACE;
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{ensure_decoder_compatible, CancelFlag, Error, LatentDecoder, Result};
use mlx_gen_flux::{pack_latents, unpack_latents};
use mlx_gen_z_image::vae::{Vae, VaeDecoderConfig};
use mlx_rs::Array;

/// **Compile-time inheritance proof.** Chroma's production VAE loader must return the shared
/// [`mlx_gen_z_image::vae::Vae`] *by identity*, and its dense-source quantization seam must accept
/// that same type. If Chroma ever grows its own decoder type — the only way it could miss a fix
/// landed in the shared one — these bindings stop compiling.
const _: fn(&Path) -> Result<Vae> = mlx_gen_chroma::loader::load_vae;
const _: fn(&mut Vae, i32) -> Result<()> = mlx_gen_chroma::loader::quantize_vae_for_dense_source;

/// The z-image tiny decoder fixture (`mlx-gen-z-image/tools/dump_z_image_vae_decoder.py`). Referenced
/// across the crate boundary because it is the only committed weights-free instance of the shared
/// decoder; the same `CARGO_MANIFEST_DIR`-relative form is already used for the `tools/golden` tree.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../mlx-gen-z-image/tests/fixtures/vae_decoder.safetensors"
);

/// The fixture's decoder shape (copied from `mlx-gen-z-image/tests/vae_decoder.rs`): conv_in → mid →
/// 2 up-blocks (only the first upsamples) → norm-out → SiLU → conv_out. Spatial ×2, not production's
/// ×8 — the architecture is width/depth-parametric and this keeps the fixture ~2 MB.
fn small_cfg() -> VaeDecoderConfig {
    VaeDecoderConfig {
        up_blocks: vec![(3, true), (3, false)],
    }
}

/// The tile edge the bounded assertions use. `Vae::decode_tiled` divides by
/// [`VaeTiling::QWEN_IMAGE`]'s ×8 spatial scale, so 8 output px is a 1-latent-px tile over the
/// fixture's 8×8 latent — the most adversarial partition available, and the one
/// `mlx-gen-z-image/tests/vae_decoder.rs` pins the shared decoder at.
const BOUNDED_TILE_PX: i32 = 8;

fn fixture_vae() -> (Vae, Array) {
    let w = Weights::from_file(FIXTURE)
        .unwrap_or_else(|e| panic!("read {FIXTURE}: {e} (z-image tiny VAE decoder fixture)"));
    let vae = Vae::from_weights(&w, "", &small_cfg()).expect("build the shared z-image Vae");
    let latent = w.require("in.latent").expect("in.latent").clone(); // (1,16,8,8)
    (vae, latent)
}

/// The fixture tail's own spatial upsample (`small_cfg` has one upsampling up-block), used to plan
/// the pre-fix control's crops. Production's ×8 lives in [`VaeTiling::QWEN_IMAGE`]; the fixture is a
/// narrower/shallower instance of the same decoder, so its plan geometry must match what it is.
const FIXTURE_TAIL_SCALE: i32 = 2;

/// The **pre-sc-19753 route**, reconstructed from public API: run the dense head once
/// (`decode_pre_upsample` — that part never changed), then tile the *whole* tail per crop
/// (`Decoder::forward_upsample_tail`) and blend, so each crop's GroupNorm statistics reduce only its
/// own window. `out_tile_px` is in output pixels, matching `Vae::decode_tiled`'s own contract, so the
/// control and the shipped path partition at the same granularity.
fn whole_tail_tiled(vae: &Vae, latent: &Array, out_tile_px: i32) -> Array {
    let head = vae.decode_pre_upsample(latent).expect("dense head"); // (B,C,h,w)
    let head5 = head.expand_dims(2).expect("lift to (B,C,1,h,w)");
    let geometry = VaeTiling {
        spatial_scale: FIXTURE_TAIL_SCALE,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: 3,
    };
    let s = head5.shape().to_vec();
    let plan = TilingConfig::spatial_only(out_tile_px, 0).plan(geometry, s[2], s[3], s[4]);
    assert!(
        plan.h.len() > 1 && plan.w.len() > 1,
        "the pre-fix control must actually crop, got {}x{}",
        plan.h.len(),
        plan.w.len()
    );
    mlx_gen::vae_tiling::tiled_decode(&head5, &plan, [2, 3, 4], None, |tile| {
        let t = tile.shape().to_vec();
        let tail = vae
            .decoder()
            .forward_upsample_tail(&tile.reshape(&[t[0], t[1], t[3], t[4]])?)?;
        let o = tail.shape().to_vec();
        Ok(tail.reshape(&[o[0], o[1], 1, o[2], o[3]])?)
    })
    .expect("pre-fix whole-tail tiled decode")
}

fn max_abs_delta(a: &Array, b: &Array) -> f32 {
    a.eval().unwrap();
    b.eval().unwrap();
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// The production seam body of `Chroma::decode_with_vae` reconstructed from public API: unpack the
/// packed DiT latent (Chroma calls `mlx_gen_flux::unpack_latents` directly), admit the decoder for
/// the FLUX.1 latent space, then decode — bounded when a tiling config is supplied.
fn decode_via_seam(vae: &Vae, packed: &Array, tiling: Option<&TilingConfig>) -> Result<Array> {
    let unpacked = unpack_latents(packed, 64, 64)?;
    let decoder: &dyn LatentDecoder = vae;
    ensure_decoder_compatible(Some(&FLUX1_LATENT_SPACE), decoder)?;
    match tiling {
        Some(t) => decoder.decode_tiled(&unpacked, t, Some(&CancelFlag::new())),
        None => decoder.decode(&unpacked),
    }
}

/// The inheritance proof. Chroma's decode seam, driven on the shared decoder, must land on its own
/// dense decode. This is the assertion that would have failed before sc-19753: whole-tail tiling gave
/// each 1-latent-px crop its own GroupNorm statistics, which on this position-dependent fixture moves
/// the comparison by 1.851e0 (the pre-fix control below runs exactly that route). Measured here:
/// 1.073e-6 — two orders below the 1e-4 bound and six below the defect signal.
#[test]
fn chroma_bounded_decode_tracks_the_dense_decode() {
    let (vae, latent) = fixture_vae();
    let packed = pack_latents(&latent, 64, 64).expect("pack to the DiT token layout");

    let dense = decode_via_seam(&vae, &packed, None).expect("dense decode");
    let bounded = decode_via_seam(
        &vae,
        &packed,
        Some(&TilingConfig::spatial_only(BOUNDED_TILE_PX, 0)),
    )
    .expect("bounded decode");

    assert_eq!(bounded.shape(), dense.shape());
    let delta = max_abs_delta(&dense, &bounded);
    println!("[chroma inheritance] bounded vs dense max|Δ|={delta:.3e}");
    assert!(
        delta < 1e-4,
        "Chroma's bounded decode diverged from its dense decode: max|Δ|={delta:.3e} — the shared \
         z-image decoder is no longer normalization-correct under tiling, or Chroma stopped routing \
         to it"
    );
}

/// The defect control — the **pre-sc-19753 route**, reconstructed from public API and executed on
/// the same fixture at the same tile granularity: the dense head once, then the *whole* tail per
/// crop (`Decoder::forward_upsample_tail`), so every crop's GroupNorm statistics reduce only its own
/// window. Without this the tolerance above could be satisfied by a decoder with no global
/// normalization at all; with it, the pair pins "bounded and correct" apart from "bounded per crop".
#[test]
fn the_pre_fix_whole_tail_route_is_observably_wrong_against_the_dense_decode() {
    let (vae, latent) = fixture_vae();
    let dense = vae.decode(&latent).expect("dense decode"); // (1,3,1,16,16)
    let whole_tail = whole_tail_tiled(&vae, &latent, BOUNDED_TILE_PX);

    assert_eq!(whole_tail.shape(), dense.shape());
    let delta = max_abs_delta(&dense, &whole_tail);
    println!("[chroma inheritance] pre-fix whole-tail vs dense max|Δ|={delta:.3e}");
    assert!(
        delta > 1e-1,
        "the pre-sc-19753 whole-tail route is indistinguishable from the dense decode on this \
         fixture (max|Δ|={delta:.3e}) — the bounded-vs-dense tolerance above no longer \
         discriminates and this fixture must be replaced with a position-dependent one"
    );
}

/// The tiling control. The bounded request must genuinely partition this geometry, and a request too
/// wide to tile must be the *exact* single-pass decode — otherwise the tracking assertion could be
/// passing against an untiled fall-through and prove nothing.
#[test]
fn the_bounded_request_actually_tiles_and_the_wide_request_does_not() {
    let (vae, latent) = fixture_vae();
    let sh = latent.shape();
    let (h, w) = (sh[2], sh[3]);

    let bounded = TilingConfig::spatial_only(BOUNDED_TILE_PX, 0);
    assert!(
        bounded.needs_tiling(VaeTiling::QWEN_IMAGE, 1, h, w),
        "the bounded request must fire for a {h}x{w} latent"
    );
    let plan = bounded.plan(VaeTiling::QWEN_IMAGE, 1, h, w);
    assert!(
        plan.h.len() > 1 && plan.w.len() > 1,
        "the bounded request must split both spatial axes, got {}x{}",
        plan.h.len(),
        plan.w.len()
    );

    let wide = TilingConfig::spatial_only(4096, 64);
    assert!(!wide.needs_tiling(VaeTiling::QWEN_IMAGE, 1, h, w));
    let packed = pack_latents(&latent, 64, 64).unwrap();
    let dense = decode_via_seam(&vae, &packed, None).unwrap();
    let fallthrough = decode_via_seam(&vae, &packed, Some(&wide)).unwrap();
    assert_eq!(
        max_abs_delta(&dense, &fallthrough),
        0.0,
        "a request too wide to tile must be the exact single-pass decode"
    );
}

/// A pre-tripped cancel must be observed before any tensor work on the Chroma seam too.
#[test]
fn chroma_bounded_decode_honors_a_pretripped_cancel() {
    let (vae, latent) = fixture_vae();
    let cancel = CancelFlag::new();
    cancel.cancel();
    let decoder: &dyn LatentDecoder = &vae;
    for cfg in [
        TilingConfig::spatial_only(BOUNDED_TILE_PX, 0),
        TilingConfig::spatial_only(4096, 64),
    ] {
        assert!(matches!(
            decoder.decode_tiled(&latent, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }
}
