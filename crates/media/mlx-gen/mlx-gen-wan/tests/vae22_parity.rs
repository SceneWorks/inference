//! sc-2680 parity gate: the Wan 2.2 z48 `Wan22Vae` (vae22) must reproduce the `mlx_video`
//! reference's decode (causal `first_chunk`) + chunked encode.
//!
//! Like the S2 z16 gate, the 5B's production VAE weights are heavy, so this runs against a
//! **self-contained committed fixture**: a tiny `dec_dim=8`/`enc_dim=8` instance with the real
//! `z_dim=48`, seeded random weights, + reference decode/encode IO (`tools/dump_vae22_fixtures.py`,
//! ~4.4 MB). The architecture is width-parametric, so this exercises every vae22 path (channels-last
//! causal 3-D conv, channel-L2 `RMS_norm` eps 1e-24, per-frame attention, `DupUp3D`/`AvgDown3D`,
//! up/down `Resample` `time_conv` incl. the `first_chunk` interleave + chunk-cache, spatial 2×2
//! patchify, the chunked-encode `feat_cache`, mean/std denorm). It runs on Metal in CI — no `#[ignore]`.
//!
//! Honors "divergence is not rounding": the reference runs the VAE in f32; this port does too. The
//! only expected gap is the float-summation order between mlx `conv3d` and the reference's
//! conv2d-per-temporal-slice decomposition of the same convolution (bounded, like the 2.1 gate).

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder};
use mlx_gen_wan::{Wan22Vae, Wan22VideoDecoder};
use mlx_rs::{Array, Dtype};

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vae22.safetensors"
    );
    Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run tools/dump_vae22_fixtures.py)"))
}

/// `(max|Δ|, Σ|Δ| / Σ|ref|)` over two equal-length f32 slices.
fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

/// The tiny fixture uses `dec_dim = enc_dim = 8` with the real `z_dim = 48`.
fn vae(w: &Weights) -> Wan22Vae {
    Wan22Vae::from_weights_dims(w, 8, 8, 48).expect("build Wan22Vae")
}

/// Cosine similarity of two equal-length f32 slices.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

#[test]
fn vae22_decode_matches_reference() {
    let w = fixture();
    let vae = vae(&w);

    let dec_in = w.require("dec_in").expect("dec_in"); // [48, T, H, W] (channels-first, normalized)
    let exp = w.require("dec_out").expect("dec_out"); // [1, T', 16H, 16W, 3]
    let got = vae.decode(dec_in).expect("decode");
    assert_eq!(got.shape(), exp.shape(), "decode output shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[vae22 decode] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "decode diverged: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

/// SC-18309 N1: the z48 adapter must preserve the historical de-normalize/decode exactly and only
/// relayout the vendored channels-last `[B,T,H,W,3]` result to the generic NCTHW contract.
#[test]
fn vae22_video_adapter_is_byte_exact_to_legacy_route() {
    let w = fixture();
    let vae = vae(&w);
    let latent = w.require("dec_in").unwrap();
    let legacy = vae
        .decode(latent)
        .unwrap()
        .transpose_axes(&[0, 4, 1, 2, 3])
        .unwrap();
    let seam = Wan22VideoDecoder::new(&vae).decode(latent).unwrap();
    assert_eq!(seam.shape(), legacy.shape());
    assert_eq!(
        seam.reshape(&[-1]).unwrap().as_slice::<f32>(),
        legacy.reshape(&[-1]).unwrap().as_slice::<f32>()
    );
}

/// A valid z48 latent and valid spatial policy prove pre-cancellation on both an actually-firing
/// tiled-head route and an actually non-firing monolithic fallback. The explicit `needs_tiling`
/// assertions prevent malformed input from making those route labels vacuous.
#[test]
fn vae22_valid_firing_and_fallback_routes_precancel() {
    let w = fixture();
    let vae = vae(&w);
    let latent = Array::zeros::<f32>(&[48, 2, 3, 3]).unwrap();
    let shape = latent.shape();
    let firing = mlx_gen::tiling::TilingConfig::spatial_only(32, 16);
    let fallback = mlx_gen::tiling::TilingConfig::spatial_only(4096, 64);
    assert!(firing.needs_tiling(
        mlx_gen::tiling::VaeTiling::WAN22,
        shape[1],
        shape[2],
        shape[3]
    ));
    assert!(!fallback.needs_tiling(
        mlx_gen::tiling::VaeTiling::WAN22,
        shape[1],
        shape[2],
        shape[3]
    ));

    let cancel = CancelFlag::new();
    cancel.cancel();
    for cfg in [firing, fallback] {
        assert!(matches!(
            Wan22VideoDecoder::new(&vae).decode_tiled(&latent, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            vae.decode_tiled(&latent, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }
}

/// Malformed-input dominance is a separate guarantee from the valid route proof: cancellation must
/// win before the adapter rank check and before the inherent VAE transposes or inspects geometry.
#[test]
fn vae22_precancel_wins_before_malformed_input_validation() {
    let w = fixture();
    let vae = vae(&w);
    let malformed = Array::from_slice(&[1.0f32], &[1]);
    let cancel = CancelFlag::new();
    cancel.cancel();
    for cfg in [
        mlx_gen::tiling::TilingConfig::spatial_only(32, 16),
        mlx_gen::tiling::TilingConfig::spatial_only(4096, 64),
    ] {
        assert!(matches!(
            Wan22VideoDecoder::new(&vae).decode_tiled(&malformed, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
        assert!(matches!(
            vae.decode_tiled(&malformed, &cfg, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }
}

#[test]
fn vae22_bf16_decode_is_finite_and_close_to_f32() {
    // sc-5039: a bf16 decode (weights + activations cast to bf16, keeping the f32 RMS_norm reduction
    // and the latent denorm) must stay finite and close to the f32 decode. The tiny dec_dim=8
    // fixture can't surface the 1024-channel dynamic range (that's the real-weight wedge check), but
    // it gates NaNs and the structural bf16 path across every op (causal conv3d, RMS_norm, attention,
    // DupUp3D, time_conv interleave, unpatchify).
    let w = fixture();
    let dec_in = w.require("dec_in").expect("dec_in"); // stays f32 — the latent isn't pre-cast
    let f32_out = vae(&w).decode(dec_in).expect("f32 decode");

    let mut wb = fixture();
    wb.cast_all(Dtype::Bfloat16).expect("cast fixture to bf16");
    let bf16_out = vae(&wb).decode(dec_in).expect("bf16 decode");
    assert_eq!(bf16_out.shape(), f32_out.shape());

    let (g, f) = (bf16_out.as_slice::<f32>(), f32_out.as_slice::<f32>());
    assert!(
        g.iter().all(|v| v.is_finite()),
        "bf16 decode produced non-finite values (NaN/Inf)"
    );
    let cos = cosine(g, f);
    println!("[vae22 bf16 decode] cosine(bf16, f32) = {cos:.6}");
    assert!(cos > 0.99, "bf16 decode cosine {cos:.4} too low vs f32");
}

#[test]
fn vae22_encode_single_frame_matches_reference() {
    // T=1 single-image encode (the TI2V conditioning path) — distinct chunking from the T=5 case.
    let w = fixture();
    let vae = vae(&w);
    let enc_in = w.require("enc_in1").expect("enc_in1");
    let exp = w.require("enc_out1").expect("enc_out1");
    let got = vae.encode(enc_in).expect("encode T=1");
    assert_eq!(got.shape(), exp.shape(), "T=1 encode output shape");
    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[vae22 encode T=1] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "T=1 encode diverged: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

#[test]
fn vae22_encode_matches_reference() {
    let w = fixture();
    let vae = vae(&w);

    let enc_in = w.require("enc_in").expect("enc_in"); // [1, T, H, W, 3] (channels-last, [-1,1])
    let exp = w.require("enc_out").expect("enc_out"); // [1, T_lat, H_lat, W_lat, 48]
    let got = vae.encode(enc_in).expect("encode");
    assert_eq!(got.shape(), exp.shape(), "encode output shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[vae22 encode] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "encode diverged: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}
