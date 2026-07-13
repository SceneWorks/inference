//! DC-AE **encoder** parity gate vs diffusers `AutoencoderDC` (sc-11803).
//!
//! The mlx-gen port (sc-8486 / mlx-gen #612) only ported the DECODER, so there was no encoder to
//! parity against and [`DcAeEncoder`] shipped with a round-trip **shape-only** smoke — a wrong
//! `DCDownBlock2d` op order or shortcut would pass silently. This gate sources a real reference
//! straight from diffusers (`tools/dump_dcae_encode_golden.py`): it encodes the SAME fixed image
//! through the candle port and checks it reproduces the diffusers encoder's latent.
//!
//! Dense f32 path, same "divergence is not rounding" logic as `decode_parity.rs`: a port bug (wrong
//! transpose/op order/layout) wrecks the mean; reduced-precision matmul rounding does not.
//!
//! `#[ignore]`d: needs the real `dc-ae-f32c32-sana-1.0` weights (~1.25 GB). The golden defaults to the
//! committed `tests/fixtures/dcae_encode_golden.safetensors` (256² image → 8² latent; ~0.8 MB).
//!
//! Run (CUDA):
//!   SANA_DCAE_WEIGHTS=/path/diffusion_pytorch_model.safetensors \
//!   cargo test -p candle-gen-sana --test encode_parity --features cuda --release -- --ignored --nocapture

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::Weights;
use candle_gen_sana::{DcAeConfig, DcAeEncoder};

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).expect("cuda device")
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::new_metal(0).expect("metal device")
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        Device::Cpu
    }
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let diff = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let denom = want
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    diff / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|`.
fn mean_rel(got: &Tensor, want: &Tensor) -> f32 {
    let num = (got - want)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let den = want
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    num / den.max(1e-12)
}

#[test]
#[ignore = "needs dc-ae-f32c32-sana-1.0 weights (golden defaults to the committed fixture)"]
fn encode_matches_reference() {
    let dev = device();
    let weights_path = std::env::var("SANA_DCAE_WEIGHTS").expect("set SANA_DCAE_WEIGHTS");
    let golden_path = std::env::var("SANA_DCAE_ENCODE_GOLDEN").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dcae_encode_golden.safetensors"
        )
        .into()
    });

    let golden = Weights::from_file(std::path::Path::new(&golden_path), &dev, DType::F32)
        .expect("load golden");
    let image = golden.require("image").expect("golden image"); // [1,3,256,256] NCHW
    let want = golden.require("latent").expect("golden latent"); // [1,32,8,8] NCHW

    let weights = Weights::from_file(std::path::Path::new(&weights_path), &dev, DType::F32)
        .expect("load weights");
    let encoder = DcAeEncoder::from_weights(&weights, &DcAeConfig::sana_f32c32()).expect("build");
    let got = encoder.encode(&image).expect("encode"); // [1,32,8,8] NCHW

    assert_eq!(got.dims(), want.dims(), "shape");
    let peak = peak_rel(&got, &want);
    let mean = mean_rel(&got, &want);
    println!(
        "DC-AE candle encode parity vs diffusers reference: peak_rel={peak:.5}  mean_rel={mean:.5}"
    );

    // Dense f32 path — `mean_rel` is the faithfulness gate: a `DCDownBlock2d` order/shortcut bug wrecks
    // the mean; reduced-precision matmul rounding does not. Same ballpark ceiling as the decode gate.
    assert!(
        mean < 1e-2,
        "mean_rel {mean} too high — that IS a port bug, not rounding"
    );
    // `peak_rel` looser by design (the linear-attention 1/(Σ+eps) normalizer amplifies a handful of
    // reduced-precision matmul noise); decode gate set this ceiling at 0.10.
    assert!(
        peak < 0.10,
        "peak_rel {peak} above the attention-normalizer precision ceiling"
    );
}
