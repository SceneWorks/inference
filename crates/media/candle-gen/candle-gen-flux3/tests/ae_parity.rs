//! Autoencoder encode/decode vs upstream `flux3.autoencoder.AutoEncoder` on the miniature
//! `tiny-snapshot/ae.safetensors` — the same `flux3_ae_golden.safetensors` the MLX twin's
//! `ae_parity` reads (`crates/media/mlx-gen/tools/dump_flux3_ae_golden.py`).
//!
//! The oracle ran the frozen upstream module on CPU in f32; every gate here is a native f32
//! comparison on Candle CPU at the shared peak-relative tolerance. Golden tensors keep upstream's
//! `[C, H, W]` / `[h·w, C]` layouts, which are this crate's own (NCHW) up to the batch axis.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_flux3::autoencoder::DOWNSAMPLING_RATIO;
use candle_gen_flux3::Flux3AutoEncoder;
use gen_core_flux3::latent_ids;

use crate::common::{
    assert_close, batched_nchw, cpu, device, errors, fixture, host_f32, host_i32, tiny_autoencoder,
    Fixture,
};

const GOLDEN: &str = "flux3_ae_golden.safetensors";
/// Peak-relative f32 tolerance shared by every gate (`assert_close`).
const TOL: f32 = 1e-4;

fn golden() -> Fixture {
    fixture(GOLDEN)
}

fn autoencoder(device: &Device) -> Flux3AutoEncoder {
    Flux3AutoEncoder::load(&tiny_autoencoder(), device).expect("the miniature autoencoder loads")
}

/// The golden's token grid `(h, w)` from its image size and the contract's ratio.
fn golden_grid(g: &Fixture) -> (usize, usize) {
    let sh = g.require("image").dims().to_vec();
    assert_eq!(sh.len(), 3, "golden image is [3, H, W]");
    (sh[1] / DOWNSAMPLING_RATIO, sh[2] / DOWNSAMPLING_RATIO)
}

#[test]
fn encode_matches_upstream() {
    let g = golden();
    let ae = autoencoder(&cpu());
    let (gh, gw) = golden_grid(&g);
    let want = g.require("tokens");

    let image = batched_nchw(g.require("image"));
    assert_eq!(image.dim(1).unwrap(), 3);
    let (tokens, (h, w)) = ae.encode(&image).expect("encode");

    assert_eq!((h, w), (gh, gw), "token grid");
    assert_eq!(
        tokens.dims(),
        want.dims(),
        "token layout [h·w, 4·EMBED_DIM]"
    );
    assert_eq!(
        ae.latent_channels(),
        want.dim(1).unwrap(),
        "latent_channels() must be the packed width the oracle produced"
    );
    assert_close("encode tokens", &tokens, want, TOL);
}

/// The oracle's ids are `latent_ids(h, w)`; the port emits tokens in that order (proven by the
/// value parity above) and `gen_core_flux3::latent_ids` reproduces the same table, so the DiT's
/// positional ids and the autoencoder's token order agree.
#[test]
fn token_order_is_row_major_latent_ids() {
    let g = golden();
    let (h, w) = golden_grid(&g);
    let ids = g.require("ids");
    assert_eq!(ids.dims(), &[h * w, 4]);
    let want: Vec<i32> = latent_ids(h, w).into_iter().flatten().collect();
    assert_eq!(host_i32(ids), want, "golden ids are gen_core's latent_ids");

    // The port's order is load-bearing: feeding the golden tokens to `decode` in column-major
    // order (the same multiset of tokens, transposed grid walk) must land visibly outside the
    // gate, otherwise the order half of the contract would be untested.
    let ae = autoencoder(&cpu());
    let tokens = g.require("tokens");
    let decoded_want = batched_nchw(g.require("decoded"));
    let column_major: Vec<u32> = (0..w)
        .flat_map(|c| (0..h).map(move |r| (r * w + c) as u32))
        .collect();
    let perm = Tensor::from_vec(column_major, h * w, &Device::Cpu).unwrap();
    let shuffled = tokens.index_select(&perm, 0).unwrap();
    let decoded = ae.decode(&shuffled, h, w).expect("decode");
    let (max_abs, peak, _) = errors(&decoded, &decoded_want);
    assert!(
        max_abs > 100.0 * TOL * peak.max(1.0),
        "column-major token order must not reproduce the row-major decode (max|Δ|={max_abs:.3e})"
    );
}

/// `decode(encode(image))` through the port against the oracle's own round trip — exercises the
/// bn de-normalization, the 2×2 unpatchify, `post_quant_conv` and the whole decoder.
#[test]
fn decode_round_trip_matches_upstream() {
    let g = golden();
    let ae = autoencoder(&cpu());
    let (h, w) = golden_grid(&g);
    let want = batched_nchw(g.require("decoded"));

    let image = batched_nchw(g.require("image"));
    let (tokens, (eh, ew)) = ae.encode(&image).expect("encode");
    assert_eq!((eh, ew), (h, w));
    let decoded = ae.decode(&tokens, h, w).expect("decode");
    assert_eq!(decoded.dims(), image.dims(), "decode restores [1, 3, H, W]");
    assert_eq!(decoded.dims(), want.dims());
    assert_close("decode(encode(image))", &decoded, &want, TOL);

    // The decoder output is the raw `conv_out` (unclamped): the oracle's golden exceeds [-1, 1]
    // on this seeded miniature, and the port must not clamp it.
    let peak = host_f32(&want).iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        peak > 1.0,
        "golden must exercise the unclamped range (peak {peak})"
    );
}

/// Decode of seeded N(0, 1) tokens — the DiT-output regime, independent of the encoder.
#[test]
fn decode_random_tokens_matches_upstream() {
    let g = golden();
    let ae = autoencoder(&cpu());
    let (h, w) = golden_grid(&g);
    let tokens = g.require("tokens_rand");
    let want = batched_nchw(g.require("decoded_rand"));

    let decoded = ae.decode(tokens, h, w).expect("decode");
    assert_close("decode(tokens_rand)", &decoded, &want, TOL);

    // `tokens.float()`: a bf16 token tensor is accepted and cast (the sampler hands over bf16
    // values), and the result tracks the f32 decode to bf16's own rounding, not to zero.
    let bf16 = tokens.to_dtype(DType::BF16).unwrap();
    let decoded_bf16 = ae.decode(&bf16, h, w).expect("decode bf16 tokens");
    let (max_abs, peak, _) = errors(&decoded_bf16, &want);
    assert!(
        max_abs < 5e-2 * peak.max(1.0),
        "bf16 tokens decode within bf16 rounding of the f32 golden (max|Δ|={max_abs:.3e})"
    );
    assert!(
        max_abs > 0.0,
        "bf16 input rounding must be observable (cast happened)"
    );
}

/// The same decode on the build's device: under `--features cuda` the real CUDA conv / GroupNorm /
/// attention kernels, held to the f32 oracle at a kernel-order tolerance (cuDNN-free candle convs
/// are im2col GEMMs; f32 on both sides).
#[test]
fn decode_random_tokens_matches_upstream_on_device() {
    let device = device();
    let g = golden();
    let ae = autoencoder(&device);
    let (h, w) = golden_grid(&g);
    let tokens = g.require("tokens_rand").to_device(&device).unwrap();
    let want = batched_nchw(g.require("decoded_rand"));
    let decoded = ae.decode(&tokens, h, w).expect("decode");
    assert_close("decode(tokens_rand) on device", &decoded, &want, 1e-3);
}

#[test]
fn load_reads_geometry_and_rejects_missing_keys() {
    let g = golden();
    let ae = autoencoder(&cpu());
    assert_eq!(ae.latent_channels(), g.require("tokens").dim(1).unwrap());

    // Geometry errors are typed and named.
    let (h, w) = golden_grid(&g);
    let bad_image = Tensor::zeros(
        (1, 3, h * DOWNSAMPLING_RATIO + 8, w * DOWNSAMPLING_RATIO),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let err = ae
        .encode(&bad_image)
        .expect_err("non-multiple side is refused");
    assert!(
        err.to_string().contains("multiples of 16"),
        "named geometry error, got {err}"
    );
    let short = Tensor::zeros((h * w - 1, ae.latent_channels()), DType::F32, &Device::Cpu).unwrap();
    assert!(
        ae.decode(&short, h, w).is_err(),
        "token count must match the grid"
    );

    // Each of these keys is load-bearing: dropping it is an error naming that key.
    let full = candle_gen::candle_core::safetensors::load(tiny_autoencoder(), &Device::Cpu)
        .expect("tiny ae.safetensors loads");
    for key in [
        "encoder.down.1.block.0.nin_shortcut.weight",
        "decoder.mid.attn_1.proj_out.bias",
        "decoder.up.1.upsample.conv.weight",
        "encoder.down.0.downsample.conv.bias",
        "encoder.quant_conv.weight",
        "decoder.post_quant_conv.bias",
        "bn.running_var",
    ] {
        let mut partial = full.clone();
        assert!(partial.remove(key).is_some(), "{key} is in the fixture");
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ae.safetensors");
        candle_gen::candle_core::safetensors::save(&partial, &file).unwrap();
        match Flux3AutoEncoder::load(&file, &Device::Cpu) {
            Err(e) => assert!(
                e.to_string().contains(&format!("missing tensor: {key}")),
                "{key}: the missing key is named, got {e}"
            ),
            Ok(_) => panic!("{key}: the autoencoder loaded without it"),
        }
    }
}
