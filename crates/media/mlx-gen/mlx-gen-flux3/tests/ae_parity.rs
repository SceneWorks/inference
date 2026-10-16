//! Autoencoder encode/decode vs upstream `flux3.autoencoder.AutoEncoder` on the miniature
//! `tests/fixtures/tiny-snapshot/ae.safetensors` (`../tools/dump_flux3_ae_golden.py`, delegated
//! leaf of sc-26234).
//!
//! The oracle ran the frozen upstream module on CPU in f32; every gate here is a native f32
//! comparison on the MLX CPU stream (`cpu_process` + `on_cpu`) at the shared peak-relative
//! tolerance. Golden tensors keep upstream's `[C, H, W]` / `[h·w, C]` layouts and are converted
//! at the test boundary with `nchw_to_nhwc` / `nhwc_to_nchw`.

use std::collections::HashMap;

use mlx_gen::gen_core::flux3::latent_ids;
use mlx_gen::weights::Weights;
use mlx_gen_flux3::autoencoder::DOWNSAMPLING_RATIO;
use mlx_gen_flux3::Flux3AutoEncoder;
use mlx_rs::{Array, Dtype};

use crate::common::{
    assert_close, cpu_process, errors, fixture, host_f32, host_i32, nchw_to_nhwc, nhwc_to_nchw,
    tiny_autoencoder,
};

const GOLDEN: &str = "flux3_ae_golden.safetensors";
/// Peak-relative f32 tolerance shared by every gate (`assert_close`).
const TOL: f32 = 1e-4;

fn golden() -> Weights {
    fixture(GOLDEN)
}

fn autoencoder() -> Flux3AutoEncoder {
    let w = Weights::from_file(tiny_autoencoder()).expect("tiny ae.safetensors loads");
    Flux3AutoEncoder::from_weights(&w).expect("the miniature autoencoder loads")
}

/// The golden's token grid `(h, w)` from its image size and the contract's ratio.
fn golden_grid(g: &Weights) -> (i32, i32) {
    let sh = g.require("image").unwrap().shape().to_vec();
    assert_eq!(sh.len(), 3, "golden image is [3, H, W]");
    (sh[1] / DOWNSAMPLING_RATIO, sh[2] / DOWNSAMPLING_RATIO)
}

/// `[C, H, W]` golden → `[1, C, H, W]`, the layout `nhwc_to_nchw` yields for the port's output.
fn batched(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap().expand_dims(0).unwrap()
}

#[test]
fn encode_matches_upstream() {
    cpu_process(
        "ae_parity::encode_matches_upstream",
        encode_matches_upstream_body,
    );
}

fn encode_matches_upstream_body() {
    let g = golden();
    let ae = autoencoder();
    let (gh, gw) = golden_grid(&g);
    let want = g.require("tokens").unwrap();

    let image = nchw_to_nhwc(g.require("image").unwrap());
    assert_eq!(image.shape()[3], 3);
    let (tokens, (h, w)) = ae.encode(&image).expect("encode");
    tokens.eval().unwrap();

    assert_eq!((h, w), (gh, gw), "token grid");
    assert_eq!(
        tokens.shape(),
        want.shape(),
        "token layout [h·w, 4·EMBED_DIM]"
    );
    assert_eq!(
        ae.latent_channels(),
        want.shape()[1],
        "latent_channels() must be the packed width the oracle produced"
    );
    assert_close("encode tokens", &tokens, want, TOL);
}

#[test]
fn token_order_is_row_major_latent_ids() {
    cpu_process(
        "ae_parity::token_order_is_row_major_latent_ids",
        token_order_is_row_major_latent_ids_body,
    );
}

/// The oracle's ids are `latent_ids(h, w)`; the port emits tokens in that order (proven by the
/// value parity above) and `gen_core::flux3::latent_ids` reproduces the same table, so the DiT's
/// positional ids and the autoencoder's token order agree.
fn token_order_is_row_major_latent_ids_body() {
    let g = golden();
    let (h, w) = golden_grid(&g);
    let ids = g.require("ids").unwrap();
    assert_eq!(ids.shape(), &[h * w, 4]);
    let want: Vec<i32> = latent_ids(h as usize, w as usize)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(host_i32(ids), want, "golden ids are gen_core's latent_ids");

    // The port's order is load-bearing: feeding the golden tokens to `decode` in column-major
    // order (the same multiset of tokens, transposed grid walk) must land visibly outside the
    // gate, otherwise the order half of the contract would be untested.
    let ae = autoencoder();
    let tokens = g.require("tokens").unwrap();
    let decoded_want = batched(g.require("decoded").unwrap());
    let column_major: Vec<i32> = (0..w)
        .flat_map(|c| (0..h).map(move |r| r * w + c))
        .collect();
    let perm = Array::from_slice(&column_major, &[h * w]);
    let shuffled = tokens.take_axis(&perm, 0).unwrap();
    let decoded = nhwc_to_nchw(&ae.decode(&shuffled, h, w).expect("decode"));
    decoded.eval().unwrap();
    let (max_abs, peak, _) = errors(&decoded, &decoded_want);
    assert!(
        max_abs > 100.0 * TOL * peak.max(1.0),
        "column-major token order must not reproduce the row-major decode (max|Δ|={max_abs:.3e})"
    );
}

#[test]
fn decode_round_trip_matches_upstream() {
    cpu_process(
        "ae_parity::decode_round_trip_matches_upstream",
        decode_round_trip_matches_upstream_body,
    );
}

/// `decode(encode(image))` through the port against the oracle's own round trip — exercises the
/// bn de-normalization, the 2×2 unpatchify, `post_quant_conv` and the whole decoder.
fn decode_round_trip_matches_upstream_body() {
    let g = golden();
    let ae = autoencoder();
    let (h, w) = golden_grid(&g);
    let want = batched(g.require("decoded").unwrap());

    let image = nchw_to_nhwc(g.require("image").unwrap());
    let (tokens, (eh, ew)) = ae.encode(&image).expect("encode");
    assert_eq!((eh, ew), (h, w));
    let decoded = ae.decode(&tokens, h, w).expect("decode");
    decoded.eval().unwrap();
    assert_eq!(
        decoded.shape(),
        image.shape(),
        "decode restores [1, H, W, 3]"
    );
    let decoded = nhwc_to_nchw(&decoded);
    assert_eq!(decoded.shape(), want.shape());
    assert_close("decode(encode(image))", &decoded, &want, TOL);

    // The decoder output is the raw `conv_out` (unclamped): the oracle's golden exceeds [-1, 1]
    // on this seeded miniature, and the port must not clamp it.
    let peak = host_f32(&want).iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        peak > 1.0,
        "golden must exercise the unclamped range (peak {peak})"
    );
}

#[test]
fn decode_random_tokens_matches_upstream() {
    cpu_process(
        "ae_parity::decode_random_tokens_matches_upstream",
        decode_random_tokens_matches_upstream_body,
    );
}

/// Decode of seeded N(0, 1) tokens — the DiT-output regime, independent of the encoder.
fn decode_random_tokens_matches_upstream_body() {
    let g = golden();
    let ae = autoencoder();
    let (h, w) = golden_grid(&g);
    let tokens = g.require("tokens_rand").unwrap();
    let want = batched(g.require("decoded_rand").unwrap());

    let decoded = nhwc_to_nchw(&ae.decode(tokens, h, w).expect("decode"));
    decoded.eval().unwrap();
    assert_close("decode(tokens_rand)", &decoded, &want, TOL);

    // `tokens.float()`: a bf16 token tensor is accepted and cast (the sampler hands over bf16),
    // and the result tracks the f32 decode to bf16's own rounding, not to zero.
    let bf16 = tokens.as_dtype(Dtype::Bfloat16).unwrap();
    let decoded_bf16 = nhwc_to_nchw(&ae.decode(&bf16, h, w).expect("decode bf16 tokens"));
    decoded_bf16.eval().unwrap();
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

#[test]
fn load_reads_geometry_and_rejects_missing_keys() {
    cpu_process(
        "ae_parity::load_reads_geometry_and_rejects_missing_keys",
        load_reads_geometry_and_rejects_missing_keys_body,
    );
}

fn load_reads_geometry_and_rejects_missing_keys_body() {
    let full = Weights::from_file(tiny_autoencoder()).expect("tiny ae.safetensors loads");
    let ae = Flux3AutoEncoder::from_weights(&full).expect("loads");
    let g = golden();
    // Every parameter is f32 and reachable for the load-time evaluation sweep. The count is the
    // fixture's tensor count minus the ignored `bn.num_batches_tracked`: every stored weight,
    // bias and norm parameter is held as-is, `bn.running_mean` as-is, and `bn.running_var` as
    // the derived `bn_std` (var folded with eps) — one array per stored tensor.
    let params = ae.arrays();
    assert!(params.iter().all(|a| a.dtype() == Dtype::Float32));
    let stored = full
        .keys()
        .filter(|k| *k != "bn.num_batches_tracked")
        .count();
    assert_eq!(
        params.len(),
        stored,
        "arrays() covers every stored parameter"
    );
    assert_eq!(
        ae.latent_channels(),
        g.require("tokens").unwrap().shape()[1]
    );

    // Geometry errors are typed and named.
    let (h, w) = golden_grid(&g);
    let bad_image =
        Array::zeros::<f32>(&[1, h * DOWNSAMPLING_RATIO + 8, w * DOWNSAMPLING_RATIO, 3]).unwrap();
    let err = ae
        .encode(&bad_image)
        .expect_err("non-multiple side is refused");
    assert!(
        err.to_string().contains("multiples of 16"),
        "named geometry error, got {err}"
    );
    let short = Array::zeros::<f32>(&[h * w - 1, ae.latent_channels()]).unwrap();
    assert!(
        ae.decode(&short, h, w).is_err(),
        "token count must match the grid"
    );

    // Each of these keys is load-bearing: dropping it is a `MissingTensor` naming that key.
    for key in [
        "encoder.down.1.block.0.nin_shortcut.weight",
        "decoder.mid.attn_1.proj_out.bias",
        "decoder.up.1.upsample.conv.weight",
        "encoder.down.0.downsample.conv.bias",
        "encoder.quant_conv.weight",
        "decoder.post_quant_conv.bias",
        "bn.running_var",
    ] {
        let mut tensors: HashMap<String, Array> = full
            .keys()
            .filter(|k| *k != key)
            .map(|k| (k.to_owned(), full.get(k).unwrap().clone()))
            .collect();
        assert!(tensors.remove(key).is_none());
        let partial = Weights::from_map(tensors);
        match Flux3AutoEncoder::from_weights(&partial) {
            Err(mlx_gen::Error::MissingTensor(missing)) => {
                assert_eq!(missing, key, "the missing key is named")
            }
            Err(other) => panic!("{key}: expected MissingTensor, got {other}"),
            Ok(_) => panic!("{key}: the autoencoder loaded without it"),
        }
    }
}
