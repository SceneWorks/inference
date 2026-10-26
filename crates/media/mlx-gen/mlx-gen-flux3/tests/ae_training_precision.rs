//! SYNTHETIC actual-HF training precision oracle; actual execution is macOS-only.
use mlx_gen::weights::Weights;
use mlx_gen_flux3::autoencoder::FrozenTrainingEncoder;
use mlx_rs::{Array, Dtype};

use crate::common::{cpu_process, fixture, nchw_to_nhwc, tiny_autoencoder};

// Never slice a strided array in physical storage order.
fn host(a: &Array) -> Vec<f32> {
    let flat = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
    flat.eval().unwrap();
    flat.as_slice::<f32>().to_vec()
}

#[test]
fn frozen_training_encoder_selected_precision() {
    cpu_process(
        "ae_training_precision::frozen_training_encoder_selected_precision",
        || {
            let g = fixture("flux3_ae_training_precision.safetensors");
            let mut weights = Weights::from_file(tiny_autoencoder()).unwrap();
            // An encoder-only inventory must load without ANY decoder parameter.
            weights.remove_prefix("decoder.");
            // A deliberately strided NHWC view of a non-square, distinct-channel image.
            let image = nchw_to_nhwc(g.require("image").unwrap());
            for (dtype, label, max_bound, mean_bound) in [
                (Dtype::Float32, "f32", 0.0002, 0.00004),
                (Dtype::Bfloat16, "bf16", 0.08, 0.02),
                (Dtype::Float16, "f16", 0.01, 0.002),
            ] {
                let ae = FrozenTrainingEncoder::from_weights(&weights, dtype).unwrap();
                assert_eq!(ae.compute_dtype(), dtype);
                assert_eq!(ae.latent_channels(), 16);
                let arrays = ae.arrays();
                assert!(arrays.len() > 2);
                for p in &arrays[..arrays.len() - 2] {
                    assert_eq!(p.dtype(), dtype);
                }
                for p in &arrays[arrays.len() - 2..] {
                    assert_eq!(p.dtype(), Dtype::Float32);
                }
                assert_eq!(
                    host(arrays[arrays.len() - 2]),
                    host(g.require("original_bn_mean").unwrap())
                );
                // Independent CPU sqrt may differ by one F32 ULP; half-rounded
                // source statistics differ by orders of magnitude more.
                let std_delta = host(arrays[arrays.len() - 1])
                    .into_iter()
                    .zip(host(g.require("original_bn_std").unwrap()))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                assert!(std_delta < 2.5e-7, "original F32 BN std delta={std_delta}");
                let (tokens, grid) = ae.encode(&image).unwrap();
                assert_eq!(grid, (2, 3));
                assert_eq!(tokens.dtype(), Dtype::Float32);
                assert_eq!(tokens.shape(), [6, 16]);
                let got = host(&tokens);
                let want = host(g.require(&format!("tokens_{label}")).unwrap());
                let diffs: Vec<_> = got.iter().zip(want).map(|(a, b)| (a - b).abs()).collect();
                let max = diffs.iter().copied().fold(0., f32::max);
                let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
                assert!(
                    max < max_bound && mean < mean_bound,
                    "{label} max={max} mean={mean}"
                );
            }
        },
    );
}
