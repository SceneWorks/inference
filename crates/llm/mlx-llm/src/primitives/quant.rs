//! Group-wise affine quantization (Q4 / Q8) for linear projections.
//!
//! Moved to [`mlx_layers::llm_projection`] (`SW-6957` / `SW-F335`) and re-exported here so all
//! internal decoders and external consumers remain 100% compatible.

pub use mlx_layers::llm_projection::{
    in_compute_dtype, quantize_on_stored_grid, QuantizedEmbedding, QuantizedLinear,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::nn::{embed, input_ids, linear};
    use mlx_rs::ops::{dequantize, quantize};
    use mlx_rs::{Array, Dtype};

    fn relative_error(got: &Array, want: &Array) -> f64 {
        let want = want.as_dtype(Dtype::Float32).unwrap();
        let got = got.as_dtype(Dtype::Float32).unwrap();
        let fro =
            |a: &Array| f64::from(a.square().unwrap().sum(None).unwrap().item::<f32>()).sqrt();
        fro(&got.subtract(&want).unwrap()) / fro(&want)
    }

    /// Dequantize with the scales and biases widened to f32 — what `quantized_matmul` computes
    /// in (it never rounds the dequantized weight to the scale dtype, as `dequantize` does).
    fn dequantized(words: &Array, scales: &Array, biases: &Array, bits: i32) -> Array {
        let f = |a: &Array| a.as_dtype(Dtype::Float32).unwrap();
        dequantize(words, f(scales), Some(&f(biases)), 64, bits).unwrap()
    }

    fn seeded(shape: &[i32], seed: u64) -> Array {
        let key = mlx_rs::random::key(seed).unwrap();
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&key)).unwrap()
    }

    /// sc-26266: `quantize_on_stored_grid` packs in MLX's layout — on an f32 weight (whose f32
    /// scales are the grid MLX rounded to) it reproduces `quantize`'s words exactly — and on a
    /// bf16 weight its q8 packing matches the f32-grid optimum where MLX's lands ~1.5 x above.
    ///
    /// *Mutations that red this:* a wrong bit order or word width (words differ); rounding to the
    /// f32 grid (the bf16 q8 ratio).
    #[test]
    fn stored_grid_packing_is_mlx_layout_and_reaches_the_f32_grid_optimum() {
        for bits in [4, 8] {
            let w = seeded(&[64, 256], 7 + bits as u64);
            let (mlx_words, ..) = quantize(&w, 64, bits).unwrap();
            let (words, scales, biases) = quantize_on_stored_grid(&w, 64, bits).unwrap();
            assert_eq!(words.dtype(), Dtype::Uint32);
            assert_eq!(words.shape(), mlx_words.shape());
            let same = words
                .eq(&mlx_words)
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>();
            assert!(same, "q{bits}: the f32 packing differs from MLX's words");

            let bf16 = w.as_dtype(Dtype::Bfloat16).unwrap();
            let optimum = relative_error(&dequantized(&words, &scales, &biases, bits), &w);
            let (words, scales, biases) = quantize_on_stored_grid(&bf16, 64, bits).unwrap();
            assert_eq!(scales.dtype(), Dtype::Bfloat16);
            let stored = relative_error(&dequantized(&words, &scales, &biases, bits), &bf16);
            let (words, scales, biases) = quantize(&bf16, 64, bits).unwrap();
            let computed = relative_error(&dequantized(&words, &scales, &biases, bits), &bf16);
            eprintln!(
                "q{bits}: f32 grid {optimum:.4e}, stored grid {stored:.4e}, MLX {computed:.4e}"
            );
            assert!(
                stored <= 1.05 * optimum,
                "q{bits}: stored grid {stored} vs optimum {optimum}"
            );
            if bits == 8 {
                assert!(
                    computed > 1.3 * optimum,
                    "MLX's q8 bf16 packing is already optimal"
                );
            }
        }
    }

    /// A `QuantGrid::Stored` spec reaches the load-time quantizer (`Projection::load`).
    ///
    /// *Mutation that reds this:* `Projection::load_with_bias` ignoring `QuantSpec::grid`.
    #[test]
    fn a_stored_grid_spec_packs_projections_on_the_stored_grid() {
        use crate::primitives::projection::{Projection, QuantGrid, QuantSpec};
        let w = seeded(&[32, 128], 3).as_dtype(Dtype::Bfloat16).unwrap();
        let spec = QuantSpec {
            grid: QuantGrid::Stored,
            ..QuantSpec::q8()
        };
        let Projection::Quantized(q) = Projection::load(w.clone(), Some(spec)).unwrap() else {
            panic!("not quantized")
        };
        let (words, ..) = quantize_on_stored_grid(&w, 64, 8).unwrap();
        assert!(q
            .weight
            .eq(&words)
            .unwrap()
            .all(None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn quantized_embedding_gathers_rows_without_expanding_the_table() {
        let dense = Array::from_slice(
            &(0..4 * 64)
                .map(|i| ((i * 7 % 29) as f32 / 29.0) - 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        );
        let linear = QuantizedLinear::quantize(&dense, 64, 4, None).unwrap();
        let embedding = QuantizedEmbedding::from_quantized(
            linear.weight.clone(),
            linear.scales.clone(),
            linear.biases.clone(),
            linear.group_size,
            linear.bits,
            Dtype::Float32,
        )
        .unwrap();
        let ids = input_ids(&[3, 1]);
        let actual = embedding.forward(&ids).unwrap();
        let expected = embed(&linear.dequantize_weight().unwrap(), &ids).unwrap();
        assert_eq!(actual.shape(), &[1, 2, 64]);
        let actual = actual.as_slice::<f32>();
        let expected = expected.as_slice::<f32>();
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (*left - *right).abs() <= f32::EPSILON));
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::quant::tests::quantized_embedding_gathers_rows_without_expanding_the_table");
    }

    /// Quantize→dequantize should round-trip within the affine grid's tolerance.
    #[test]
    fn quantize_dequantize_roundtrip_q8() {
        // [out=2, in=64] so a single group of 64 covers the input dim.
        let n = 2 * 64;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - 0.5).collect();
        let w = Array::from_slice(&data, &[2, 64]);
        let q = QuantizedLinear::quantize(&w, 64, 8, None).unwrap();
        let recon = q.dequantize_weight().unwrap();
        let orig = w.as_slice::<f32>().to_vec();
        let back = recon.as_slice::<f32>().to_vec();
        let max_err = orig
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // 8-bit affine over a ~1.0 range: error well under 1%.
        assert!(max_err < 0.01, "max_err = {max_err}");
    }

    /// Quantized matmul should approximate the dense linear it replaces.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn quantized_matmul_approximates_linear_q8() {
        let n = 4 * 64;
        let wdata: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
        let w = Array::from_slice(&wdata, &[4, 64]); // [out=4, in=64]
        let x = Array::from_slice(
            &(0..64).map(|i| (i as f32 / 64.0) - 0.5).collect::<Vec<_>>(),
            &[1, 64],
        );
        let dense = linear(&x, &w, None).unwrap().as_slice::<f32>().to_vec();
        let q = QuantizedLinear::quantize(&w, 64, 8, None).unwrap();
        let quant = q.forward(&x).unwrap().as_slice::<f32>().to_vec();
        for (a, b) in dense.iter().zip(&quant) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::quant::tests::quantized_matmul_approximates_linear_q8");
    }
}
