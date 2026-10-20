//! MOSS FP32 arithmetic must stay FP32 even when MLX enables TF32 on NAX hardware.
use mlx_gen::attention::slice_axis;
use mlx_gen::{CancelFlag, Result};
use mlx_rs::ops::{concatenate_axis, multiply, softmax_axis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

/// Compute x W^T with independent column vectors, so the pinned Metal dispatcher retains N=1
/// and uses float-accumulating GEMV. W broadcasts as a view; it is not replicated per token.
/// Ordinary x W^T can flatten token rows into a TF32 GEMM on M5 even with FP32 arrays.
pub(crate) fn linear_fp32(x: &Array, weight: &Array) -> Result<Array> {
    Ok(weight.matmul(&x.expand_dims(-1)?)?.squeeze_axes(&[-1])?)
}

/// Preserve every query's full key domain while avoiding NAX's reduced-precision full SDPA.
/// At the production head dimension 128, <=8 query rows use MLX's fused FP32 vector kernel.
/// Unsupported tiny oracle dimensions use the same FP32 GEMV formulation for QK and PV.
/// Evaluate each chunk before continuing so fallback score storage is at most B*H*8*Sk.
pub(crate) fn attention_fp32(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    cancel: &CancelFlag,
) -> Result<Array> {
    let rows = q.shape()[2];
    let vector_kernel = matches!(q.shape()[3], 64 | 96 | 128 | 256);
    let mut outputs = Vec::new();
    for start in (0..rows).step_by(8) {
        super::transformer::check(cancel)?;
        let query = slice_axis(q, 2, start, 8.min(rows - start))?;
        let out = if vector_kernel {
            mlx_rs::fast::scaled_dot_product_attention(&query, k, v, scale, None, None)?
        } else {
            let scaled = multiply(&query, Array::from_slice(&[scale], &[1]))?;
            let scores = k
                .expand_dims(2)?
                .matmul(&scaled.expand_dims(-1)?)?
                .squeeze_axes(&[-1])?;
            let probabilities = softmax_axis(&scores, -1, true)?;
            v.transpose_axes(&[0, 1, 3, 2])?
                .expand_dims(2)?
                .matmul(&probabilities.expand_dims(-1)?)?
                .squeeze_axes(&[-1])?
        };
        eval([&out])?;
        super::transformer::check(cancel)?;
        outputs.push(out);
    }
    let refs: Vec<_> = outputs.iter().collect();
    Ok(concatenate_axis(&refs, 2)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::logical_f32_readback;

    fn values(n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + 1.0) * 0.137 + phase).sin() * 0.37)
            .collect()
    }

    #[test]
    fn fp32_projection_preserves_low_bits_and_broadcast_rows() {
        let (batches, rows, input, output) = (2, 17, 128, 33);
        let x = values(batches * rows * input, 0.19);
        let w = values(output * input, 0.53);
        let actual = linear_fp32(
            &Array::from_slice(&x, &[batches as i32, rows as i32, input as i32]),
            &Array::from_slice(&w, &[output as i32, input as i32]),
        )
        .unwrap();
        assert_eq!(
            actual.shape(),
            &[batches as i32, rows as i32, output as i32]
        );
        let got = logical_f32_readback(&actual).unwrap();
        assert!(got.iter().all(|v| v.is_finite()));
        let mut max_error = 0.0f32;
        for row in 0..batches * rows {
            for channel in 0..output {
                let expected: f64 = (0..input)
                    .map(|i| x[row * input + i] as f64 * w[channel * input + i] as f64)
                    .sum();
                max_error = max_error.max((got[row * output + channel] - expected as f32).abs());
            }
        }
        assert!(max_error < 2e-5, "FP32 projection max_abs={max_error}");
    }

    #[test]
    fn fp32_attention_preserves_author_math_at_production_and_tiny_head_dimensions() {
        // 17 rows crosses two vector-kernel boundaries; D128 exercises the production kernel,
        // while D8 exercises the fallback used by the independently frozen tiny author fixture.
        for dim in [8usize, 128] {
            let (heads, rows, keys) = (2usize, 17usize, 19usize);
            let q = values(heads * rows * dim, 0.17);
            let k = values(heads * keys * dim, 0.41);
            let v = values(heads * keys * dim, 0.83);
            let scale = 1.0 / (dim as f32).sqrt();
            let actual = attention_fp32(
                &Array::from_slice(&q, &[1, heads as i32, rows as i32, dim as i32]),
                &Array::from_slice(&k, &[1, heads as i32, keys as i32, dim as i32]),
                &Array::from_slice(&v, &[1, heads as i32, keys as i32, dim as i32]),
                scale,
                &CancelFlag::new(),
            )
            .unwrap();
            assert_eq!(actual.shape(), &[1, heads as i32, rows as i32, dim as i32]);
            let got = logical_f32_readback(&actual).unwrap();
            assert!(got.iter().all(|v| v.is_finite()));
            let mut max_error = 0.0f32;
            for head in 0..heads {
                for row in 0..rows {
                    let scores: Vec<f64> = (0..keys)
                        .map(|key| {
                            (0..dim)
                                .map(|d| {
                                    q[(head * rows + row) * dim + d] as f64
                                        * k[(head * keys + key) * dim + d] as f64
                                })
                                .sum::<f64>()
                                * scale as f64
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probabilities: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = probabilities.iter().sum();
                    for d in 0..dim {
                        let expected = (0..keys)
                            .map(|key| {
                                probabilities[key] / total * v[(head * keys + key) * dim + d] as f64
                            })
                            .sum::<f64>();
                        max_error = max_error
                            .max((got[(head * rows + row) * dim + d] - expected as f32).abs());
                    }
                }
            }
            assert!(
                max_error < 2e-5,
                "FP32 attention D={dim} max_abs={max_error}"
            );
        }
    }
}
