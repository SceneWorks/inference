//! MOSS FP32 arithmetic must stay FP32 even when MLX enables TF32 on NAX hardware.
use mlx_gen::attention::slice_axis;
use mlx_gen::{CancelFlag, Result};
use mlx_rs::ops::{concatenate_axis, divide, multiply, softmax_axis, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

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

/// Query tiling bounds the source-faithful BF16 fallback without truncating keys.
/// Eight reuses the existing F32 correctness bound; it is not a measured throughput optimum.
const EDIT_QUERY_ROWS: usize = 8;

fn edit_query_chunks(
    rows: i32,
    cancel: &CancelFlag,
    mut run: impl FnMut(i32, i32) -> Result<()>,
) -> Result<()> {
    for start in (0..rows).step_by(EDIT_QUERY_ROWS) {
        super::transformer::check(cancel)?;
        run(start, (EDIT_QUERY_ROWS as i32).min(rows - start))?;
        super::transformer::check(cancel)?;
    }
    Ok(())
}

/// Upper bound for this function's additional arrays, excluding the borrowed BF16 Q/K/V.
/// Per tile: dot, scaled/centered scores, exponential, rounded BF16 operand and widened
/// operand = 22 bytes per score. Q cast, numerator and normalized result = 12 bytes
/// per query channel; max/sum reductions = 8 bytes per query/head. BF16 results are
/// counted with the retained outputs below.
/// Persistent K/V casts = 8 bytes per key channel; retained chunks plus concatenation
/// = 2 bytes per output channel during tiles, 4 during concatenation. Those phases
/// cannot overlap: the tile graph is evaluated and dropped before advancing.
pub(crate) fn edit_attention_workspace(rows: u64, keys: u64, dim: u64, heads: u64) -> Option<u64> {
    let tile = rows.min(EDIT_QUERY_ROWS as u64);
    let scores = tile
        .checked_mul(keys)?
        .checked_mul(heads)?
        .checked_mul(22)?;
    let query = tile.checked_mul(dim)?.checked_mul(12)?;
    let reductions = tile.checked_mul(heads)?.checked_mul(8)?;
    let casts = keys.checked_mul(dim)?.checked_mul(8)?;
    let outputs = rows.checked_mul(dim)?.checked_mul(2)?;
    let tile_peak = scores
        .checked_add(query)?
        .checked_add(reductions)?
        .checked_add(outputs)?;
    casts.checked_add(tile_peak.max(outputs.checked_mul(2)?))
}

/// Match the frozen PyTorch BF16 SDPA operand boundary, also used by Candle edit.
/// Exponentials are rounded BEFORE the value product, while the normalization sum
/// and products remain F32. GEMV avoids NAX TF32 arithmetic; K/V broadcast as views.
/// This is edit-only: ordinary/DMD attention and the existing F32 path are unchanged.
pub(crate) fn attention_edit_bf16(
    q: &Array,
    k: &Array,
    v: &Array,
    cancel: &CancelFlag,
) -> Result<Array> {
    super::transformer::check(cancel)?;
    let keys = k.as_dtype(Dtype::Float32)?;
    let values = v.as_dtype(Dtype::Float32)?.transpose_axes(&[0, 1, 3, 2])?;
    eval([&keys, &values])?;
    let divisor = Array::from_slice(&[(q.shape()[3] as f32).sqrt()], &[1]);
    let mut outputs = Vec::new();
    edit_query_chunks(q.shape()[2], cancel, |start, length| {
        let output = {
            let query = slice_axis(q, 2, start, length)?.as_dtype(Dtype::Float32)?;
            let dot = keys
                .expand_dims(2)?
                .matmul(&query.expand_dims(-1)?)?
                .squeeze_axes(&[-1])?;
            let scores = divide(&dot, &divisor)?;
            let centered = subtract(&scores, scores.max_axis(-1, true)?)?;
            let exponentials = centered.exp()?;
            let denominator = exponentials.sum_axis(-1, true)?;
            let rounded = exponentials.as_dtype(Dtype::Bfloat16)?;
            let operand = rounded.as_dtype(Dtype::Float32)?;
            let numerator = values
                .expand_dims(2)?
                .matmul(&operand.expand_dims(-1)?)?
                .squeeze_axes(&[-1])?;
            let normalized = divide(&numerator, &denominator)?;
            let output = normalized.as_dtype(Dtype::Bfloat16)?;
            eval([&output])?;
            // All named tile arrays and their completed lazy graph leave this scope.
            output
        };
        outputs.push(output);
        Ok(())
    })?;
    let refs: Vec<_> = outputs.iter().collect();
    let output = concatenate_axis(&refs, 2)?;
    eval([&output])?;
    super::transformer::check(cancel)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::logical_f32_readback;

    #[test]
    fn edit_attention_chunk_geometry_and_cancellation_cpu() {
        let cancel = CancelFlag::new();
        let mut chunks = Vec::new();
        edit_query_chunks(19, &cancel, |start, count| {
            chunks.push((start, count));
            Ok(())
        })
        .unwrap();
        assert_eq!(chunks, [(0, 8), (8, 8), (16, 3)]);
        let mut calls = 0;
        assert!(matches!(
            edit_query_chunks(19, &cancel, |_, _| {
                calls += 1;
                cancel.cancel();
                Ok(())
            }),
            Err(mlx_gen::Error::Canceled)
        ));
        assert_eq!(calls, 1, "cancellation stops before another tile");
        assert!(matches!(
            edit_query_chunks(19, &cancel, |_, _| panic!("cancelled work started")),
            Err(mlx_gen::Error::Canceled)
        ));
    }

    #[test]
    fn edit_attention_workspace_checked_cpu() {
        // Independently sum live byte widths for H2,D16,Q19,K23 and an 8-query tile.
        assert_eq!(
            edit_attention_workspace(19, 23, 16, 2),
            Some(22 * 8 * 23 * 2 + 12 * 8 * 16 + 8 * 8 * 2 + 8 * 23 * 16 + 2 * 19 * 16)
        );
        assert_eq!(
            edit_attention_workspace(3, 23, 16, 2),
            Some(22 * 3 * 23 * 2 + 12 * 3 * 16 + 8 * 3 * 2 + 8 * 23 * 16 + 2 * 3 * 16)
        );
        assert!(edit_attention_workspace(19, u64::MAX, 16, 2).is_none());
        assert!(edit_attention_workspace(u64::MAX, 23, 16, 2).is_none());
    }

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
