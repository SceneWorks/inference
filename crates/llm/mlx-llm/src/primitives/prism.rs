//! Prism ternary affine operators for Ternary Bonsai checkpoints.
//!
//! Moved to [`mlx_layers::prism`] (`SW-6957` / `SW-F335`) and re-exported here so all internal
//! decoders and external consumers remain 100% compatible.

#[cfg(test)]
pub(crate) use mlx_layers::prism::{
    block_hadamard, block_hadamard_unfused, fused_block_hadamard, fused_rotation_applies,
    hadamard_scale, recording_rotation_routes, RotationRoute, BITS, GROUP_SIZE,
};
pub use mlx_layers::prism::{PrismEmbedding, PrismLinear};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::primitives::nn::linear;
    use mlx_rs::ops::dequantize;
    use mlx_rs::{Array, Dtype, Stream};

    pub(crate) use mlx_layers::prism::with_unfused_rotation;

    fn bits(a: &Array) -> Vec<u32> {
        // Widening to F32 is exact for F16/BF16 and keeps the sign of zero and NaN payload class.
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn random_signs(width: i32, seed: u64) -> Array {
        let key = mlx_rs::random::key(seed).unwrap();
        let coin = mlx_rs::random::bernoulli(None, &[width][..], Some(&key)).unwrap();
        mlx_rs::ops::r#where(&coin, Array::from_f32(1.0), Array::from_f32(-1.0)).unwrap()
    }

    /// AC1: the fused kernel is bit-identical to the unfused cast -> sign -> Hadamard -> cast chain
    /// on random inputs at Ternary Bonsai 2 27B's projection input widths (hidden 5120, attention
    /// output 6144, MLP 17408, block 1024), for decode and prefill row counts, both directions and
    /// every activation dtype — plus smaller power-of-two blocks covering each final-radix branch.
    #[test]
    fn fused_rotation_is_bit_identical_to_the_unfused_chain() {
        mlx_rs::with_new_default_stream(Stream::gpu(), fused_rotation_matches_at_every_case);
    }

    fn fused_rotation_matches_at_every_case() {
        let gpu = Stream::task_local_or_default();
        let mut cases = Vec::new();
        for width in [5120, 6144, 17408] {
            cases.push((width, 1024));
        }
        for block in [2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 8192] {
            cases.push((block * 2, block));
        }
        let mut seed = 0u64;
        for (width, block) in cases {
            for rows in [1, 3, 17] {
                for dtype in [Dtype::Bfloat16, Dtype::Float16, Dtype::Float32] {
                    seed += 1;
                    let key = mlx_rs::random::key(seed).unwrap();
                    let x = mlx_rs::random::normal::<f32>(
                        &[1, rows, width][..],
                        None,
                        None,
                        Some(&key),
                    )
                    .unwrap()
                    .multiply(Array::from_f32(3.0))
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                    let signs = random_signs(width, seed + 10_000);
                    let scale = Array::from_slice(&[hadamard_scale(block)], &[1]);
                    for inverse in [false, true] {
                        assert!(fused_rotation_applies(&x, block, &gpu).is_ok());
                        let fused =
                            fused_block_hadamard(&x, &signs, &scale, block, width, inverse, &gpu)
                                .unwrap();
                        let reference = block_hadamard_unfused(
                            &x,
                            &signs,
                            block,
                            inverse,
                            hadamard_scale(block),
                        )
                        .unwrap();
                        assert_eq!(fused.dtype(), dtype);
                        assert_eq!(fused.shape(), x.shape());
                        assert!(
                            bits(&fused) == bits(&reference),
                            "width {width} block {block} rows {rows} {dtype:?} inverse {inverse}"
                        );
                    }
                }
            }
        }
    }

    /// Signed zeros, exact powers of two and extreme finite values keep their exact bits too.
    #[test]
    fn fused_rotation_matches_on_edge_values() {
        mlx_rs::with_new_default_stream(Stream::gpu(), fused_rotation_matches_edge_values);
    }

    fn fused_rotation_matches_edge_values() {
        let gpu = Stream::task_local_or_default();
        let width = 1024;
        let mut values = vec![0.0f32; width as usize];
        for (i, v) in values.iter_mut().enumerate() {
            *v = match i % 8 {
                0 => 0.0,
                1 => -0.0,
                2 => 65504.0,
                3 => -65504.0,
                4 => f32::MIN_POSITIVE,
                5 => 1.0e-8,
                6 => (i as f32).exp2().min(1.0e30),
                _ => -(i as f32) / 7.0,
            };
        }
        let x = Array::from_slice(&values, &[1, width]);
        let signs = random_signs(width, 77);
        let scale = Array::from_slice(&[hadamard_scale(width)], &[1]);
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            let x = x.as_dtype(dtype).unwrap();
            for inverse in [false, true] {
                let fused =
                    fused_block_hadamard(&x, &signs, &scale, width, width, inverse, &gpu).unwrap();
                let reference =
                    block_hadamard_unfused(&x, &signs, width, inverse, hadamard_scale(width))
                        .unwrap();
                assert!(bits(&fused) == bits(&reference), "{dtype:?} {inverse}");
            }
        }
    }

    /// A block MLX splits into two passes (> 8192) stays on the unfused chain, as does a forced
    /// oracle run or a CPU stream; the dispatcher never hands the kernel a shape it was not proven
    /// on, nor a stream it cannot run on.
    #[test]
    fn rotation_dispatch_uses_the_kernel_only_where_it_is_proven() {
        let gpu = Stream::gpu();
        let x = Array::zeros::<f32>(&[1, 16384]).unwrap();
        use crate::primitives::fused::{REASON_CPU_STREAM, REASON_FORCED_REFERENCE, REASON_SHAPE};
        assert_eq!(fused_rotation_applies(&x, 16384, &gpu), Err(REASON_SHAPE));
        assert_eq!(fused_rotation_applies(&x, 1536, &gpu), Err(REASON_SHAPE));
        assert_eq!(fused_rotation_applies(&x, 1024, &gpu), Ok(()));
        assert_eq!(
            fused_rotation_applies(&x, 1024, &Stream::cpu()),
            Err(REASON_CPU_STREAM)
        );
        with_unfused_rotation(|| {
            assert_eq!(
                fused_rotation_applies(&x, 1024, &gpu),
                Err(REASON_FORCED_REFERENCE)
            )
        });
        assert_eq!(fused_rotation_applies(&x, 1024, &gpu), Ok(()));
        // E3/E5: the process switch turned off is named as the reference route's reason.
        crate::switches::FUSED_ROTATION.scoped(false, || {
            assert_eq!(
                fused_rotation_applies(&x, 1024, &gpu),
                Err(crate::primitives::fused::REASON_DISABLED)
            )
        });
        let ints = Array::zeros::<i32>(&[1, 1024]).unwrap();
        assert_eq!(fused_rotation_applies(&ints, 1024, &gpu), Err(REASON_SHAPE));
    }

    /// E3: every rotation counts in the request's fused-primitive report by the route that ran —
    /// the kernel as `fused`, the unfused chain as `reference` with why (a forced oracle run, a
    /// CPU stream).
    #[test]
    fn every_rotation_route_is_tallied_for_the_request_report() {
        use crate::primitives::fused::{fused_tally, REASON_CPU_STREAM, REASON_FORCED_REFERENCE};
        let signs = random_signs(1024, 41);
        let scale = Array::from_slice(&[hadamard_scale(1024)], &[1]);
        let x = Array::zeros::<f32>(&[1, 2, 1024]).unwrap();
        let rotate = || block_hadamard(&x, &signs, &scale, 1024, false).unwrap();
        let ran = |f: &dyn Fn()| {
            let start = fused_tally();
            f();
            fused_tally().since(&start)
        };
        let fused = ran(&|| {
            mlx_rs::with_new_default_stream(Stream::gpu(), || {
                rotate();
            })
        });
        assert_eq!(
            (fused.fused, fused.reference, fused.label()),
            (1, 0, "fused")
        );
        let forced = ran(&|| {
            mlx_rs::with_new_default_stream(Stream::gpu(), || {
                with_unfused_rotation(|| {
                    rotate();
                })
            })
        });
        assert_eq!(forced.label(), "reference");
        assert_eq!(forced.reference_reason, Some(REASON_FORCED_REFERENCE));
        let cpu = ran(&|| {
            crate::primitives::kv_cache::testing::on_cpu(|| {
                rotate();
            })
        });
        assert_eq!((cpu.fused, cpu.reference), (0, 1));
        assert_eq!(cpu.reference_reason, Some(REASON_CPU_STREAM));
    }

    /// The route follows the stream ops are issued on, never the process default device: a
    /// task-local GPU stream inside a task-local CPU scope runs the fused kernel on that stream,
    /// bit-identical to the unfused chain, and once it closes the enclosing CPU scope takes the
    /// unfused chain again. Both scopes are task-local (switching the process default device races
    /// every parallel test in this binary — sc-24439).
    #[test]
    fn fused_rotation_runs_on_the_task_local_gpu_stream_inside_a_cpu_scope() {
        let width = 5120;
        let signs = random_signs(width, 32);
        let scale = Array::from_slice(&[hadamard_scale(1024)], &[1]);
        let input = |seed| {
            let key = mlx_rs::random::key(seed).unwrap();
            mlx_rs::random::normal::<f32>(&[1, 3, width][..], None, None, Some(&key))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        crate::primitives::kv_cache::testing::on_cpu(|| {
            mlx_rs::with_new_default_stream(Stream::gpu(), || {
                let x = input(31);
                for inverse in [false, true] {
                    let (fused, routes) = recording_rotation_routes(|| {
                        block_hadamard(&x, &signs, &scale, 1024, inverse).unwrap()
                    });
                    assert_eq!(routes, [RotationRoute::Fused], "inverse {inverse}");
                    let reference =
                        block_hadamard_unfused(&x, &signs, 1024, inverse, hadamard_scale(1024))
                            .unwrap();
                    assert!(bits(&fused) == bits(&reference), "inverse {inverse}");
                }
            });
            let x = input(33);
            let (_, routes) = recording_rotation_routes(|| {
                block_hadamard(&x, &signs, &scale, 1024, false).unwrap()
            });
            assert_eq!(
                routes,
                [RotationRoute::Unfused],
                "the enclosing CPU scope is restored"
            );
        });
        // The kernel dispatches on the stream it is handed, not the default one: MLX refuses a
        // custom Metal kernel on a CPU stream.
        let x = input(34);
        let on_cpu = fused_block_hadamard(&x, &signs, &scale, 1024, width, false, &Stream::cpu())
            .and_then(|y| Ok(y.eval()?));
        assert!(
            on_cpu.is_err(),
            "the kernel ignored the stream it was handed"
        );
    }

    /// sc-24446 (E5): with the fused rotation switched off (`MLX_LLM_FUSED_ROTATION`; here its
    /// thread-scoped layer) a GPU stream runs the unfused chain, bit-identical to the kernel.
    #[test]
    fn the_fused_rotation_switch_off_runs_the_unfused_chain_on_the_gpu() {
        let width = 2048;
        let signs = random_signs(width, 32);
        let scale = Array::from_slice(&[hadamard_scale(1024)], &[1]);
        mlx_rs::with_new_default_stream(Stream::gpu(), || {
            let key = mlx_rs::random::key(35).unwrap();
            let x = mlx_rs::random::normal::<f32>(&[1, 2, width][..], None, None, Some(&key))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let rotate = || block_hadamard(&x, &signs, &scale, 1024, false).unwrap();
            let (fused, on) = recording_rotation_routes(rotate);
            assert_eq!(on, [RotationRoute::Fused]);
            let (unfused, off) =
                crate::switches::FUSED_ROTATION.scoped(false, || recording_rotation_routes(rotate));
            assert_eq!(off, [RotationRoute::Unfused], "the switch is off");
            assert!(bits(&fused) == bits(&unfused));
        });
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn packed_linear_and_inverse_embedding_match_dense_operator_oracles() {
        let width = 128i32;
        let rows = 2i32;
        let signs = Array::from_slice(
            &(0..width)
                .map(|i| if i % 3 == 0 { -1.0f32 } else { 1.0 })
                .collect::<Vec<_>>(),
            &[width],
        );
        let mut words = vec![0u32; (rows * width / 16) as usize];
        for row in 0..rows as usize {
            for col in 0..width as usize {
                let code = ((row + col) % 3) as u32;
                words[row * (width as usize / 16) + col / 16] |= code << (2 * (col % 16));
            }
        }
        let scales = Array::from_slice(&[0.25f32, 0.5], &[rows, 1]);
        let biases = Array::from_slice(&[-0.25f32, -0.5], &[rows, 1]);
        let weight = Array::from_slice(&words, &[rows, width / 16]);
        let dense_rotated = dequantize(&weight, &scales, Some(&biases), GROUP_SIZE, BITS).unwrap();
        let x = Array::from_slice(
            &(0..width)
                .map(|i| (i as f32 - 50.0) / 64.0)
                .collect::<Vec<_>>(),
            &[1, width],
        );
        let rotated =
            block_hadamard_unfused(&x, &signs, width, false, hadamard_scale(width)).unwrap();
        let expected = linear(&rotated, &dense_rotated, None).unwrap();
        let packed = PrismLinear::new(
            "test",
            weight.clone(),
            scales.clone(),
            biases.clone(),
            signs.clone(),
            width,
        )
        .unwrap()
        .forward(&x)
        .unwrap();
        assert!(packed
            .all_close(&expected, 1e-4, 1e-4, None)
            .unwrap()
            .item::<bool>());

        let ids = Array::from_slice(&[1i32, 0], &[1, 2]);
        let embedding =
            PrismEmbedding::new("embedding", weight, scales, biases, signs.clone(), width)
                .unwrap()
                .forward(&ids)
                .unwrap();
        let gathered = dense_rotated
            .take_axis(Array::from_slice(&[1i32, 0], &[2]), 0)
            .unwrap()
            .reshape(&[1, 2, width])
            .unwrap();
        let expected =
            block_hadamard_unfused(&gathered, &signs, width, true, hadamard_scale(width)).unwrap();
        assert!(embedding
            .all_close(&expected, 1e-5, 1e-5, None)
            .unwrap()
            .item::<bool>());
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::packed_linear_and_inverse_embedding_match_dense_operator_oracles");
    }

    #[test]
    fn malformed_packed_metadata_fails_closed() {
        let weight = Array::from_slice(&[0u32; 8], &[1, 8]);
        let scales = Array::from_slice(&[1.0f32], &[1, 1]);
        let biases = Array::from_slice(&[-1.0f32], &[1, 1]);
        let bad_signs = Array::from_slice(&vec![1.0f32; 127], &[127]);
        let err = PrismLinear::new("bad", weight, scales, biases, bad_signs, 128).unwrap_err();
        assert!(err.to_string().contains("geometry mismatch"), "{err}");
    }
}
