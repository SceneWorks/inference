//! Compact Prism/Bonsai affine-2 and GGUF ternary operators.
//!
//! Moved to [`candle_layers::prism`] (`SW-6957` / `SW-F335`) and re-exported here so all internal
//! decoders and external consumers remain 100% compatible.

#[cfg(all(test, feature = "cuda"))]
pub(crate) use candle_layers::prism::cuda;
pub use candle_layers::prism::{
    launch_elems, launch_grid_x, launch_u32, GdnRowMap, PrismPackedWeight, PrismRegistry,
    FUSED_ROTATION_MAX_BLOCK, PRISM_SRC,
};
#[cfg(test)]
pub(crate) use candle_layers::prism::{rotation_affine_factor, ROTATION_UPLOADS};

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use core_llm::{
        apply_hadamard_forward_in_place, apply_hadamard_inverse_in_place, GdnLayout,
        PrismHadamardMetadata, PrismPackedKind, PrismPackedMatrixRef, PrismTransformRole,
    };

    /// A one-row Prism weight `name` (no transform): what a registry needs to call a load Prism.
    pub(crate) fn one_row_weight(name: &str) -> PrismPackedWeight {
        mlx_weight(name, &[1; 128], PrismTransformRole::None)
    }

    fn metadata(name: &str, role: PrismTransformRole, signs: Vec<i8>) -> PrismHadamardMetadata {
        let mut signs_by_width = BTreeMap::new();
        signs_by_width.insert(signs.len(), signs);
        let mut forward_weight_names = BTreeSet::new();
        let mut inverse_weight_names = BTreeSet::new();
        match role {
            PrismTransformRole::Forward => {
                forward_weight_names.insert(name.into());
            }
            PrismTransformRole::Inverse => {
                inverse_weight_names.insert(name.into());
            }
            PrismTransformRole::None => {}
        }
        PrismHadamardMetadata {
            block_size: 128,
            signs_by_width,
            forward_weight_names,
            inverse_weight_names,
            gdn_v_grouped: true,
        }
    }

    fn mlx_weight(name: &str, codes: &[u8], role: PrismTransformRole) -> PrismPackedWeight {
        assert_eq!(codes.len(), 128);
        let mut words = vec![0u32; 8];
        for (lane, &code) in codes.iter().enumerate() {
            words[lane / 16] |= u32::from(code) << (2 * (lane % 16));
        }
        let device = Device::Cpu;
        PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words, (1, 8), &device).unwrap(),
            Tensor::from_vec(vec![0.5f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-0.5f32], (1, 1), &device).unwrap(),
            &metadata(
                name,
                role,
                (0..128).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect(),
            ),
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn mlx_affine2_matches_dense_reference_without_expansion() {
        let codes = (0..128).map(|i| (i % 3) as u8).collect::<Vec<_>>();
        let weight = mlx_weight("plain.weight", &codes, PrismTransformRole::None);
        let input = (0..128).map(|i| i as f32 / 64.0 - 1.0).collect::<Vec<_>>();
        let expected = input
            .iter()
            .zip(&codes)
            .map(|(&x, &code)| x * (f32::from(code) - 1.0) * 0.5)
            .sum::<f32>();
        let actual = weight
            .forward(&Tensor::from_vec(input, (1, 128), &Device::Cpu).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::mlx_affine2_matches_dense_reference_without_expansion");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn pq2_and_ptq1_match_decoded_dense_rows() {
        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let mut bytes = vec![0u8; kind.block_bytes()];
            let scale_at = if kind == PrismPackedKind::Pq2_0 {
                0
            } else {
                26
            };
            bytes[scale_at..scale_at + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            let codes = if kind == PrismPackedKind::Pq2_0 {
                &mut bytes[2..]
            } else {
                &mut bytes[..26]
            };
            for (i, byte) in codes.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(37);
            }
            let mut dense = vec![0f32; 128];
            PrismPackedMatrixRef::from_gguf(kind, &[128, 1], &bytes)
                .unwrap()
                .decode_row_into(0, &mut dense)
                .unwrap();
            let input = (0..128).map(|i| (i % 11) as f32 - 5.0).collect::<Vec<_>>();
            let expected = input.iter().zip(&dense).map(|(a, b)| a * b).sum::<f32>();
            let weight = PrismPackedWeight::from_gguf(
                "plain.weight",
                kind,
                &[128, 1],
                bytes,
                &metadata("plain.weight", PrismTransformRole::None, vec![1; 128]),
                None,
                None,
                &Device::Cpu,
            )
            .unwrap();
            let actual = weight
                .forward(&Tensor::from_vec(input, (1, 128), &Device::Cpu).unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()[0][0];
            assert!(
                (actual - expected).abs() < 1e-4,
                "{kind:?}: {actual} != {expected}"
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::pq2_and_ptq1_match_decoded_dense_rows");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn optional_ungrouped_gdn_reorder_precedes_sign_and_hadamard() {
        let name = "model.layers.0.linear_attn.ssm_out.weight";
        let codes = (0..128)
            .map(|i| if i % 7 == 0 { 2u8 } else { 1u8 })
            .collect::<Vec<_>>();
        let mut signs = vec![1i8; 128];
        signs[1] = -1;
        signs[65] = -1;
        let meta = metadata(name, PrismTransformRole::Forward, signs.clone());
        let device = Device::Cpu;
        let mut words = vec![0u32; 8];
        for (lane, &code) in codes.iter().enumerate() {
            words[lane / 16] |= u32::from(code) << (2 * (lane % 16));
        }
        let layout = GdnLayout {
            head_dim: 32,
            groups: 2,
            repetitions: 2,
        };
        let weight = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words, (1, 8), &device).unwrap(),
            Tensor::from_vec(vec![1f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-1f32], (1, 1), &device).unwrap(),
            &meta,
            None,
            Some(layout),
        )
        .unwrap();
        let input = (0..128).map(|i| i as f32 - 60.0).collect::<Vec<_>>();
        let mut transformed = input.clone();
        apply_hadamard_forward_in_place(&mut transformed, &signs, 128, Some(layout)).unwrap();
        let expected = transformed
            .iter()
            .zip(&codes)
            .map(|(&x, &code)| x * (f32::from(code) - 1.0))
            .sum::<f32>();
        let actual = weight
            .forward(&Tensor::from_vec(input, (1, 128), &device).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        assert!((actual - expected).abs() < 2e-3, "{actual} != {expected}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::optional_ungrouped_gdn_reorder_precedes_sign_and_hadamard");
    }

    /// sc-24444: the device-tensor rotation (the CUDA path) multiplies by a sign vector and
    /// gathers by a GDN index uploaded once per weight at load — repeated forwards upload
    /// nothing — and it agrees with the host rotation, forward (with the GDN reorder) and inverse.
    #[test]
    fn the_tensor_rotation_uploads_its_signs_once_at_load_never_per_forward() {
        let uploads = || ROTATION_UPLOADS.with(std::cell::Cell::get);
        let device = Device::Cpu;
        let name = "model.layers.0.linear_attn.ssm_out.weight";
        let signs = (0..128)
            .map(|i| if i % 3 == 0 { -1i8 } else { 1 })
            .collect::<Vec<_>>();
        let layout = GdnLayout {
            head_dim: 32,
            groups: 2,
            repetitions: 2,
        };
        let before = uploads();
        let forward = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::zeros((1, 8), DType::U32, &device).unwrap(),
            Tensor::from_vec(vec![1f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-1f32], (1, 1), &device).unwrap(),
            &metadata(name, PrismTransformRole::Forward, signs.clone()),
            None,
            Some(layout),
        )
        .unwrap();
        assert_eq!(uploads() - before, 2, "signs and the GDN gather, once each");
        let inverse = mlx_weight("embed.weight", &[1; 128], PrismTransformRole::Inverse);
        let at_load = uploads();

        let x = Tensor::from_vec(
            (0..3 * 128).map(|i| (i % 97) as f32 / 16.0 - 3.0).collect(),
            (3, 128),
            &device,
        )
        .unwrap();
        let host_forward = forward.transform_forward(&x).unwrap();
        let host_inverse = inverse.transform_inverse(&x).unwrap();
        for _ in 0..3 {
            let close = |a: &Tensor, b: &Tensor| {
                let d = (a - b)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(d < 1e-4, "tensor and host rotations differ by {d}");
            };
            close(
                &forward.transform_forward_tensor(&x).unwrap(),
                &host_forward,
            );
            close(
                &inverse.transform_inverse_tensor(&x).unwrap(),
                &host_inverse,
            );
        }
        assert_eq!(
            uploads(),
            at_load,
            "a forward uploaded a sign vector or index"
        );
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn published_grouped_ssm_out_does_not_permute_activation() {
        let name = "model.layers.0.linear_attn.ssm_out.weight";
        let codes = (0..128)
            .map(|i| if i % 7 == 0 { 2u8 } else { 1u8 })
            .collect::<Vec<_>>();
        let signs = (0..128)
            .map(|i| if i % 5 == 0 { -1i8 } else { 1 })
            .collect::<Vec<_>>();
        let mut words = vec![0u32; 8];
        for (lane, &code) in codes.iter().enumerate() {
            words[lane / 16] |= u32::from(code) << (2 * (lane % 16));
        }
        let device = Device::Cpu;
        let weight = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words, (1, 8), &device).unwrap(),
            Tensor::from_vec(vec![1f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-1f32], (1, 1), &device).unwrap(),
            &metadata(name, PrismTransformRole::Forward, signs.clone()),
            None,
            None,
        )
        .unwrap();
        let input = (0..128).map(|i| i as f32 - 60.0).collect::<Vec<_>>();
        let mut transformed = input.clone();
        apply_hadamard_forward_in_place(&mut transformed, &signs, 128, None).unwrap();
        let expected = transformed
            .iter()
            .zip(&codes)
            .map(|(&x, &code)| x * (f32::from(code) - 1.0))
            .sum::<f32>();
        let actual = weight
            .forward(&Tensor::from_vec(input, (1, 128), &device).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        assert!((actual - expected).abs() < 2e-3, "{actual} != {expected}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::published_grouped_ssm_out_does_not_permute_activation");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn inverse_embedding_applies_hadamard_before_signs() {
        let name = "model.embed_tokens.weight";
        let codes = (0..128)
            .map(|i| if i == 0 { 2 } else { 1 })
            .collect::<Vec<_>>();
        let weight = mlx_weight(name, &codes, PrismTransformRole::Inverse);
        let mut expected = codes
            .iter()
            .map(|&c| (f32::from(c) - 1.0) * 0.5)
            .collect::<Vec<_>>();
        let signs = (0..128)
            .map(|i| if i % 3 == 0 { -1 } else { 1 })
            .collect::<Vec<_>>();
        apply_hadamard_inverse_in_place(&mut expected, &signs, 128).unwrap();
        let actual = weight
            .embedding(&Tensor::from_vec(vec![0u32], (1,), &Device::Cpu).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "lane {i}: {actual} != {expected}"
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::inverse_embedding_applies_hadamard_before_signs");
    }

    #[test]
    fn malformed_affine_and_row_metadata_fail_closed() {
        let device = Device::Cpu;
        let meta = metadata("plain.weight", PrismTransformRole::None, vec![1; 128]);
        let error = PrismPackedWeight::from_mlx_affine2(
            "plain.weight",
            Tensor::zeros((1, 8), DType::U32, &device).unwrap(),
            Tensor::ones((1, 1), DType::F32, &device).unwrap(),
            &Tensor::zeros((1, 1), DType::F32, &device).unwrap(),
            &meta,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("biases exactly equal"));
        let error = PrismPackedWeight::from_gguf(
            "plain.weight",
            PrismPackedKind::Pq2_0,
            &[128, 1],
            vec![0; 34],
            &meta,
            Some(GdnRowMap {
                prefix: 0,
                groups: 1,
                repetitions: 2,
                unit: 1,
            }),
            None,
            &device,
        )
        .unwrap_err();
        assert!(error.to_string().contains("row-map geometry"));
    }

    #[test]
    fn runtime_cuda_source_requires_no_host_or_sdk_headers() {
        // Production NVRTC compilation intentionally supplies no include directories. Keep this
        // invariant test available on CPU CI; the CUDA oracle separately compiles and runs every
        // packed operator on the actual device.
        let source = PRISM_SRC.src;
        assert!(!source.lines().any(|line| {
            let directive = line
                .trim_start()
                .strip_prefix('#')
                .unwrap_or("")
                .trim_start();
            directive.starts_with("include")
        }));
    }

    /// sc-24440: the fused CUDA rotation is bit-identical to the op chain it replaces — forward
    /// (with and without the GDN gather; the chain's trailing f32 cast included) and inverse — for
    /// every served dtype and block size, zeros of both signs and wide magnitudes included; and a
    /// Prism forward through it equals the forward with the fused kernels switched off, the
    /// fused-primitive tally recording which ran.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn the_fused_rotation_is_bit_identical_to_the_op_chain() {
        let device = crate::device::new_cuda_for_test()
            .expect("CUDA is required for the fused rotation fixture");
        let bits = |t: &Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let cases: [(usize, usize, Option<GdnLayout>); 6] = [
            (128, 128, None),
            (256, 64, None),
            (
                512,
                128,
                Some(GdnLayout {
                    head_dim: 64,
                    groups: 2,
                    repetitions: 4,
                }),
            ),
            (2048, 1024, None),
            (8, 2, None),
            (4, 1, None),
        ];
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for (width, block, gdn) in cases {
                let rows = 3;
                let values = (0..rows * width)
                    .map(|i| match i % 17 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => 1e4 * (i as f32).sin(),
                        3 => 1e-4 * (i as f32).cos(),
                        _ => (i as f32 * 0.37).sin() * 3.0,
                    })
                    .collect::<Vec<_>>();
                let x = Tensor::from_vec(values, (rows, width), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let signs = (0..width)
                    .map(|i| if (i * 7) % 5 < 2 { -1i8 } else { 1 })
                    .collect::<Vec<_>>();
                let sign_t = upload_signs(&signs, &device).unwrap();
                let gather = gdn.map(|layout| gdn_gather_index(layout, &device).unwrap());
                let mul = rotation_affine_factor(block, dtype).unwrap();
                // Forward: the chain (`transform_forward_chain`'s ops), then the forward's cast.
                let mut chain = x.clone();
                if let Some(gather) = &gather {
                    chain = gdn_reorder_tensor(&chain, gather).unwrap();
                }
                let chain = fwht_tensor(&multiply_signs(&chain, &sign_t).unwrap(), block)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let fused =
                    cuda::rotate(&x, &sign_t, gather.as_ref(), block, mul, Rotation::Forward)
                        .unwrap();
                assert_eq!(fused.dtype(), DType::F32);
                assert_eq!(
                    bits(&fused),
                    bits(&chain),
                    "forward {dtype:?} width {width} block {block} gdn {}",
                    gdn.is_some()
                );
                // Inverse: Hadamard then signs, in the activation dtype.
                let chain = multiply_signs(&fwht_tensor(&x, block).unwrap(), &sign_t).unwrap();
                let fused = cuda::rotate(&x, &sign_t, None, block, mul, Rotation::Inverse).unwrap();
                assert_eq!(fused.dtype(), dtype);
                assert_eq!(
                    bits(&fused),
                    bits(&chain),
                    "inverse {dtype:?} width {width} block {block}"
                );
            }
        }

        // A whole Prism forward: fused vs the switch off.
        let name = "model.layers.0.mlp.down_proj.weight";
        let weight = {
            let host = mlx_weight(name, &[1; 128], PrismTransformRole::Forward);
            let PackedStorage::MlxAffine2 {
                host_words,
                host_scales,
                ..
            } = &host.storage
            else {
                panic!("an MLX affine-2 weight")
            };
            PrismPackedWeight::from_mlx_affine2(
                name,
                Tensor::from_slice(host_words.as_deref().unwrap(), (1, 8), &device).unwrap(),
                Tensor::from_slice(host_scales.as_deref().unwrap(), (1, 1), &device).unwrap(),
                &Tensor::from_vec(vec![-0.5f32], (1, 1), &device).unwrap(),
                &metadata(
                    name,
                    PrismTransformRole::Forward,
                    (0..128).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect(),
                ),
                None,
                None,
            )
            .unwrap()
        };
        let x = Tensor::from_vec(
            (0..2 * 128)
                .map(|i| (i as f32 * 0.11).cos())
                .collect::<Vec<_>>(),
            (1, 2, 128),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let before = super::super::fused::fused_tally();
        let fused = {
            let _on = super::super::fused::fused_policy_guard(Some(true));
            weight.forward(&x).unwrap()
        };
        assert!(super::super::fused::fused_tally().since(&before).fused >= 1);
        let chain = {
            let _off = super::super::fused::fused_policy_guard(Some(false));
            weight.forward(&x).unwrap()
        };
        assert_eq!(bits(&fused), bits(&chain), "Prism forward, fused vs chain");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::the_fused_rotation_is_bit_identical_to_the_op_chain");
    }

    /// sc-24440 review: the fused rotation's affine factor is the one the op chain multiplies by
    /// — candle's `affine` on a tensor of the activation dtype — for every block size and served
    /// dtype; an unserved dtype has none.
    #[test]
    fn the_rotation_affine_factor_is_the_chains() {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            for log2 in 0..=13 {
                let block = 1usize << log2;
                let chain = Tensor::ones(1, dtype, &Device::Cpu)
                    .unwrap()
                    .affine((block as f64).sqrt().recip(), 0.0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()[0];
                assert_eq!(
                    rotation_affine_factor(block, dtype).map(f32::to_bits),
                    Some(chain.to_bits()),
                    "{dtype:?} block {block}"
                );
            }
        }
        assert_eq!(rotation_affine_factor(128, DType::U32), None);
    }

    /// sc-24440 review: the fused CUDA rotation reads the signs and the GDN gather the weight
    /// uploaded at load (sc-24444's resident tensors) — with the fused path on, repeated forward
    /// and inverse rotations upload nothing — and still equals the op chain.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_fused_rotation_uploads_nothing_per_forward() {
        use super::super::fused::{fused_policy_guard, fused_tally};
        let device = crate::device::new_cuda_for_test()
            .expect("CUDA is required for the fused rotation fixture");
        let uploads = || ROTATION_UPLOADS.with(std::cell::Cell::get);
        let signs = (0..128)
            .map(|i| if i % 3 == 0 { -1i8 } else { 1 })
            .collect::<Vec<_>>();
        let weight = |name: &str, role, gdn| {
            PrismPackedWeight::from_mlx_affine2(
                name,
                Tensor::zeros((1, 8), DType::U32, &device).unwrap(),
                Tensor::from_vec(vec![1f32], (1, 1), &device).unwrap(),
                &Tensor::from_vec(vec![-1f32], (1, 1), &device).unwrap(),
                &metadata(name, role, signs.clone()),
                None,
                gdn,
            )
            .unwrap()
        };
        let before = uploads();
        let forward = weight(
            "model.layers.0.linear_attn.ssm_out.weight",
            PrismTransformRole::Forward,
            Some(GdnLayout {
                head_dim: 32,
                groups: 2,
                repetitions: 2,
            }),
        );
        let inverse = weight("embed.weight", PrismTransformRole::Inverse, None);
        let at_load = uploads();
        assert_eq!(
            at_load - before,
            3,
            "two sign vectors and one GDN gather, at load"
        );
        let x = Tensor::from_vec(
            (0..3 * 128).map(|i| (i % 97) as f32 / 16.0 - 3.0).collect(),
            (3, 128),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let bits = |t: &Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        let _on = fused_policy_guard(Some(true));
        let tally = fused_tally();
        for _ in 0..3 {
            let fused = forward.transform_forward(&x).unwrap();
            let chain = forward.transform_forward_chain(&x).unwrap();
            assert_eq!(bits(&fused), bits(&chain), "forward");
            let fused = inverse.transform_inverse(&x).unwrap();
            let chain = inverse.transform_inverse_tensor(&x).unwrap();
            assert_eq!(bits(&fused), bits(&chain), "inverse");
        }
        assert_eq!(
            fused_tally().since(&tally).fused,
            6,
            "every rotation ran fused"
        );
        assert_eq!(
            uploads(),
            at_load,
            "a fused rotation uploaded a sign vector or index"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn cuda_packed_operator_oracles_compile_and_execute_nvrtc() {
        fn rotate(mut values: Vec<f32>, signs: &[i8], inverse: bool) -> Vec<f32> {
            if !inverse {
                for (value, sign) in values.iter_mut().zip(signs) {
                    *value *= f32::from(*sign);
                }
            }
            let mut step = 1;
            while step < values.len() {
                for base in (0..values.len()).step_by(step * 2) {
                    for lane in 0..step {
                        let a = values[base + lane];
                        let b = values[base + step + lane];
                        values[base + lane] = a + b;
                        values[base + step + lane] = a - b;
                    }
                }
                step *= 2;
            }
            let scale = (values.len() as f32).sqrt().recip();
            for value in &mut values {
                *value *= scale;
            }
            if inverse {
                for (value, sign) in values.iter_mut().zip(signs) {
                    *value *= f32::from(*sign);
                }
            }
            values
        }
        fn gguf_rows(kind: PrismPackedKind, rows: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut bytes = Vec::new();
            let mut dense = Vec::new();
            for row in 0..rows {
                let scale = 0.125 * (row + 1) as f32;
                let mut block = vec![0u8; kind.block_bytes()];
                match kind {
                    PrismPackedKind::Pq2_0 => {
                        block[..2]
                            .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                        for (i, value) in block[2..].iter_mut().enumerate() {
                            *value = (i as u8).wrapping_mul(29).wrapping_add(row as u8 * 7);
                        }
                    }
                    PrismPackedKind::Ptq1_0 => {
                        for (i, value) in block[..26].iter_mut().enumerate() {
                            *value = (i as u8).wrapping_mul(37).wrapping_add(row as u8 * 11);
                        }
                        block[26..]
                            .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                    }
                }
                let mut decoded = vec![0.0f32; 128];
                match kind {
                    PrismPackedKind::Pq2_0 => {
                        for (lane, value) in decoded.iter_mut().enumerate() {
                            let code = (block[2 + lane / 4] >> (2 * (lane % 4))) & 3;
                            *value = (f32::from(code) - 1.0) * scale;
                        }
                    }
                    PrismPackedKind::Ptq1_0 => {
                        const POW3: [u16; 5] = [1, 3, 9, 27, 81];
                        let mut lane = 0;
                        for &(lo, hi, trits) in &[(0, 16, 5), (16, 24, 5), (24, 26, 4)] {
                            for &power in POW3.iter().take(trits) {
                                for &byte in &block[lo..hi] {
                                    let code =
                                        ((((u16::from(byte) * power) & 255) * 3) >> 8) as f32;
                                    decoded[lane] = (code - 1.0) * scale;
                                    lane += 1;
                                }
                            }
                        }
                    }
                }
                bytes.extend_from_slice(&block);
                dense.push(decoded);
            }
            (bytes, dense)
        }
        fn expected_matmul(
            inputs: &[Vec<f32>],
            dense: &[Vec<f32>],
            signs: &[i8],
            map: GdnRowMap,
        ) -> Vec<Vec<f32>> {
            inputs
                .iter()
                .map(|input| {
                    let rotated = rotate(input.clone(), signs, false);
                    (0..dense.len())
                        .map(|logical| {
                            let stored = map.source_row(logical).unwrap();
                            rotated.iter().zip(&dense[stored]).map(|(a, b)| a * b).sum()
                        })
                        .collect()
                })
                .collect()
        }
        fn assert_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
            for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                for (col, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() < 3e-3,
                        "[{row},{col}] {actual} != {expected}"
                    );
                }
            }
        }

        let device = crate::device::new_cuda_for_test()
            .expect("CUDA is required for the Prism NVRTC fixture");
        let rows = 4usize;
        let width = 128usize;
        let signs = (0..width)
            .map(|i| if i % 5 == 0 || i % 11 == 0 { -1i8 } else { 1 })
            .collect::<Vec<_>>();
        let map = GdnRowMap {
            prefix: 0,
            groups: 2,
            repetitions: 2,
            unit: 1,
        };
        let inputs = vec![
            (0..width)
                .map(|i| i as f32 / 71.0 - 0.8)
                .collect::<Vec<_>>(),
            (0..width)
                .map(|i| (i % 13) as f32 / 9.0 - 0.4)
                .collect::<Vec<_>>(),
        ];
        let flat_inputs = inputs.iter().flatten().copied().collect::<Vec<_>>();

        let name = "cuda.forward.weight";
        let mut words = vec![0u32; rows * width / 16];
        let scales = (0..rows)
            .map(|row| 0.125 * (row + 1) as f32)
            .collect::<Vec<_>>();
        let mut dense = vec![vec![0.0f32; width]; rows];
        for row in 0..rows {
            for col in 0..width {
                let code = ((row * 3 + col * 5) % 4) as u32;
                words[row * width / 16 + col / 16] |= code << (2 * (col % 16));
                dense[row][col] = (code as f32 - 1.0) * scales[row];
            }
        }
        let affine = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words.clone(), (rows, width / 16), &device).unwrap(),
            Tensor::from_vec(scales.clone(), (rows, 1), &device).unwrap(),
            &Tensor::from_vec(
                scales.iter().map(|s| -*s).collect::<Vec<_>>(),
                (rows, 1),
                &device,
            )
            .unwrap(),
            &metadata(name, PrismTransformRole::Forward, signs.clone()),
            Some(map),
            None,
        )
        .unwrap();
        let actual = affine
            .forward(&Tensor::from_vec(flat_inputs.clone(), (2, width), &device).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_close(&actual, &expected_matmul(&inputs, &dense, &signs, map));

        let embed_name = "cuda.embedding.weight";
        let embedding = PrismPackedWeight::from_mlx_affine2(
            embed_name,
            Tensor::from_vec(words, (rows, width / 16), &device).unwrap(),
            Tensor::from_vec(scales.clone(), (rows, 1), &device).unwrap(),
            &Tensor::from_vec(
                scales.iter().map(|s| -*s).collect::<Vec<_>>(),
                (rows, 1),
                &device,
            )
            .unwrap(),
            &metadata(embed_name, PrismTransformRole::Inverse, signs.clone()),
            None,
            None,
        )
        .unwrap();
        let actual = embedding
            .embedding(&Tensor::from_vec(vec![3u32, 1], (2,), &device).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected = vec![
            rotate(dense[3].clone(), &signs, true),
            rotate(dense[1].clone(), &signs, true),
        ];
        assert_close(&actual, &expected);

        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let (bytes, dense) = gguf_rows(kind, rows);
            let forward_name = format!("cuda.{kind:?}.forward.weight");
            let forward = PrismPackedWeight::from_gguf(
                &forward_name,
                kind,
                &[width, rows],
                bytes.clone(),
                &metadata(&forward_name, PrismTransformRole::Forward, signs.clone()),
                Some(map),
                None,
                &device,
            )
            .unwrap();
            let actual = forward
                .forward(&Tensor::from_vec(flat_inputs.clone(), (2, width), &device).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_close(&actual, &expected_matmul(&inputs, &dense, &signs, map));

            let inverse_name = format!("cuda.{kind:?}.embedding.weight");
            let inverse = PrismPackedWeight::from_gguf(
                &inverse_name,
                kind,
                &[width, rows],
                bytes,
                &metadata(&inverse_name, PrismTransformRole::Inverse, signs.clone()),
                None,
                None,
                &device,
            )
            .unwrap();
            let actual = inverse
                .embedding(&Tensor::from_vec(vec![2u32, 0], (2,), &device).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let expected = vec![
                rotate(dense[2].clone(), &signs, true),
                rotate(dense[0].clone(), &signs, true),
            ];
            assert_close(&actual, &expected);
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::cuda_packed_operator_oracles_compile_and_execute_nvrtc");
    }

    /// sc-24164 parity: the PTQ select chain that replaced the `pow3` table decodes exactly the
    /// values the table did, bit for bit, as does PQ2. Checked through the embedding kernels,
    /// which run the same per-value decode (`prism_ptq_value` / `prism_pq2_value`) the matmul
    /// kernels reduce over, for every byte value at every packed position and every lane. The
    /// scales are both positive and negative, so the sign of a decoded zero is checked too. The
    /// reference is the kernel's own formula evaluated on the host.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn cuda_gguf_decode_is_bit_exact_for_every_packed_byte() {
        const POW3: [u32; 5] = [1, 3, 9, 27, 81];
        let device = crate::device::new_cuda_for_test().expect("cuda device");
        let (width, rows) = (128usize, 256usize);
        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let block_bytes = kind.block_bytes();
            let (data, scale_at) = match kind {
                PrismPackedKind::Pq2_0 => (2..block_bytes, 0),
                PrismPackedKind::Ptq1_0 => (0..26, 26),
            };
            let mut bytes = vec![0u8; rows * block_bytes];
            let mut expected = Vec::with_capacity(rows * width);
            for row in 0..rows {
                let block = &mut bytes[row * block_bytes..(row + 1) * block_bytes];
                // Across the 256 rows, every data position takes every byte value.
                for (position, byte) in block[data.clone()].iter_mut().enumerate() {
                    *byte = (row + 37 * position) as u8;
                }
                let magnitude = 0.0625 * (1 + row % 7) as f32;
                let scale = half::f16::from_f32(if row % 2 == 0 { magnitude } else { -magnitude });
                block[scale_at..scale_at + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                let scale = scale.to_f32();
                for lane in 0..width {
                    let code = match kind {
                        PrismPackedKind::Pq2_0 => {
                            u32::from((block[2 + lane / 4] >> (2 * (lane % 4))) & 3)
                        }
                        PrismPackedKind::Ptq1_0 => {
                            let (byte_at, trit) = match lane {
                                0..80 => (lane % 16, lane / 16),
                                80..120 => (16 + (lane - 80) % 8, (lane - 80) / 8),
                                _ => (24 + (lane - 120) % 2, (lane - 120) / 2),
                            };
                            (((u32::from(block[byte_at]) * POW3[trit]) & 255) * 3) >> 8
                        }
                    };
                    expected.push(((code as f32 - 1.0) * scale).to_bits());
                }
            }
            let name = format!("cuda.{kind:?}.decode.weight");
            let weight = PrismPackedWeight::from_gguf(
                &name,
                kind,
                &[width, rows],
                bytes,
                &metadata(&name, PrismTransformRole::Inverse, vec![1; width]),
                None,
                None,
                &device,
            )
            .unwrap();
            let ids = Tensor::arange(0u32, rows as u32, &device).unwrap();
            // The raw kernel decode, before the embedding's inverse transform.
            let decoded = cuda::embedding(&weight, &ids)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(decoded.len(), expected.len());
            let mismatches = decoded
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (got, want))| got != want)
                .map(|(at, (got, want))| (at / width, at % width, *got, *want))
                .take(8)
                .collect::<Vec<_>>();
            assert!(
                mismatches.is_empty(),
                "{kind:?} decode differs (row, lane, got bits, want bits): {mismatches:?}"
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::cuda_gguf_decode_is_bit_exact_for_every_packed_byte");
    }

    /// sc-24164 timing evidence (a bench, not a gate): `prism_ptq_matmul_f32` milliseconds per
    /// token on a Bonsai-sized 17408x5120 PTQ matrix over a 64-token prefill.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "sc-24164 PTQ matmul timing; needs CUDA"]
    fn cuda_ptq_matmul_timing() {
        let device = crate::device::new_cuda_for_test().expect("cuda device");
        let (rows, width, tokens) = (17_408usize, 5_120usize, 64usize);
        let kind = PrismPackedKind::Ptq1_0;
        let mut bytes = vec![0u8; rows * width / 128 * kind.block_bytes()];
        for (block, chunk) in bytes.chunks_exact_mut(kind.block_bytes()).enumerate() {
            for (position, byte) in chunk[..26].iter_mut().enumerate() {
                *byte = (block * 7 + position * 37) as u8;
            }
            chunk[26..].copy_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        }
        let name = "cuda.timing.weight";
        let weight = PrismPackedWeight::from_gguf(
            name,
            kind,
            &[width, rows],
            bytes,
            &metadata(name, PrismTransformRole::None, vec![1; width]),
            None,
            None,
            &device,
        )
        .unwrap();
        let x = Tensor::arange(0f32, (tokens * width) as f32, &device)
            .unwrap()
            .affine(1e-4, -0.5)
            .unwrap()
            .reshape((tokens, width))
            .unwrap();
        let run = || {
            cuda::matmul(&weight, &x).unwrap();
            device.synchronize().unwrap();
        };
        run();
        let iterations = 10;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            run();
        }
        let per_token_ms = started.elapsed().as_secs_f64() * 1e3 / (iterations * tokens) as f64;
        eprintln!("[prism-ptq-timing] {rows}x{width}, {tokens} tokens: {per_token_ms:.3} ms/token");

        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED primitives::prism::tests::cuda_ptq_matmul_timing"
        );
    }
}
