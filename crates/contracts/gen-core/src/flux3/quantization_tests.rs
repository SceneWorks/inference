use super::*;

fn config(base: bool) -> DitConfig {
    DitConfig {
        hidden: 3072,
        heads: 24,
        depth: 5,
        depth_single: 28,
        mlp_ratio: 3.,
        axes_dim: vec![32; 4],
        theta: 10_000.,
        image_in_channels: 128,
        image_cond_in_channels: 128,
        text_in_channels: 20480,
        out_modes: vec!["image".into()],
        vec_in_dim: None,
        image_rope_scale: 0.5,
        flow_map: !base,
    }
}

#[test]
fn frozen_projection_inventories_are_exact_and_checkpoint_specific() {
    for base in [false, true] {
        let cfg = config(base);
        let dense = expected_tensors(&cfg, DitFormat::Dense).unwrap();
        let fp8 = expected_tensors(&cfg, DitFormat::Fp8Rowwise).unwrap();
        let nvfp4 = expected_tensors(&cfg, DitFormat::Nvfp4).unwrap();
        assert_eq!(dense.len(), if base { 357 } else { 359 });
        assert_eq!(fp8.len(), if base { 591 } else { 593 });
        assert_eq!(nvfp4.len(), if base { 951 } else { 953 });
        assert_eq!(
            fp8.values().filter(|v| v.dtype == Dtype::F8_E4M3).count(),
            234
        );
        assert_eq!(nvfp4.values().filter(|v| v.dtype == Dtype::U8).count(), 198);
        assert_eq!(dense.contains_key("ref_time_in.in_layer.weight"), !base);
        assert_eq!(fp8["txt_mode_blocks.2.q_proj.weight"].dtype, Dtype::BF16);
        assert_eq!(fp8["single_blocks.24.q_proj.weight"].dtype, Dtype::F8_E4M3);
        assert_eq!(nvfp4["single_blocks.24.q_proj.weight"].dtype, Dtype::BF16);
        assert_eq!(
            nvfp4["content_mode_blocks.image.0.q_proj.weight"].dtype,
            Dtype::BF16
        );
        assert_eq!(
            nvfp4["content_mode_blocks.image_cond.0.q_proj.weight"].dtype,
            Dtype::U8
        );
        assert_eq!(
            nvfp4["single_blocks.2.q_proj.weight_scale"].shape,
            vec![3072, 192]
        );
        assert_eq!(
            nvfp4["single_blocks.2.q_proj.input_scale"].shape,
            Vec::<usize>::new()
        );
    }
}

#[test]
fn preflight_refuses_missing_wrong_scales_extra_keys_and_model_identity() {
    let cfg = config(false);
    let expected = expected_tensors(&cfg, DitFormat::Nvfp4).unwrap();
    let headers: Vec<_> = expected
        .iter()
        .map(|(name, t)| SafetensorsTensorHeader {
            name: name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data_bytes: t.bytes().unwrap(),
        })
        .collect();
    validate_headers(&cfg, DitFormat::Nvfp4, &headers).unwrap();
    let mut bad = headers.clone();
    bad.retain(|v| v.name != "single_blocks.0.q_proj.input_scale");
    assert!(validate_headers(&cfg, DitFormat::Nvfp4, &bad).is_err());
    let mut bad = headers.clone();
    bad.iter_mut()
        .find(|v| v.name == "single_blocks.0.q_proj.weight_scale")
        .unwrap()
        .dtype = Dtype::F32;
    assert!(validate_headers(&cfg, DitFormat::Nvfp4, &bad).is_err());
    let mut bad = headers.clone();
    bad.push(headers[0].clone());
    assert!(validate_headers(&cfg, DitFormat::Nvfp4, &bad).is_err());
    assert!(validate_headers(&config(true), DitFormat::Nvfp4, &headers).is_err());
    assert!(validate_headers(&cfg, DitFormat::Fp8Rowwise, &headers).is_err());
}

#[test]
fn fp8_quantization_is_per_row_and_clamps_zero_scale() {
    let (codes, scales) = quantize_fp8_rows(&[1., -2., 100., -200., 0., 0.], 2).unwrap();
    assert_eq!(scales, vec![2. / 448., 200. / 448., 1e-12]);
    assert_eq!(&codes[..2], &codes[2..4]);
    assert_eq!(&codes[4..], &[0, 0]);
    let reconstructed = dequantize_fp8_rows(&codes, &scales, 2).unwrap();
    assert_eq!(reconstructed, vec![1., -2., 100., -200., 0., 0.]);
    assert!(quantize_fp8_rows(&[f32::NAN], 1).is_err());
    assert!(quantize_fp8_rows(&[1., 2., 3.], 2).is_err());
}

#[test]
fn nvfp4_static_scale_is_not_replaced_by_observed_amax() {
    let data = [
        1., -2., 3., -4., 0.5, -0.5, 6., -6., 1., 2., 3., 4., 5., 6., 0., 0.,
    ];
    let static_a = quantize_nvfp4_static(&data, 16, 0.25).unwrap();
    let static_b = quantize_nvfp4_static(&data, 16, 0.5).unwrap();
    assert_eq!(static_a.global_scale, 0.25);
    assert_eq!(static_b.global_scale, 0.5);
    assert_ne!(static_a.scales, static_b.scales);
    assert_ne!(static_a.global_scale, 6. / (448. * 6.));
    assert_eq!(
        static_a.packed,
        vec![0xc2, 0xe5, 0x91, 0xf7, 0x42, 0x65, 0x76, 0]
    );
    assert_eq!(static_a.scales[0], 0x48); // E4M3(4), divisor = 4 * .25 = 1.
    assert_eq!(static_b.scales[0], 0x40); // E4M3(2), same divisor, different static scale.
    assert_eq!(static_a.scales.len(), 128 * 4);
    assert!(quantize_nvfp4_static(&data, 16, f32::INFINITY).is_err());
    assert!(quantize_nvfp4_static(&data, 16, 0.).is_err());
}

#[test]
fn swizzle_and_nibble_order_are_explicit() {
    let mut row_major = vec![0u8; 129 * 5];
    for (i, v) in row_major.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let swizzled = swizzle_scales(&row_major, 129, 5).unwrap();
    assert_eq!(swizzled.len(), 256 * 8);
    // Exact FlashInfer 0.6.8.post1 quantization.cu: K-atoms are fastest. These
    // coordinates deliberately distinguish the legacy HOST column-major grid.
    for (r, c, offset) in [
        (0, 0, 0),
        (1, 0, 16),
        (31, 3, 499),
        (32, 0, 4),
        (127, 3, 511),
        (128, 0, 1024),
        (0, 4, 512),
    ] {
        assert_eq!(scale_offset(r, c, 8).unwrap(), offset);
        assert_eq!(swizzled[offset], row_major[r * 5 + c]);
    }
    for r in 0..129 {
        for c in 0..5 {
            assert_eq!(
                swizzled[scale_offset(r, c, 8).unwrap()],
                row_major[r * 5 + c]
            );
        }
    }
    let q = quantize_nvfp4_static(
        &[
            6., -6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        16,
        1.,
    )
    .unwrap();
    assert_eq!(q.packed[0] & 7, (q.packed[0] >> 4) & 7);
    assert_eq!(q.packed[0] & 8, 0);
    assert_ne!(q.packed[0] & 0x80, 0);
    assert!(swizzle_scales(&[1], usize::MAX, 2).is_err());
}

#[test]
fn nvfp4_quantizer_clamp_does_not_rewrite_source_gemm_alpha() {
    let data = [1.; 16];
    let tiny = quantize_nvfp4_static(&data, 16, 1e-12).unwrap();
    let clamped = quantize_nvfp4_static(&data, 16, 1e-8).unwrap();
    // Frozen nvfp4.py clamps ONLY the quantizer's reciprocal scale; alpha still
    // uses original input_scale * weight_scale_2. Both cases pack identically.
    assert_eq!(tiny.packed, clamped.packed);
    assert_eq!(tiny.scales, clamped.scales);
    assert_eq!(tiny.global_scale, 1e-12);
    assert_ne!(tiny.global_scale, clamped.global_scale);
    let extreme = quantize_nvfp4_static(&[f32::MAX; 16], 16, 1e-12).unwrap();
    assert!(extreme.packed.iter().all(|v| *v == 0x77));
    let zero = quantize_nvfp4_static(&[0.; 16], 16, 0.25).unwrap();
    assert!(zero.packed.iter().all(|v| *v == 0));
    assert!(zero.scales.iter().all(|v| *v == 0));
}

#[test]
fn format_detection_is_explicit_not_a_scale_dtype_guess() {
    assert_eq!(
        DitFormat::from_marker(&serde_json::Value::Null).unwrap(),
        DitFormat::Dense
    );
    assert_eq!(
        DitFormat::from_marker(&serde_json::json!("fp8r")).unwrap(),
        DitFormat::Fp8Rowwise
    );
    assert!(DitFormat::from_marker(&serde_json::json!("fp8")).is_err());
    assert!(DitFormat::from_marker(&serde_json::json!({"bits":4})).is_err());
    assert!(DitFormat::Nvfp4.check_device(QuantDevice::Mlx).is_err());
    assert!(DitFormat::Nvfp4
        .check_device(QuantDevice::Cuda { major: 9, minor: 0 })
        .is_err());
    DitFormat::Nvfp4
        .check_device(QuantDevice::Cuda {
            major: 12,
            minor: 0,
        })
        .unwrap();
}
