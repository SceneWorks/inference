use super::*;

fn cfg() -> DitConfig {
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
        flow_map: true,
    }
}
fn headers(storage: NativeDitStorage) -> Vec<SafetensorsTensorHeader> {
    storage
        .expected_tensors(&cfg())
        .unwrap()
        .into_iter()
        .map(|(name, t)| SafetensorsTensorHeader {
            name,
            dtype: t.dtype,
            data_bytes: t.bytes().unwrap(),
            shape: t.shape,
        })
        .collect()
}

#[test]
fn native_storage_marker_round_trips_without_source_format_or_bit_aliases() {
    let formats = [
        NativeDitStorage::Publisher(DitFormat::Dense),
        NativeDitStorage::Publisher(DitFormat::Fp8Rowwise),
        NativeDitStorage::Publisher(DitFormat::Nvfp4),
        NativeDitStorage::MlxAffine {
            bits: 4,
            group_size: 64,
        },
        NativeDitStorage::MlxAffine {
            bits: 8,
            group_size: 32,
        },
        NativeDitStorage::GgmlQ4_1,
        NativeDitStorage::GgmlQ8_0,
    ];
    for format in formats {
        assert_eq!(
            NativeDitStorage::from_marker(&format.marker().unwrap()).unwrap(),
            format
        );
    }
    assert_ne!(
        NativeDitStorage::GgmlQ8_0.key().unwrap(),
        NativeDitStorage::MlxAffine {
            bits: 8,
            group_size: 64
        }
        .key()
        .unwrap()
    );
    assert_ne!(
        NativeDitStorage::GgmlQ8_0.key().unwrap(),
        NativeDitStorage::Publisher(DitFormat::Fp8Rowwise)
            .key()
            .unwrap()
    );
    for bad in [
        serde_json::json!({"bits":4,"group_size":64}),
        serde_json::json!({"format":"mlx-affine","bits":8,"group_size":17,"algorithm_version":1}),
        serde_json::json!({"format":"ggml-q4_1","bits":8,"group_size":32,"algorithm_version":1}),
        serde_json::json!({"format":"ggml-q8_0","bits":8,"group_size":32,"algorithm_version":2}),
    ] {
        assert!(NativeDitStorage::from_marker(&bad).is_err());
    }
}

#[test]
fn explicit_storage_graph_entry_does_not_enable_legacy_dense_misreads() {
    let marker = NativeDitStorage::MlxAffine {
        bits: 4,
        group_size: 64,
    }
    .marker()
    .unwrap();
    let value = serde_json::json!({
        "format_version":1,"model":"flux-3-image-klein","quantization":marker,"torch_dtype":"bfloat16",
        "dit":{"hidden":3072,"heads":24,"depth":5,"depth_single":28,"mlp_ratio":3.0,"axes_dim":[32,32,32,32],"theta":10000,"in_channels":{"image":128,"image_cond":128,"text":20480},"out_modes":["image"],"vec_in_dim":null,"image_rope_scale":0.5,"flow_map":true},
        "text_encoder":{"path":"text_encoder","tokenizer":"text_encoder","layers":[4,8,12,16,20,24,28,32],"pad_multiple":80,"max_tokens":8192,"dtype":"bfloat16"},
        "vae":{"path":"ae.safetensors","dtype":"float32"},
        "sampler":{"steps":16,"guidance":null,"alpha":6.93,"size":[1024,1024]}
    });
    let parsed = super::super::Flux3Config::parse(&value.to_string()).unwrap();
    assert_eq!(parsed.quantization.as_deref(), Some("mlx-affine"));
    parsed.validate_native_storage_graph().unwrap();
    assert!(parsed.validate_supported().is_err());
    assert_eq!(parsed.sampler.steps, 16);
    assert_eq!(parsed.sampler.guidance, None);
    assert_eq!(parsed.text_encoder.window_max_length(), 4096);
}

#[test]
fn honest_backend_matrix_does_not_intersect_away_cuda_only_formats() {
    let cuda = ArtifactBackend::CandleCuda {
        major: 12,
        minor: 0,
    };
    for storage in [
        NativeDitStorage::Publisher(DitFormat::Fp8Rowwise),
        NativeDitStorage::Publisher(DitFormat::Nvfp4),
        NativeDitStorage::GgmlQ4_1,
        NativeDitStorage::GgmlQ8_0,
    ] {
        storage.check_backend(cuda).unwrap();
        assert!(storage.check_backend(ArtifactBackend::Mlx).is_err());
    }
    let affine = NativeDitStorage::MlxAffine {
        bits: 4,
        group_size: 64,
    };
    affine.check_backend(ArtifactBackend::Mlx).unwrap();
    assert!(affine.check_backend(cuda).is_err());
    NativeDitStorage::GgmlQ8_0
        .check_backend(ArtifactBackend::CandleCpu)
        .unwrap();
    assert!(NativeDitStorage::Publisher(DitFormat::Nvfp4)
        .check_backend(ArtifactBackend::CandleCuda { major: 9, minor: 0 })
        .is_err());
}

#[test]
fn each_storage_format_has_exact_physical_tensor_and_scale_inventory() {
    let ggml4 = NativeDitStorage::GgmlQ4_1.expected_tensors(&cfg()).unwrap();
    let ggml8 = NativeDitStorage::GgmlQ8_0.expected_tensors(&cfg()).unwrap();
    assert_eq!(ggml4.len(), 359);
    assert_eq!(ggml4.values().filter(|t| t.dtype == Dtype::U8).count(), 234);
    assert_eq!(
        ggml4["single_blocks.0.q_proj.weight"].shape,
        vec![3072, 1920]
    );
    assert_eq!(
        ggml8["single_blocks.0.q_proj.weight"].shape,
        vec![3072, 3264]
    );
    assert_eq!(ggml4["txt_in.weight"].dtype, Dtype::BF16);
    let affine = NativeDitStorage::MlxAffine {
        bits: 4,
        group_size: 64,
    }
    .expected_tensors(&cfg())
    .unwrap();
    assert_eq!(affine.len(), 827);
    assert_eq!(affine["single_blocks.0.q_proj.weight"].dtype, Dtype::U32);
    assert_eq!(
        affine["single_blocks.0.q_proj.weight"].shape,
        vec![3072, 384]
    );
    for key in [
        "single_blocks.0.q_proj.scales",
        "single_blocks.0.q_proj.biases",
    ] {
        assert_eq!(affine[key].shape, vec![3072, 48]);
        assert_eq!(affine[key].dtype, Dtype::BF16);
    }
}

#[test]
fn cross_format_missing_scales_and_wrong_block_width_fail_preflight() {
    let affine = NativeDitStorage::MlxAffine {
        bits: 8,
        group_size: 64,
    };
    let mut good = headers(affine);
    affine.validate_headers(&cfg(), &good).unwrap();
    assert!(NativeDitStorage::GgmlQ8_0
        .validate_headers(&cfg(), &good)
        .is_err());
    good.retain(|t| t.name != "single_blocks.0.q_proj.biases");
    assert!(affine.validate_headers(&cfg(), &good).is_err());
    let mut ggml = headers(NativeDitStorage::GgmlQ4_1);
    let weight = ggml
        .iter_mut()
        .find(|t| t.name == "single_blocks.0.q_proj.weight")
        .unwrap();
    weight.shape[1] += 1;
    assert!(NativeDitStorage::GgmlQ4_1
        .validate_headers(&cfg(), &ggml)
        .is_err());
}

#[test]
fn historically_valid_dense_f32_remains_dense_not_a_quantized_fallback() {
    let dense = NativeDitStorage::Publisher(DitFormat::Dense);
    let mut source = headers(dense);
    for tensor in &mut source {
        tensor.dtype = Dtype::F32;
        tensor.data_bytes = tensor.data_bytes / 2 * 4;
    }
    dense.validate_headers(&cfg(), &source).unwrap();
    assert!(NativeDitStorage::GgmlQ4_1
        .validate_headers(&cfg(), &source)
        .is_err());
    assert!(dense
        .validate_selection(Some(Quant::Q4), Precision::Bf16)
        .is_err());
    assert!(NativeDitStorage::Publisher(DitFormat::Fp8Rowwise)
        .validate_selection(Some(Quant::Q8), Precision::Bf16)
        .is_err());
    assert!(NativeDitStorage::Publisher(DitFormat::Fp8Rowwise)
        .validate_selection(Some(Quant::Fp8Rowwise), Precision::Fp32)
        .is_err());
}
