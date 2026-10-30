use super::*;
use gen_core::{AdapterSpec, Image, MemoryStrategySupport};

fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize])]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for &(name, dtype, shape) in tensors {
        let width = match dtype {
            "F32" | "U32" => 4,
            "F16" | "BF16" => 2,
            _ => panic!("dtype"),
        };
        let bytes = shape.iter().product::<usize>() * width;
        header.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes as u64],
            }),
        );
        data.resize(data.len() + bytes, name.len() as u8);
        offset += bytes as u64;
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

/// Write a structurally real converted `t5_encoder.safetensors` (SC-22667).
///
/// The MLX loader packs exactly the seven bias-less projections of each block and leaves the
/// token embedding, the per-block norms and position-bias tables and the final norm dense, so a
/// fixture carrying a single tensor named `weight` cannot exercise
/// [`mlx_text_encoder_bytes`] at all — every arm it distinguishes needs its own key.
/// `dim` is a whole number of MLX affine groups, as the real encoder's is.
fn write_umt5_encoder(path: &Path, blocks: usize, dim: usize) {
    let square = vec![dim, dim];
    let vector = vec![dim];
    let mut owned: Vec<(String, &str, Vec<usize>)> = vec![
        (
            "token_embedding.weight".to_owned(),
            "BF16",
            vec![dim * 4, dim],
        ),
        ("norm.weight".to_owned(), "BF16", vector.clone()),
    ];
    for block in 0..blocks {
        for leaf in [
            "attn.q",
            "attn.k",
            "attn.v",
            "attn.o",
            "ffn.gate_proj",
            "ffn.fc1",
            "ffn.fc2",
        ] {
            owned.push((
                format!("blocks.{block}.{leaf}.weight"),
                "BF16",
                square.clone(),
            ));
        }
        for leaf in ["norm1.weight", "norm2.weight"] {
            owned.push((format!("blocks.{block}.{leaf}"), "BF16", vector.clone()));
        }
        owned.push((
            format!("blocks.{block}.pos_embedding.embedding.weight"),
            "BF16",
            vec![32, 8],
        ));
    }
    let tensors = owned
        .iter()
        .map(|(name, dtype, shape)| (name.as_str(), *dtype, shape.as_slice()))
        .collect::<Vec<_>>();
    write_safetensors(path, &tensors);
}

fn mlx_fixture(route: WanI2vRoute, quant: Option<Quant>) -> (tempfile::TempDir, LoadSpec) {
    let tmp = tempfile::tempdir().unwrap();
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    };
    let (repository, revision) = repository_policy(WanI2vBackend::Mlx, route, tier);
    let tier_name = match quant {
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        None => "bf16",
        _ => unreachable!(),
    };
    let root = tmp
        .path()
        .join(repo_dir(repository))
        .join("snapshots")
        .join(fixture_revision(revision))
        .join(tier_name);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        format!(
            "{{\"route\":\"{}\",\"tier\":\"{tier_name}\"}}",
            route.provider_id()
        ),
    )
    .unwrap();
    std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
    let files: &[(&str, usize)] = match route {
        WanI2vRoute::Ti2v5b => &[
            ("model.safetensors", 11),
            ("t5_encoder.safetensors", 7),
            ("vae.safetensors", 5),
        ],
        WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => &[
            ("high_noise_model.safetensors", 13),
            ("low_noise_model.safetensors", 17),
            ("t5_encoder.safetensors", 7),
            ("vae.safetensors", 5),
        ],
        WanI2vRoute::Vace => panic!("use a dedicated VACE fixture"),
        WanI2vRoute::VaceFun => panic!("use a dedicated VACE-Fun fixture"),
    };
    for &(name, logical) in files {
        if quant.is_some() && name.contains("model") {
            let packed_columns = if quant == Some(Quant::Q4) { 8 } else { 16 };
            write_safetensors(
                &root.join(name),
                &[
                    ("proj.weight", "U32", &[logical, packed_columns]),
                    ("proj.scales", "BF16", &[logical, 1]),
                    ("proj.biases", "BF16", &[logical, 1]),
                ],
            );
        } else if name == "t5_encoder.safetensors" {
            write_umt5_encoder(&root.join(name), logical, 64);
        } else {
            write_safetensors(&root.join(name), &[("weight", "BF16", &[logical, 64])]);
        }
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(route.provider_id());
    spec.quantize = quant;
    prepare_load_spec(&mut spec, WanI2vBackend::Mlx, route.provider_id()).unwrap();
    (tmp, spec)
}

fn candle_fixture(route: WanI2vRoute, quant: Option<Quant>) -> (tempfile::TempDir, LoadSpec) {
    let tmp = tempfile::tempdir().unwrap();
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    };
    let (repository, revision) = repository_policy(WanI2vBackend::Candle, route, tier);
    let mut root = tmp
        .path()
        .join(repo_dir(repository))
        .join("snapshots")
        .join(fixture_revision(revision));
    if let Some(quant) = quant {
        root = root.join(if quant == Quant::Q4 { "q4" } else { "q8" });
    }
    std::fs::create_dir_all(root.join("tokenizer")).unwrap();
    std::fs::write(root.join("model_index.json"), "{}").unwrap();
    std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
    let components: &[&str] = match route {
        WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
        WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => {
            &["text_encoder", "transformer", "transformer_2", "vae"]
        }
        WanI2vRoute::Vace => panic!("use a dedicated VACE fixture"),
        WanI2vRoute::VaceFun => panic!("use a dedicated VACE-Fun fixture"),
    };
    for (index, component) in components.iter().enumerate() {
        std::fs::create_dir_all(root.join(component)).unwrap();
        std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
        if let Some(quant) = quant.filter(|_| component.starts_with("transformer")) {
            let packed_columns = if quant == Quant::Q4 { 8 } else { 16 };
            write_safetensors(
                &root.join(component).join("model.safetensors"),
                &[
                    ("proj.weight", "U32", &[index + 2, packed_columns]),
                    ("proj.scales", "F32", &[index + 2, 1]),
                    ("proj.biases", "F32", &[index + 2, 1]),
                ],
            );
            std::fs::write(
                root.join(component).join("quantize_config.json"),
                format!(
                    "{{\"bits\":{},\"quantization\":{{\"group_size\":64}}}}",
                    quant.bits()
                ),
            )
            .unwrap();
        } else {
            let shape = if component.starts_with("transformer") {
                vec![index + 2, 64]
            } else {
                vec![index + 2, index + 3]
            };
            write_safetensors(
                &root.join(component).join("model.safetensors"),
                &[("weight", "F32", &shape)],
            );
        }
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(route.provider_id());
    spec.quantize = quant;
    prepare_load_spec(&mut spec, WanI2vBackend::Candle, route.provider_id()).unwrap();
    (tmp, spec)
}

fn vace_route_fixture(
    backend: WanI2vBackend,
    quant: Option<Quant>,
    route: WanI2vRoute,
) -> (tempfile::TempDir, LoadSpec) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("vace");
    match backend {
        WanI2vBackend::Mlx => {
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
            if route == WanI2vRoute::VaceFun {
                std::fs::create_dir_all(root.join("transformer_2")).unwrap();
                std::fs::write(root.join("transformer_2/config.json"), "{}").unwrap();
                write_safetensors(
                    &root.join("transformer_2/model.safetensors"),
                    &[("blocks.0.self_attn.to_q.weight", "BF16", &[9, 8])],
                );
            }
            std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
            write_safetensors(
                &root.join("transformer/model.safetensors"),
                &[("blocks.0.self_attn.to_q.weight", "BF16", &[8, 8])],
            );
            write_umt5_encoder(&root.join("t5_encoder.safetensors"), 1, 64);
            write_safetensors(
                &root.join("vae.safetensors"),
                &[("weight", "BF16", &[4, 4])],
            );
        }
        WanI2vBackend::Candle => {
            for component in ["transformer", "text_encoder", "vae"]
                .into_iter()
                .chain((route == WanI2vRoute::VaceFun).then_some("transformer_2"))
            {
                std::fs::create_dir_all(root.join(component)).unwrap();
                std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
                write_safetensors(
                    &root.join(component).join("model.safetensors"),
                    &[("weight", "BF16", &[4, 4])],
                );
            }
            std::fs::create_dir_all(root.join("tokenizer")).unwrap();
            std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
        }
    }
    let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(route.provider_id());
    spec.quantize = quant;
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => unreachable!(),
    };
    if route == WanI2vRoute::VaceFun {
        ensure_wan_vace_fun_source_receipt(root, backend)
    } else {
        ensure_wan_vace_source_receipt(root, backend)
    }
    .unwrap();
    prepare_load_spec(&mut spec, backend, route.provider_id()).unwrap();
    (tmp, spec)
}

fn vace_fixture(backend: WanI2vBackend, quant: Option<Quant>) -> (tempfile::TempDir, LoadSpec) {
    vace_route_fixture(backend, quant, WanI2vRoute::Vace)
}

fn vace_fun_fixture(backend: WanI2vBackend, quant: Option<Quant>) -> (tempfile::TempDir, LoadSpec) {
    vace_route_fixture(backend, quant, WanI2vRoute::VaceFun)
}

fn vace_extend_request() -> GenerationRequest {
    let image = Image {
        width: 832,
        height: 480,
        pixels: vec![17; 832 * 480 * 3],
    };
    GenerationRequest {
        prompt: "continue the source motion".to_owned(),
        width: 832,
        height: 480,
        count: 1,
        seed: Some(29),
        steps: Some(50),
        guidance: Some(5.0),
        frames: Some(45),
        fps: Some(16),
        video_mode: Some("extend_clip".to_owned()),
        control_scale: Some(1.0),
        conditioning: vec![Conditioning::ControlClip {
            frames: vec![image.clone(); 45],
            mask: vec![image; 45],
            masking_strength: 1.0,
            start_frame: 0,
            mode: gen_core::ReplacementMode::default(),
        }],
        ..Default::default()
    }
}

fn ti2v_extend_request() -> GenerationRequest {
    GenerationRequest {
        prompt: "continue from the exact source boundary".to_owned(),
        negative_prompt: Some("camera cut".to_owned()),
        width: 832,
        height: 480,
        count: 1,
        seed: Some(31),
        steps: Some(20),
        guidance: Some(5.0),
        frames: Some(61),
        fps: Some(16),
        video_mode: Some("extend_clip".to_owned()),
        conditioning: vec![Conditioning::Keyframe {
            image: Image {
                width: 832,
                height: 480,
                pixels: vec![23; 832 * 480 * 3],
            },
            frame_idx: 0,
            strength: 0.75,
        }],
        ..Default::default()
    }
}

fn bridge_keyframes_request() -> GenerationRequest {
    let mut request = ti2v_extend_request();
    request.prompt = "bridge the two clip boundaries".to_owned();
    request.video_mode = Some("video_bridge".to_owned());
    request.conditioning = vec![
        Conditioning::Keyframe {
            image: Image {
                width: 832,
                height: 480,
                pixels: vec![31; 832 * 480 * 3],
            },
            frame_idx: 0,
            strength: 0.25,
        },
        Conditioning::Keyframe {
            image: Image {
                width: 832,
                height: 480,
                pixels: vec![73; 832 * 480 * 3],
            },
            frame_idx: -1,
            strength: 0.75,
        },
    ];
    request
}

fn vace_bridge_request() -> GenerationRequest {
    let mut request = vace_extend_request();
    request.prompt = "bridge the two source clips through a neutral gap".to_owned();
    request.video_mode = Some("video_bridge".to_owned());
    if let Conditioning::ControlClip { frames, mask, .. } = &mut request.conditioning[0] {
        frames[..1].fill(Image {
            width: 832,
            height: 480,
            pixels: vec![11; 832 * 480 * 3],
        });
        // This is the exact mid-gray that SceneWorks' production
        // `build_extend_bridge_vace_conditioning` writes for its generated span.  Do not use
        // black here: black is reserved for the anchor-mask polarity.
        frames[1..42].fill(Image {
            width: 832,
            height: 480,
            pixels: vec![WAN_VACE_BRIDGE_NEUTRAL_RGB; 832 * 480 * 3],
        });
        frames[42..].fill(Image {
            width: 832,
            height: 480,
            pixels: vec![89; 832 * 480 * 3],
        });
        mask[..1].fill(Image {
            width: 832,
            height: 480,
            pixels: vec![0; 832 * 480 * 3],
        });
        mask[1..42].fill(Image {
            width: 832,
            height: 480,
            pixels: vec![255; 832 * 480 * 3],
        });
        mask[42..].fill(Image {
            width: 832,
            height: 480,
            pixels: vec![0; 832 * 480 * 3],
        });
    }
    request
}

fn replace_person_request(references: usize, mode: gen_core::ReplacementMode) -> GenerationRequest {
    let frame = Image {
        width: 832,
        height: 480,
        pixels: vec![37; 832 * 480 * 3],
    };
    let mask = Image {
        width: 832,
        height: 480,
        pixels: vec![255; 832 * 480 * 3],
    };
    let mut conditioning = vec![Conditioning::ControlClip {
        frames: vec![frame; 45],
        mask: vec![mask; 45],
        masking_strength: 0.625,
        start_frame: 0,
        mode,
    }];
    conditioning.extend((0..references).map(|ordinal| Conditioning::Reference {
        image: Image {
            width: 16,
            height: 16,
            pixels: vec![ordinal as u8 + 1; 16 * 16 * 3],
        },
        strength: None,
    }));
    GenerationRequest {
        prompt: "replace the tracked person".to_owned(),
        negative_prompt: Some("identity drift".to_owned()),
        width: 832,
        height: 480,
        count: 1,
        seed: Some(0x1234_5678),
        steps: Some(40),
        guidance: Some(5.0),
        frames: Some(45),
        fps: Some(16),
        video_mode: Some(format!("replace_person@{}", "a".repeat(64))),
        control_scale: Some(0.75),
        conditioning,
        ..Default::default()
    }
}

#[test]
fn replace_person_seals_vace_and_vace_fun_routes_rungs_tiers_and_full_carrier() {
    for (route, backend, quant) in [
        (WanI2vRoute::Vace, WanI2vBackend::Mlx, Some(Quant::Q4)),
        (WanI2vRoute::Vace, WanI2vBackend::Mlx, Some(Quant::Q8)),
        (WanI2vRoute::Vace, WanI2vBackend::Mlx, None),
        (WanI2vRoute::Vace, WanI2vBackend::Candle, None),
        (WanI2vRoute::VaceFun, WanI2vBackend::Mlx, Some(Quant::Q4)),
        (WanI2vRoute::VaceFun, WanI2vBackend::Mlx, Some(Quant::Q8)),
        (WanI2vRoute::VaceFun, WanI2vBackend::Mlx, None),
        (WanI2vRoute::VaceFun, WanI2vBackend::Candle, None),
    ] {
        let (_tmp, spec) = vace_route_fixture(backend, quant, route);
        let prepared = PreparedWanI2vMemory::prepare(&spec, backend, route.provider_id()).unwrap();
        assert_eq!(prepared.tier.quant, quant);
        assert_eq!(prepared.route, route);
        assert_eq!(prepared.backend, backend);
        assert_eq!(prepared.artifact_identity.len(), 64);
        assert_eq!(prepared.revision.len(), 64);
        for replacement in [
            gen_core::ReplacementMode::FaceOnly,
            gen_core::ReplacementMode::FullPersonKeepOutfit,
            gen_core::ReplacementMode::FullPersonReplaceOutfit,
        ] {
            for references in 1..=4 {
                let mut request = replace_person_request(references, replacement);
                assert_eq!(
                    validate_request(&prepared, &request).unwrap(),
                    WanPublicMode::ReplacePerson
                );
                assert_eq!(
                    geometry_from_request(&request).reference_count,
                    references as u32
                );
                let contract = request_contract(&prepared, &request).unwrap();
                for strategy in MemoryStrategy::ALL {
                    let expected = matches!(
                        strategy,
                        MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
                    ) || (route == WanI2vRoute::VaceFun
                        && strategy == MemoryStrategy::StagedResidency);
                    assert_eq!(
                        contract.capability(strategy).unwrap().support,
                        if expected {
                            MemoryStrategySupport::Implemented
                        } else {
                            MemoryStrategySupport::Missing
                        }
                    );
                    if expected {
                        let selected = selection(&prepared, strategy);
                        request.memory = contract.generation_memory(&selected);
                        let evidence = request_evidence_revision(&prepared, &request).unwrap();
                        let run_context = context(&prepared, &request, selected, evidence);
                        validate_context(&prepared, &run_context).unwrap();
                    }
                }
            }
        }
    }

    let (_tmp, mut packed_candle) = vace_fun_fixture(WanI2vBackend::Candle, None);
    packed_candle.quantize = Some(Quant::Q4);
    assert!(prepare_load_spec(
        &mut packed_candle,
        WanI2vBackend::Candle,
        WanI2vRoute::VaceFun.provider_id()
    )
    .is_err());
}

#[test]
fn replace_person_receipt_rejects_crossed_public_and_physical_identity() {
    let (_tmp, spec) = vace_fun_fixture(WanI2vBackend::Mlx, Some(Quant::Q4));
    let prepared = PreparedWanI2vMemory::prepare(
        &spec,
        WanI2vBackend::Mlx,
        WanI2vRoute::VaceFun.provider_id(),
    )
    .unwrap();
    let mut request = replace_person_request(2, gen_core::ReplacementMode::FaceOnly);
    let resident = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: prepared.tier,
    };
    request.memory = request_contract(&prepared, &request)
        .unwrap()
        .generation_memory(&resident);
    let receipt = request_evidence_revision(&prepared, &request).unwrap();

    request.conditioning.swap(1, 2);
    assert_ne!(
        request_evidence_revision(&prepared, &request).unwrap(),
        receipt
    );
    request.conditioning.swap(1, 2);
    if let Conditioning::ControlClip {
        mask,
        masking_strength,
        mode,
        ..
    } = &mut request.conditioning[0]
    {
        mask[0].pixels[0] ^= 1;
        *masking_strength = 0.5;
        *mode = gen_core::ReplacementMode::FullPersonKeepOutfit;
    }
    assert_ne!(
        request_evidence_revision(&prepared, &request).unwrap(),
        receipt
    );
    request.video_mode = Some(format!("replace_person@{}", "b".repeat(64)));
    assert_ne!(
        request_evidence_revision(&prepared, &request).unwrap(),
        receipt
    );

    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => unreachable!(),
    };
    assert!(ensure_wan_vace_fun_source_receipt(root, WanI2vBackend::Candle).is_err());
    assert!(PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan_vace").is_err());
}

#[test]
fn vace_fun_seals_exact_ordered_shared_and_expert_adapters_on_both_backends() {
    for (backend, quant) in [
        (WanI2vBackend::Mlx, Some(Quant::Q4)),
        (WanI2vBackend::Candle, None),
    ] {
        let (tmp, fixture) = vace_fun_fixture(backend, quant);
        let shared = tmp.path().join("shared-adapter.safetensors");
        let high = tmp.path().join("high-adapter.safetensors");
        let low = tmp.path().join("low-adapter.safetensors");
        for path in [&shared, &high, &low] {
            write_safetensors(path, &[("lora_A.weight", "F32", &[2, 3])]);
        }

        let mut ordered = LoadSpec::new(fixture.weights.clone())
            .with_resolved_route(WanI2vRoute::VaceFun.provider_id());
        ordered.quantize = quant;
        ordered.adapters = vec![
            AdapterSpec::new(shared.clone(), 0.5, AdapterKind::Lora),
            AdapterSpec::new(high.clone(), 0.75, AdapterKind::Lora)
                .with_moe_expert(MoeExpert::High),
            AdapterSpec::new(low.clone(), 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
        ];
        prepare_load_spec(&mut ordered, backend, WanI2vRoute::VaceFun.provider_id()).unwrap();
        let prepared =
            PreparedWanI2vMemory::prepare(&ordered, backend, WanI2vRoute::VaceFun.provider_id())
                .unwrap();
        assert_eq!(
            prepared
                .adapters
                .iter()
                .map(|adapter| adapter.ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            prepared
                .adapters
                .iter()
                .map(|adapter| adapter.expert)
                .collect::<Vec<_>>(),
            vec!["shared", "high", "low"]
        );
        assert_eq!(prepared.adapters[0].scale_bits, 0.5_f32.to_bits());
        assert_eq!(prepared.adapters[1].scale_bits, 0.75_f32.to_bits());
        assert_eq!(prepared.adapters[2].scale_bits, 1.0_f32.to_bits());

        let mut reversed = LoadSpec::new(fixture.weights.clone())
            .with_resolved_route(WanI2vRoute::VaceFun.provider_id());
        reversed.quantize = quant;
        reversed.adapters = ordered.adapters.iter().cloned().rev().collect();
        prepare_load_spec(&mut reversed, backend, WanI2vRoute::VaceFun.provider_id()).unwrap();
        let reversed =
            PreparedWanI2vMemory::prepare(&reversed, backend, WanI2vRoute::VaceFun.provider_id())
                .unwrap();
        assert_ne!(prepared.adapter_identity, reversed.adapter_identity);
        assert_ne!(prepared.artifact_identity, reversed.artifact_identity);
        assert_eq!(prepared.revision, reversed.revision);
    }
}

#[test]
fn vace_bridge_admission_requires_the_production_mid_gray_generated_gap() {
    let (_tmp, spec) = vace_fixture(WanI2vBackend::Mlx, None);
    let prepared = PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan_vace").unwrap();
    let request = vace_bridge_request();
    assert_eq!(
        validate_request(&prepared, &request).unwrap(),
        WanPublicMode::VideoBridge,
        "the production builder's raw RGB 128 gap must be admitted before normalization"
    );

    let mut black_gap = request;
    if let Conditioning::ControlClip { frames, .. } = &mut black_gap.conditioning[0] {
        frames[1].pixels.fill(0);
    }
    assert!(
        validate_request(&prepared, &black_gap).is_err(),
        "black is not a neutral generated gap and must not be admitted as one"
    );
}

#[test]
fn video_bridge_is_zero_reference_with_sealed_vace_and_keyframe_carriers() {
    for (backend, quant) in [
        (WanI2vBackend::Mlx, Some(Quant::Q4)),
        (WanI2vBackend::Mlx, Some(Quant::Q8)),
        (WanI2vBackend::Mlx, None),
        (WanI2vBackend::Candle, None),
    ] {
        let (_tmp, spec) = vace_fixture(backend, quant);
        let prepared = PreparedWanI2vMemory::prepare(&spec, backend, "wan_vace").unwrap();
        let mut request = vace_bridge_request();
        let contract = request_contract(&prepared, &request).unwrap();
        for strategy in MemoryStrategy::ALL {
            let expected = matches!(
                strategy,
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
            );
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                if expected {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        let strategy = MemoryStrategy::BoundedDecode;
        let selection = selection(&prepared, strategy);
        request.memory = contract.generation_memory(&selection);
        let receipt = request_evidence_revision(&prepared, &request).unwrap();
        let mut run_context = context(&prepared, &request, selection, receipt.clone());
        run_context.has_reference = false;
        assert_eq!(run_context.geometry.reference_count, 0);
        assert!(validate_context(&prepared, &run_context).is_ok());
        if let Conditioning::ControlClip { mask, .. } = &mut request.conditioning[0] {
            mask[1].pixels[0] ^= 1;
        }
        assert!(request_evidence_revision(&prepared, &request).is_err());
    }

    for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
        let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, quant);
        let prepared = PreparedWanI2vMemory::prepare(
            &spec,
            WanI2vBackend::Mlx,
            WanI2vRoute::Ti2v5b.provider_id(),
        )
        .unwrap();
        let mut request = bridge_keyframes_request();
        let contract = request_contract(&prepared, &request).unwrap();
        for strategy in MemoryStrategy::ALL {
            let expected = matches!(
                strategy,
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
            );
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                if expected {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        let selection = selection(&prepared, MemoryStrategy::BoundedDecode);
        request.memory = contract.generation_memory(&selection);
        let receipt = request_evidence_revision(&prepared, &request).unwrap();
        let mut run_context = context(&prepared, &request, selection, receipt.clone());
        run_context.has_reference = false;
        assert_eq!(run_context.geometry.reference_count, 0);
        assert!(validate_context(&prepared, &run_context).is_ok());
        request.conditioning.swap(0, 1);
        assert!(request_evidence_revision(&prepared, &request).is_err());
    }
}

#[test]
fn vace_extend_seals_all_mlx_tiers_and_dense_candle_with_only_resident_and_decode() {
    for (backend, quants) in [
        (
            WanI2vBackend::Mlx,
            vec![Some(Quant::Q4), Some(Quant::Q8), None],
        ),
        (WanI2vBackend::Candle, vec![None]),
    ] {
        for quant in quants {
            let (_tmp, spec) = vace_fixture(backend, quant);
            let prepared = PreparedWanI2vMemory::prepare(&spec, backend, "wan_vace").unwrap();
            assert_eq!(prepared.tier.quant, quant);
            assert_eq!(
                prepared.contract.load_shape,
                gen_core::LoadShape::DeferredMaterialization
            );
            assert_eq!(prepared.revision.len(), 64);
            let contract = request_contract(&prepared, &vace_extend_request()).unwrap();
            for strategy in MemoryStrategy::ALL {
                let expected = matches!(
                    strategy,
                    MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
                );
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    }
                );
            }
            let mut request = vace_extend_request();
            let resident = selection(&prepared, MemoryStrategy::Resident);
            request.memory = contract.generation_memory(&resident);
            let evidence =
                request_evidence_revision_for_selection(&prepared, &request, &resident).unwrap();
            let staged_context = context(
                &prepared,
                &request,
                selection(&prepared, MemoryStrategy::StagedResidency),
                evidence,
            );
            assert!(validate_context(&prepared, &staged_context).is_err());
        }
    }
}

#[test]
fn vace_lazy_load_shape_is_declared_at_preparation_and_rejects_crossed_reuse() {
    for backend in [WanI2vBackend::Mlx, WanI2vBackend::Candle] {
        let (_tmp, mut spec) = vace_fixture(backend, None);
        assert_eq!(
            spec.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );
        assert_eq!(
            spec.load_shape_declaration_result,
            gen_core::LoadShapeDeclarationResult::Applied
        );
        let prepared = PreparedWanI2vMemory::prepare(&spec, backend, "wan_vace").unwrap();
        assert_eq!(
            prepared.contract.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        );

        // A warmed provider may reuse only the same declared shape.  Mutating the
        // cached spec after preparation cannot relabel lazy VACE as eager.
        spec.load_shape = gen_core::LoadShape::EagerMaterialization;
        assert!(PreparedWanI2vMemory::prepare(&spec, backend, "wan_vace").is_err());
    }
}

#[test]
fn vace_requires_the_exact_immutable_source_receipt() {
    let (_tmp, mut spec) = vace_fixture(WanI2vBackend::Mlx, None);
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        _ => unreachable!(),
    };
    std::fs::remove_file(root.join(VACE_SOURCE_RECEIPT_FILE)).unwrap();
    assert!(prepare_load_spec(&mut spec, WanI2vBackend::Mlx, "wan_vace").is_err());

    ensure_wan_vace_source_receipt(&root, WanI2vBackend::Mlx).unwrap();
    let path = root.join(VACE_SOURCE_RECEIPT_FILE);
    let mut crossed: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    crossed["components"][0]["revision"] =
        serde_json::json!("0000000000000000000000000000000000000000");
    std::fs::write(path, serde_json::to_vec(&crossed).unwrap()).unwrap();
    assert!(prepare_load_spec(&mut spec, WanI2vBackend::Mlx, "wan_vace").is_err());
}

#[test]
fn vace_extend_binds_control_pixels_masks_scale_and_exact_carrier() {
    let (_tmp, spec) = vace_fixture(WanI2vBackend::Mlx, Some(Quant::Q4));
    let prepared = PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan_vace").unwrap();
    let mut request = vace_extend_request();
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: MemoryStrategyParameters::default(),
        tier: prepared.tier,
    };
    request.memory = request_contract(&prepared, &request)
        .unwrap()
        .generation_memory(&selection);
    let receipt = request_evidence_revision(&prepared, &request).unwrap();
    if let Conditioning::ControlClip { frames, .. } = &mut request.conditioning[0] {
        frames[0].pixels[0] ^= 1;
    }
    assert_ne!(
        request_evidence_revision(&prepared, &request).unwrap(),
        receipt
    );
    if let Conditioning::ControlClip { frames, mask, .. } = &mut request.conditioning[0] {
        frames[0].pixels[0] ^= 1;
        mask[0].pixels[0] ^= 1;
    }
    assert_ne!(
        request_evidence_revision(&prepared, &request).unwrap(),
        receipt
    );
    request.conditioning.push(Conditioning::Reference {
        image: Image {
            width: 1,
            height: 1,
            pixels: vec![0; 3],
        },
        strength: None,
    });
    assert!(validate_request(&prepared, &request).is_err());
}

#[test]
fn ti2v_extend_is_an_mlx_only_single_boundary_keyframe_with_the_exact_frame_menu() {
    for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
        let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, quant);
        let prepared = PreparedWanI2vMemory::prepare(
            &spec,
            WanI2vBackend::Mlx,
            WanI2vRoute::Ti2v5b.provider_id(),
        )
        .unwrap();
        let mut request = ti2v_extend_request();
        let contract = request_contract(&prepared, &request).unwrap();
        for strategy in MemoryStrategy::ALL {
            let expected = matches!(
                strategy,
                MemoryStrategy::Resident | MemoryStrategy::BoundedDecode
            );
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                if expected {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        request.memory =
            contract.generation_memory(&selection(&prepared, MemoryStrategy::Resident));
        let receipt = request_evidence_revision(&prepared, &request).unwrap();
        if let Conditioning::Keyframe {
            image, strength, ..
        } = &mut request.conditioning[0]
        {
            image.pixels[0] ^= 1;
            *strength = 0.5;
        }
        assert_ne!(
            request_evidence_revision(&prepared, &request).unwrap(),
            receipt
        );
    }

    let (_tmp, spec) = candle_fixture(WanI2vRoute::Ti2v5b, None);
    let prepared = PreparedWanI2vMemory::prepare(
        &spec,
        WanI2vBackend::Candle,
        WanI2vRoute::Ti2v5b.provider_id(),
    )
    .unwrap();
    assert!(validate_request(&prepared, &ti2v_extend_request()).is_err());
}

/// Every Candle dense leg is bound to the snapshot it is STAGED at, not to a revision the
/// manifest never pinned (sc-22736 review): a well-formed snapshot id other than the fixture's
/// prepares and is the receipt's revision, and a malformed one is still refused. Mutation that
/// reds this: restoring a hard revision on any of the three `(Candle, route, None)` arms of
/// `repository_policy`.
#[test]
fn every_candle_dense_leg_accepts_the_staged_snapshot_and_refuses_a_malformed_one() {
    const STAGED: &str = "fedcba9876543210fedcba9876543210fedcba98";
    for route in [
        WanI2vRoute::Ti2v5b,
        WanI2vRoute::T2v14b,
        WanI2vRoute::I2v14b,
    ] {
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        };
        let (repository, policy_revision) = repository_policy(WanI2vBackend::Candle, route, tier);
        assert_eq!(policy_revision, UNPINNED_UPSTREAM_REVISION, "{route:?}");
        let (tmp, _spec) = candle_fixture(route, None);
        let snapshots = tmp.path().join(repo_dir(repository)).join("snapshots");
        for (staged, expected) in [
            (STAGED, Ok(STAGED)),
            (
                "FEDCBA9876543210FEDCBA9876543210FEDCBA98",
                Err("upper-case"),
            ),
            ("fedcba9876543210fedcba9876543210fedcba9", Err("39 chars")),
        ] {
            let root = snapshots.join(staged);
            if root.exists() {
                std::fs::remove_dir_all(&root).unwrap();
            }
            std::fs::rename(snapshots.join(FIXTURE_UNPINNED_REVISION), &root).unwrap();
            // A fresh spec over the moved root: the fixture's own pins name absolute paths
            // under the directory that was just renamed away.
            let mut moved = LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_resolved_route(route.provider_id());
            moved.quantize = None;
            let prepared = prepare_load_spec(
                &mut moved,
                WanI2vBackend::Candle,
                route.provider_id(),
            )
            .and_then(|()| {
                PreparedWanI2vMemory::prepare(&moved, WanI2vBackend::Candle, route.provider_id())
            });
            match expected {
                Ok(revision) => {
                    let prepared =
                        prepared.unwrap_or_else(|error| panic!("{route:?} at {staged}: {error}"));
                    assert_eq!(prepared.revision, revision, "{route:?}");
                    assert_eq!(prepared.repository, repository, "{route:?}");
                }
                Err(why) => {
                    let error = prepared
                        .err()
                        .unwrap_or_else(|| panic!("{route:?}: {why} must be refused"))
                        .to_string();
                    assert!(
                        error.contains("does not match immutable"),
                        "{route:?} {why}: {error}"
                    );
                }
            }
            std::fs::rename(&root, snapshots.join(FIXTURE_UNPINNED_REVISION)).unwrap();
        }
    }
}

#[test]
fn mlx_q4_q8_bf16_facts_are_nonzero_differentiated_and_rsd_only() {
    let mut totals = Vec::new();
    for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
        let (_tmp, spec) = mlx_fixture(WanI2vRoute::I2v14b, quant);
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
        assert!(prepared.contract.asset_facts.conditioning_bytes > 0);
        assert!(prepared.contract.asset_facts.transformer_bytes > 0);
        assert!(prepared.contract.asset_facts.decoder_bytes > 0);
        totals.push(prepared.contract.asset_facts.base_bytes);
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                prepared.contract.capability(strategy).unwrap().support,
                if matches!(
                    strategy,
                    MemoryStrategy::Resident
                        | MemoryStrategy::StagedResidency
                        | MemoryStrategy::BoundedDecode
                ) {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
    }
    assert!(totals[0] < totals[1] && totals[1] < totals[2], "{totals:?}");

    let (_tmp, mut prepacked) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    prepacked.quantize = None;
    let prepared = PreparedWanI2vMemory::prepare(
        &prepacked,
        WanI2vBackend::Mlx,
        WanI2vRoute::Ti2v5b.provider_id(),
    )
    .expect("the production worker selects a prepacked MLX q4 directory with quantize=None");
    assert_eq!(prepared.tier.quant, Some(Quant::Q4));
}

/// Feature-end review (SC-22667, E1). `mlx_gen_wan::text_encoder` packs the UMT5 encoder's seven
/// bias-less projections per block to the Q8 floor on **every** MLX-affine quantized tier, and
/// the sealed receipt must charge the packed encoder rather than the dense bf16 stack it was
/// built from — the largest staged component of both the 5B and 14B routes.
///
/// The expected total is spelled out from the fixture's own geometry rather than captured, so
/// the test is an oracle and not a snapshot. Mutations that fail this: restoring the single
/// `_ => projected_dense_bytes(path, 2)` arm for the quantized `MemoryPhase::Conditioning`
/// case, which reports the 439_680 B shard sum on the quantized assertions; and dropping the
/// dense-MLX conditioning arm (SC-22667 review), which reports that same 439_680 B on the dense
/// assertion instead of the 445_184 B the f32 leaf upcasts make it.
#[test]
fn the_mlx_encoder_is_charged_packed_on_every_quantized_tier() {
    // `write_umt5_encoder(_, blocks = 7, dim = 64)`, Q8 at MLX group 64:
    //   token embedding  [256, 64] bf16, cloned as stored              = 32_768
    //   final norm       [64]      upcast f32                          =    256
    //   per block:  7 projections [64, 64]
    //                 codes  4096 * 8 / 8                              =  4_096
    //                 scales+biases  64 * (64/64) * 2 planes * 2 B     =    256
    //               2 norms [64] upcast f32                            =    512
    //               pos-bias [32, 8] upcast f32                        =  1_024
    //                                                       per block  = 32_000
    const PACKED_ENCODER_BYTES: u64 = 32_768 + 256 + 7 * 32_000;
    // The dense tier (SC-22667 review): the seven projections stay bf16 as stored
    // (7 * 4096 * 2 = 57_344 per block), but `Umt5Block::from_weights` / `assemble` upcast the
    // same leaves on this path too — the final norm (256), the two block norms (512) and the
    // pos-bias table (1_024) — so the dense figure is NOT the plain bf16 shard sum
    // (32_768 + 128 + 7 * 58_112 = 439_680); it is 445_184.
    const DENSE_ENCODER_BYTES: u64 = 32_768 + 256 + 7 * (57_344 + 512 + 1_024);

    let mut dense_conditioning = 0;
    for route in [WanI2vRoute::Ti2v5b, WanI2vRoute::I2v14b] {
        for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
            let (_tmp, spec) = mlx_fixture(route, quant);
            let prepared =
                PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, route.provider_id())
                    .unwrap();
            let conditioning = prepared.contract.asset_facts.conditioning_bytes;
            match quant {
                // Q4 and Q8 agree: `effective_te_quant` floors the encoder at Q8 for both.
                Some(_) => assert_eq!(
                    conditioning, PACKED_ENCODER_BYTES,
                    "{route:?} {quant:?} conditioning"
                ),
                None => {
                    assert_eq!(
                        conditioning, DENSE_ENCODER_BYTES,
                        "{route:?} dense conditioning"
                    );
                    dense_conditioning = conditioning;
                }
            }
        }
    }
    assert!(
        PACKED_ENCODER_BYTES < dense_conditioning,
        "packing the encoder must reduce its charged residency"
    );
}

/// Feature-end review (SC-22667, E1). Both candle VACE routes open their video autoencoder at
/// `VAE_DTYPE = DType::F32` (`model_vace.rs`, `model_vace_fun.rs`), so the sealed decoder bytes
/// are four per stored bf16 element, not two.
///
/// Mutation that fails this: widening the VACE early return in `physical_resident_bytes` back to
/// every phase, which routes the decoder past the `Decode` arm and halves this number.
#[test]
fn candle_vace_decodes_f32_and_is_charged_at_that_width() {
    for route in [WanI2vRoute::Vace, WanI2vRoute::VaceFun] {
        let (_tmp, spec) = vace_route_fixture(WanI2vBackend::Candle, None, route);
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Candle, route.provider_id())
                .unwrap();
        // The fixture's `vae/model.safetensors` is one bf16 [4, 4] tensor: 16 elements.
        assert_eq!(
            prepared.contract.asset_facts.decoder_bytes,
            16 * 4,
            "{route:?} decoder is materialized f32"
        );
        // The conditioning encoder and the transformer stay bf16 on this route.
        assert_eq!(
            prepared.contract.asset_facts.conditioning_bytes,
            16 * 2,
            "{route:?} conditioning is materialized bf16"
        );
    }
}

#[test]
fn candle_routes_and_dense_commits_are_sealed() {
    for route in [WanI2vRoute::Ti2v5b, WanI2vRoute::I2v14b] {
        let mut totals = Vec::new();
        for quant in [Some(Quant::Q4), Some(Quant::Q8), None] {
            let (_tmp, spec) = candle_fixture(route, quant);
            let prepared =
                PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Candle, route.provider_id())
                    .unwrap();
            assert_eq!(prepared.revision.len(), 40);
            assert!(prepared.contract.asset_facts.base_bytes > 0);
            totals.push(prepared.contract.asset_facts.base_bytes);
        }
        assert!(
            totals[0] < totals[1] && totals[1] < totals[2],
            "{route:?}: {totals:?}"
        );
    }
}

#[test]
fn same_length_mutation_and_inventory_or_repository_crossing_fail_closed() {
    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => unreachable!(),
    };
    let path = root.join("model.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&path, bytes).unwrap();
    assert!(prepared.ensure_unchanged().is_err());

    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => unreachable!(),
    };
    write_safetensors(&root.join("extra.safetensors"), &[("x", "F32", &[1])]);
    assert!(PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").is_err());

    let mut crossed = spec.clone();
    crossed.weights = WeightsSource::Dir(
        root.parent()
            .unwrap()
            .join("ffffffffffffffffffffffffffffffffffffffff/q4"),
    );
    assert!(PreparedWanI2vMemory::prepare(&crossed, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").is_err());
}

/// sc-22738 (defect B): every packed file of the shipped a14b rehosts
/// (`SceneWorks/wan2.2-i2v-a14b-mlx@c6c786170031eccc3a1fac0f98f1ad4ff988271e` and its t2v
/// sibling) leaves the output projection `head.head.weight` dense BF16 `[64, 5120]` with no
/// `.scales`/`.biases` — the standing converter convention. The first pass demanded scales for
/// every 2-D `.weight` and refused before the second pass, which already prices a dense
/// residual correctly.
#[test]
fn a_packed_file_prices_a_dense_output_head_but_still_refuses_stripped_quant_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let shipped: &[(&str, &str, &[usize])] = &[
        ("blocks.0.self_attn.q.weight", "U32", &[64, 8]),
        ("blocks.0.self_attn.q.scales", "BF16", &[64, 1]),
        ("blocks.0.self_attn.q.biases", "BF16", &[64, 1]),
        ("blocks.0.norm1.weight", "BF16", &[64]),
        // Dense by convention on every packed tier — no quant metadata at all.
        ("head.head.weight", "BF16", &[64, 64]),
        ("head.head.bias", "BF16", &[64]),
    ];
    let path = tmp.path().join("high_noise_model.safetensors");
    write_safetensors(&path, shipped);

    // Packed triple (MLX prices stored bytes): 64*8*4 + 64*2 + 64*2 = 2304.
    // Dense residual at the bf16 resident width: (64*64 + 64 + 64) * 2 = 8448.
    let mlx = packed_transformer_bytes(&path, Quant::Q4, WanI2vBackend::Mlx).unwrap();
    assert_eq!(mlx, 2304 + 8448);
    // Candle prices the packed block form (20 bytes / 32 Q4 elements) and the same residual.
    let candle = packed_transformer_bytes(&path, Quant::Q4, WanI2vBackend::Candle).unwrap();
    assert_eq!(candle, (64 * 64 / 32) * 20 + 8448);

    // The discriminator is "float storage AND no quant metadata at all", so genuine corruption
    // still fails closed: packed U32 storage that lost both leaves, that lost one of them, and
    // a float weight carrying a lone leaf are each refused.
    // Each case pairs the corruption with the message that has to name it, so a guard loosened
    // to any one of its three conjuncts is caught rather than being absorbed by the second
    // pass's orphan-leaf backstop.
    let valid_triple: &[(&str, &str, &[usize])] = &[
        ("blocks.0.self_attn.q.weight", "U32", &[64, 8]),
        ("blocks.0.self_attn.q.scales", "BF16", &[64, 1]),
        ("blocks.0.self_attn.q.biases", "BF16", &[64, 1]),
    ];
    for (corruption, expected) in [
        // U32 storage with both leaves stripped, alongside a healthy packed sibling: packed
        // storage is unambiguous, so this can never be read as dense by convention.
        (
            vec![
                ("blocks.0.ffn.0.weight", "U32", &[64, 8][..]),
                ("head.head.weight", "BF16", &[64, 64][..]),
            ],
            "packed blocks.0.ffn.0 lacks scales",
        ),
        // A float weight with a lone `.biases` leaf carries quant metadata, so it is a packed
        // tensor missing its scales — not a dense head.
        (
            vec![
                ("head.head.weight", "BF16", &[64, 64][..]),
                ("head.head.biases", "BF16", &[64, 1][..]),
            ],
            "packed head.head lacks scales",
        ),
        // …and the mirror case, a lone `.scales` leaf.
        (
            vec![
                ("head.head.weight", "BF16", &[64, 64][..]),
                ("head.head.scales", "BF16", &[64, 1][..]),
            ],
            "packed head.head lacks biases",
        ),
    ] {
        let mut mutated = valid_triple.to_vec();
        mutated.extend(corruption.iter().copied());
        let path = tmp.path().join("mutated.safetensors");
        write_safetensors(&path, &mutated);
        let error = packed_transformer_bytes(&path, Quant::Q4, WanI2vBackend::Mlx)
            .expect_err("a packed tensor missing quant metadata must still be refused")
            .to_string();
        assert!(
            error.contains(expected),
            "expected {expected:?}, got {error:?}"
        );
    }

    // A file whose only 2-D weight is the dense head has no packed weights at all and is not a
    // packed tier — the emptiness guard must still catch it.
    let path = tmp.path().join("dense_only.safetensors");
    write_safetensors(&path, &[("head.head.weight", "BF16", &[64, 64])]);
    assert!(packed_transformer_bytes(&path, Quant::Q4, WanI2vBackend::Mlx).is_err());
}

fn request(route: WanI2vRoute) -> GenerationRequest {
    let (width, height, frames, fps, steps, guidance) = match route {
        WanI2vRoute::Ti2v5b => (832, 480, 121, 24, 20, Some(5.0)),
        WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => (1280, 720, 77, 16, 40, None),
        WanI2vRoute::Vace => panic!("use a dedicated VACE request"),
        WanI2vRoute::VaceFun => panic!("use a dedicated VACE-Fun request"),
    };
    // T2V-A14B is the one carrier-free route: it spells `text_to_video` and carries no
    // conditioning at all. Building it from the same helper keeps the two A14B routes' geometry
    // in one place, which is the point — they differ only in the carrier.
    if route == WanI2vRoute::T2v14b {
        return GenerationRequest {
            prompt: "a slow dolly across an empty room".to_owned(),
            width,
            height,
            count: 1,
            seed: Some(17),
            steps: Some(steps),
            guidance,
            frames: Some(frames),
            fps: Some(fps),
            video_mode: Some("text_to_video".to_owned()),
            conditioning: Vec::new(),
            ..Default::default()
        };
    }
    GenerationRequest {
        prompt: "animate the still".to_owned(),
        width,
        height,
        count: 1,
        seed: Some(17),
        steps: Some(steps),
        guidance,
        frames: Some(frames),
        fps: Some(fps),
        video_mode: Some("image_to_video".to_owned()),
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: 16,
                height: 16,
                pixels: vec![7; 16 * 16 * 3],
            },
            strength: None,
        }],
        ..Default::default()
    }
}

fn flf_request(backend: WanI2vBackend) -> GenerationRequest {
    let image = |pixel| Image {
        width: 832,
        height: 480,
        pixels: vec![pixel; 832 * 480 * 3],
    };
    GenerationRequest {
        prompt: "move between the two exact endpoints".to_owned(),
        negative_prompt: Some("camera cut".to_owned()),
        width: 832,
        height: 480,
        count: 1,
        seed: Some(23),
        steps: Some(20),
        guidance: Some(5.0),
        sampler: (backend == WanI2vBackend::Mlx).then(|| "dpmpp_2m".to_owned()),
        frames: Some(117),
        fps: Some(24),
        video_mode: None,
        conditioning: vec![
            Conditioning::Keyframe {
                image: image(17),
                frame_idx: 0,
                strength: 0.25,
            },
            Conditioning::Keyframe {
                image: image(91),
                frame_idx: -1,
                strength: 0.75,
            },
        ],
        ..Default::default()
    }
}

fn selection(prepared: &PreparedWanI2vMemory, strategy: MemoryStrategy) -> MemorySelection {
    MemorySelection {
        strategy,
        parameters: if strategy == MemoryStrategy::BoundedDecode {
            MemoryStrategyParameters {
                decode_tile_edge: Some(192),
                decode_overlap: Some(64),
                ..Default::default()
            }
        } else {
            Default::default()
        },
        tier: prepared.tier,
    }
}

fn context(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
    selection: MemorySelection,
    evidence_revision: String,
) -> MemoryRunContext {
    MemoryRunContext {
        reference_resources: None,
        selection,
        optimization_authority: if selection.strategy.is_optimized() {
            gen_core::MemoryOptimizationAuthority::Estimated
        } else {
            gen_core::MemoryOptimizationAuthority::Resident
        },
        calibration_abi: gen_core::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: String::new(),
        load_shape: prepared.contract.load_shape,
        mode: gen_core::MemoryMode::Other(
            match request.video_mode.as_deref() {
                Some("image_to_video") => "image_to_video",
                Some("extend_clip") => "extend_clip",
                Some("video_bridge") => "video_bridge",
                Some(mode) if mode.starts_with("replace_person@") => "replace_person",
                _ => "first_last_frame",
            }
            .to_owned(),
        ),
        has_reference: request.video_mode.as_deref() != Some("video_bridge"),
        use_pid: false,
        has_phases: false,
        geometry: geometry_from_request(request),
        overlay: Some(prepared.adapter_identity.clone()),
        budget: gen_core::MemoryBudget {
            total_bytes: u64::MAX,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1,
        cache_state: gen_core::MemoryCacheState::Cold,
        evidence_revision,
    }
}

#[test]
fn request_identity_binds_reference_geometry_rate_seed_schedule_and_singular_carrier() {
    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
    let resident = selection(&prepared, MemoryStrategy::Resident);
    let baseline = request_evidence_revision_for_selection(
        &prepared,
        &request(WanI2vRoute::Ti2v5b),
        &resident,
    )
    .unwrap();
    let mut crossed = request(WanI2vRoute::Ti2v5b);
    crossed.seed = Some(18);
    assert_ne!(
        baseline,
        request_evidence_revision_for_selection(&prepared, &crossed, &resident).unwrap()
    );
    let mut crossed = request(WanI2vRoute::Ti2v5b);
    crossed.conditioning.push(crossed.conditioning[0].clone());
    assert!(request_evidence_revision_for_selection(&prepared, &crossed, &resident).is_err());
    for (width, height) in [(832, 480), (1280, 704), (704, 1280)] {
        let mut accepted = request(WanI2vRoute::Ti2v5b);
        (accepted.width, accepted.height) = (width, height);
        assert!(validate_request(&prepared, &accepted).is_ok());
    }
    let mut off_menu = request(WanI2vRoute::Ti2v5b);
    (off_menu.width, off_menu.height) = (480, 832);
    assert!(validate_request(&prepared, &off_menu).is_err());
}

#[test]
fn selection_receipt_binds_strategy_and_every_parameter_field() {
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: Some(Quant::Q4),
        component_precision_floors: &[],
    };
    let baseline = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier,
    };
    let mut variants = Vec::new();
    let mut changed = baseline;
    changed.strategy = MemoryStrategy::StagedResidency;
    variants.push(changed);
    let setters: &[fn(&mut MemoryStrategyParameters)] = &[
        |parameters| parameters.decode_tile_edge = Some(192),
        |parameters| parameters.decode_overlap = Some(64),
        |parameters| parameters.attention_chunk_size = Some(1024),
        |parameters| parameters.transformer_window_size = Some(2),
        |parameters| {
            parameters.transformer_window_component =
                Some(gen_core::TransformerComponent::TextEncoder)
        },
    ];
    for setter in setters {
        let mut changed = baseline;
        setter(&mut changed.parameters);
        variants.push(changed);
    }
    let baseline_digest = selection_receipt_digest(&baseline);
    let digests = variants
        .iter()
        .map(selection_receipt_digest)
        .collect::<BTreeSet<_>>();
    assert_eq!(digests.len(), variants.len());
    assert!(digests.iter().all(|digest| digest != &baseline_digest));
}

#[test]
fn provider_and_configure_boundaries_reject_crossed_strategy_or_decode_parameters() {
    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
    let request = request(WanI2vRoute::Ti2v5b);
    let resident = selection(&prepared, MemoryStrategy::Resident);
    let resident_evidence =
        request_evidence_revision_for_selection(&prepared, &request, &resident).unwrap();
    let resident_context = context(&prepared, &request, resident, resident_evidence.clone());
    assert!(validate_context(&prepared, &resident_context).is_ok());

    let staged = selection(&prepared, MemoryStrategy::StagedResidency);
    let mut crossed_context = resident_context.clone();
    crossed_context.selection = staged;
    crossed_context.optimization_authority = gen_core::MemoryOptimizationAuthority::Estimated;
    assert!(validate_context(&prepared, &crossed_context).is_err());

    let staged_evidence =
        request_evidence_revision_for_selection(&prepared, &request, &staged).unwrap();
    let staged_context = context(&prepared, &request, staged, staged_evidence.clone());
    assert!(validate_context(&prepared, &staged_context).is_ok());
    let mut staged_request = request.clone();
    staged_request.memory = prepared.contract.generation_memory(&staged);
    assert_eq!(
        validate_request_evidence(&prepared, &staged_request, &staged, &staged_evidence,).unwrap(),
        staged_evidence
    );

    let bounded = selection(&prepared, MemoryStrategy::BoundedDecode);
    let bounded_evidence =
        request_evidence_revision_for_selection(&prepared, &request, &bounded).unwrap();
    let mut crossed_parameters = bounded;
    crossed_parameters.parameters.decode_tile_edge = Some(256);
    let mut crossed_context = context(
        &prepared,
        &request,
        crossed_parameters,
        bounded_evidence.clone(),
    );
    assert!(validate_context(&prepared, &crossed_context).is_err());
    crossed_context.evidence_revision =
        request_evidence_revision_for_selection(&prepared, &request, &crossed_parameters).unwrap();
    assert!(validate_context(&prepared, &crossed_context).is_ok());

    let mut crossed_request = request;
    crossed_request.memory = prepared.contract.generation_memory(&crossed_parameters);
    assert!(
        validate_request_evidence(&prepared, &crossed_request, &bounded, &bounded_evidence,)
            .is_err()
    );
}

#[test]
fn first_last_frame_exact_menu_and_request_specific_rungs_cover_both_backends_and_tiers() {
    for backend in [WanI2vBackend::Mlx, WanI2vBackend::Candle] {
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let (_tmp, spec) = match backend {
                WanI2vBackend::Mlx => mlx_fixture(WanI2vRoute::Ti2v5b, quant),
                WanI2vBackend::Candle => candle_fixture(WanI2vRoute::Ti2v5b, quant),
            };
            let prepared =
                PreparedWanI2vMemory::prepare(&spec, backend, WanI2vRoute::Ti2v5b.provider_id())
                    .unwrap();
            let mut request = flf_request(backend);
            let contract = request_contract(&prepared, &request).unwrap();
            for strategy in MemoryStrategy::ALL {
                let expected = match strategy {
                    MemoryStrategy::Resident | MemoryStrategy::BoundedDecode => {
                        MemoryStrategySupport::Implemented
                    }
                    MemoryStrategy::StagedResidency if backend == WanI2vBackend::Candle => {
                        MemoryStrategySupport::Implemented
                    }
                    _ => MemoryStrategySupport::Missing,
                };
                assert_eq!(contract.capability(strategy).unwrap().support, expected);
            }
            let bounded = selection(&prepared, MemoryStrategy::BoundedDecode);
            let memory = contract.generation_memory(&bounded).unwrap();
            assert!(memory.tile_vae_decode);
            if backend == WanI2vBackend::Mlx {
                assert!(!memory.stage_residency);
            }

            for (fps, frames) in [(16, [61, 77, 93, 109, 125]), (24, [93, 117, 141, 165, 189])] {
                request.fps = Some(fps);
                for frame_count in frames {
                    request.frames = Some(frame_count);
                    assert_eq!(
                        validate_request(&prepared, &request).unwrap(),
                        WanPublicMode::FirstLastFrame
                    );
                }
            }
            request.fps = Some(24);
            request.frames = Some(121);
            assert!(validate_request(&prepared, &request).is_err(), "117 != 121");
        }
    }
}

#[test]
fn first_last_frame_receipt_binds_roles_indices_strengths_shapes_and_mode() {
    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
    let request = flf_request(WanI2vBackend::Mlx);
    let resident = selection(&prepared, MemoryStrategy::Resident);
    let baseline = request_evidence_revision_for_selection(&prepared, &request, &resident).unwrap();
    let mut changed = request.clone();
    changed.conditioning.swap(0, 1);
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
    let mut changed = request.clone();
    if let Conditioning::Keyframe { frame_idx, .. } = &mut changed.conditioning[0] {
        *frame_idx = 1;
    }
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
    let mut changed = request.clone();
    if let Conditioning::Keyframe { strength, .. } = &mut changed.conditioning[1] {
        *strength = 0.5;
    }
    assert_ne!(
        baseline,
        request_evidence_revision_for_selection(&prepared, &changed, &resident).unwrap()
    );
    let mut changed = request.clone();
    if let Conditioning::Keyframe { image, .. } = &mut changed.conditioning[0] {
        image.width = 480;
    }
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
    let mut changed = request.clone();
    changed.video_mode = Some("image_to_video".to_owned());
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
    let mut changed = request.clone();
    changed.frames = Some(121);
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
    let mut changed = request.clone();
    changed.fps = Some(25);
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
    let mut no_guidance = request.clone();
    no_guidance.guidance = None;
    let mut zero_guidance = request.clone();
    zero_guidance.guidance = Some(0.0);
    assert_ne!(
        request_evidence_revision_for_selection(&prepared, &no_guidance, &resident).unwrap(),
        request_evidence_revision_for_selection(&prepared, &zero_guidance, &resident).unwrap()
    );
    let mut changed = request.clone();
    changed.width = 1280;
    assert!(request_evidence_revision_for_selection(&prepared, &changed, &resident).is_err());
}

#[test]
fn first_last_frame_context_and_configure_refuse_crossed_mode_or_mlx_staged() {
    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q8));
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
    let mut request = flf_request(WanI2vBackend::Mlx);
    let contract = request_contract(&prepared, &request).unwrap();
    let bounded = selection(&prepared, MemoryStrategy::BoundedDecode);
    request.memory = contract.generation_memory(&bounded);
    let evidence = request_evidence_revision(&prepared, &request).unwrap();
    let valid = context(&prepared, &request, bounded, evidence.clone());
    assert!(validate_context(&prepared, &valid).is_ok());

    let mut crossed = valid.clone();
    crossed.mode = gen_core::MemoryMode::Other("image_to_video".to_owned());
    assert!(validate_context(&prepared, &crossed).is_err());
    let staged = selection(&prepared, MemoryStrategy::StagedResidency);
    let mut crossed = valid.clone();
    crossed.selection = staged;
    assert!(validate_context(&prepared, &crossed).is_err());
    let mut crossed_request = request.clone();
    crossed_request.memory = prepared.contract.generation_memory(&staged);
    assert!(validate_request_evidence(&prepared, &crossed_request, &bounded, &evidence,).is_err());
}

#[test]
fn first_last_frame_sampler_set_is_backend_truthful() {
    for backend in [WanI2vBackend::Mlx, WanI2vBackend::Candle] {
        let (_tmp, spec) = match backend {
            WanI2vBackend::Mlx => mlx_fixture(WanI2vRoute::Ti2v5b, None),
            WanI2vBackend::Candle => candle_fixture(WanI2vRoute::Ti2v5b, None),
        };
        let prepared =
            PreparedWanI2vMemory::prepare(&spec, backend, WanI2vRoute::Ti2v5b.provider_id())
                .unwrap();
        for sampler in [None, Some("uni_pc"), Some("euler")] {
            let mut request = flf_request(backend);
            request.sampler = sampler.map(str::to_owned);
            assert!(validate_request(&prepared, &request).is_ok());
        }
        let mut dpm = flf_request(backend);
        dpm.sampler = Some("dpmpp_2m".to_owned());
        assert_eq!(
            validate_request(&prepared, &dpm).is_ok(),
            backend == WanI2vBackend::Mlx
        );
        let mut broad = flf_request(backend);
        broad.sampler = Some("heun".to_owned());
        assert!(validate_request(&prepared, &broad).is_err());
    }
}

#[test]
fn adapter_residency_is_dense_zero_packed_shared_twice_targeted_once_and_loha_refuses() {
    let make_adapter = |dir: &Path, name: &str, key: &str| {
        let path = dir.join(name);
        write_safetensors(&path, &[(key, "F32", &[2, 3])]);
        path
    };
    let (tmp, dense_fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
    let mut dense =
        LoadSpec::new(dense_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
    let shared = make_adapter(tmp.path(), "shared.safetensors", "lora_A.weight");
    let high = make_adapter(tmp.path(), "high.safetensors", "lora_A.weight");
    dense.adapters = vec![
        AdapterSpec::new(shared.clone(), 0.5, AdapterKind::Lora),
        AdapterSpec::new(high.clone(), 0.75, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
    ];
    let packed_adapters = dense.adapters.clone();
    prepare_load_spec(&mut dense, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    let prepared =
        PreparedWanI2vMemory::prepare(&dense, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    assert_eq!(prepared.contract.asset_facts.overlay_bytes, 0);
    assert_eq!(
        prepared.adapters[0].path,
        std::path::absolute(&shared).unwrap()
    );
    assert_eq!(prepared.adapters[0].scale_bits, 0.5_f32.to_bits());
    assert!(prepared.adapters[0].pass_scale_bits.is_empty());
    assert_eq!(prepared.adapters[0].expert, "shared");
    assert_eq!(prepared.adapters[0].realization, "dense_folded");
    assert_eq!(
        prepared.adapters[0].stability,
        "pinned_digest_and_tensor_geometry"
    );

    let (_tmp, packed_fixture) = mlx_fixture(WanI2vRoute::I2v14b, Some(Quant::Q4));
    let mut packed =
        LoadSpec::new(packed_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
    packed.quantize = Some(Quant::Q4);
    packed.adapters = packed_adapters;
    prepare_load_spec(&mut packed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    let prepared =
        PreparedWanI2vMemory::prepare(&packed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    assert_eq!(
        prepared.adapters[0].persistent_bytes,
        prepared.adapters[0].source_bytes * 2
    );
    assert_eq!(
        prepared.adapters[1].persistent_bytes,
        prepared.adapters[1].source_bytes
    );
    assert_eq!(prepared.adapters[0].realization, "packed_additive_factors");
    assert_eq!(prepared.adapters[1].expert, "high");

    let loha = make_adapter(tmp.path(), "loha.safetensors", "block.hada_w1_a");
    let (_dense_loha_tmp, dense_loha_fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
    let mut dense_loha =
        LoadSpec::new(dense_loha_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
    dense_loha.adapters = vec![AdapterSpec::new(loha.clone(), 1.0, AdapterKind::Lora)];
    prepare_load_spec(&mut dense_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    let dense_loha =
        PreparedWanI2vMemory::prepare(&dense_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    assert_eq!(dense_loha.adapters[0].kind, "loha");
    assert_eq!(dense_loha.adapters[0].persistent_bytes, 0);

    let (_tmp, packed_loha_fixture) = mlx_fixture(WanI2vRoute::I2v14b, Some(Quant::Q4));
    let mut packed_loha =
        LoadSpec::new(packed_loha_fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
    packed_loha.quantize = Some(Quant::Q4);
    packed_loha.adapters = vec![AdapterSpec::new(loha, 1.0, AdapterKind::Lora)];
    prepare_load_spec(&mut packed_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    assert!(
        PreparedWanI2vMemory::prepare(&packed_loha, WanI2vBackend::Mlx, "wan2_2_i2v_14b",).is_err()
    );
}

#[test]
fn lightning_recipe_requires_the_exact_repository_revision_order_and_request_schedule() {
    let (tmp, fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
    let lightning = tmp
        .path()
        .join(repo_dir(LIGHTNING_REPOSITORY))
        .join("snapshots")
        .join(LIGHTNING_REVISION)
        .join("Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1");
    std::fs::create_dir_all(&lightning).unwrap();
    let high = lightning.join("high_noise_model.safetensors");
    let low = lightning.join("low_noise_model.safetensors");
    write_safetensors(&high, &[("lora_A.weight", "F32", &[2, 3])]);
    write_safetensors(&low, &[("lora_A.weight", "F32", &[2, 3])]);
    let mut spec = LoadSpec::new(fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
    spec.adapters = vec![
        AdapterSpec::new(high, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
        AdapterSpec::new(low, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
    ];
    prepare_load_spec(&mut spec, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    let mut lightning_request = request(WanI2vRoute::I2v14b);
    lightning_request.steps = Some(4);
    lightning_request.guidance = Some(1.0);
    assert!(validate_request(&prepared, &lightning_request).is_ok());

    let mut native = request(WanI2vRoute::I2v14b);
    assert!(validate_request(&prepared, &native).is_err());
    native.steps = Some(4);
    native.guidance = Some(1.1);
    assert!(validate_request(&prepared, &native).is_err());

    let (_tmp, fixture) = mlx_fixture(WanI2vRoute::I2v14b, None);
    let bogus_dir = tempfile::tempdir().unwrap();
    let bogus_high = bogus_dir.path().join("high_noise_model.safetensors");
    let bogus_low = bogus_dir.path().join("low_noise_model.safetensors");
    write_safetensors(&bogus_high, &[("lora_A.weight", "F32", &[2, 3])]);
    write_safetensors(&bogus_low, &[("lora_A.weight", "F32", &[2, 3])]);
    let mut crossed = LoadSpec::new(fixture.weights.clone()).with_resolved_route("wan2_2_i2v_14b");
    crossed.adapters = vec![
        AdapterSpec::new(bogus_high, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::High),
        AdapterSpec::new(bogus_low, 1.0, AdapterKind::Lora).with_moe_expert(MoeExpert::Low),
    ];
    prepare_load_spec(&mut crossed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    let crossed =
        PreparedWanI2vMemory::prepare(&crossed, WanI2vBackend::Mlx, "wan2_2_i2v_14b").unwrap();
    assert!(validate_request(&crossed, &lightning_request).is_err());
}

/// sc-20799: the evidence receipt carries its own frame rate in plain text, so admission pins
/// the exact public rate menu instead of OR-ing 16 fps and 24 fps into one indistinguishable
/// cell, and a capture record can key the two Ti2V-5B cells apart.
#[test]
fn the_receipt_carries_its_frame_rate_and_admission_pins_the_exact_public_menu() {
    let (_tmp, spec) = mlx_fixture(WanI2vRoute::Ti2v5b, Some(Quant::Q4));
    let prepared =
        PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, "wan2_2_ti2v_5b").unwrap();
    let mut request = request(WanI2vRoute::Ti2v5b);
    // 121 frames is on the 24 fps menu and NOT on the 16 fps menu.
    assert_eq!(request.fps, Some(24));
    assert!(prepared.route.accepts_rate(24, 121));
    assert!(!prepared.route.accepts_rate(16, 121));

    let selection = selection(&prepared, MemoryStrategy::Resident);
    let contract = request_contract(&prepared, &request).unwrap();
    request.memory = contract.generation_memory(&selection);
    let receipt = request_evidence_revision(&prepared, &request).unwrap();
    assert!(
        receipt.starts_with(&format!("{RECEIPT_VERSION}:image_to_video:fps24:")),
        "{receipt}"
    );
    let run_context = context(&prepared, &request, selection, receipt.clone());
    validate_context(&prepared, &run_context).unwrap();

    // Re-spelling the rate must move the cell, and the crossed cell must be refused because
    // 121 frames is not on the 16 fps menu.
    let mut forged = run_context.clone();
    forged.evidence_revision = receipt.replacen(":fps24:", ":fps16:", 1);
    assert_ne!(forged.evidence_revision, receipt);
    assert!(validate_context(&prepared, &forged).is_err());

    // A receipt with no rate field at all, or an unparsable one, fails closed.
    for crossed in [
        receipt.replacen(":fps24:", ":", 1),
        receipt.replacen(":fps24:", ":fps:", 1),
        receipt.replacen(":fps24:", ":fps+24:", 1),
        receipt.replacen(":fps24:", ":24:", 1),
    ] {
        let mut forged = run_context.clone();
        forged.evidence_revision = crossed.clone();
        assert!(
            validate_context(&prepared, &forged).is_err(),
            "{crossed} must not be admitted"
        );
    }
}

/// sc-20799: the VACE routes have no exact public I2V request contract, so admission must not
/// hand out a row that `request_mode` and `contract_for_mode_key` both refuse.
#[test]
fn vace_admission_refuses_the_image_to_video_mode_no_vace_request_can_execute() {
    let (_tmp, spec) = vace_fixture(WanI2vBackend::Candle, None);
    let prepared = PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Candle, "wan_vace").unwrap();
    assert!(contract_for_mode_key(&prepared, "image_to_video").is_err());

    let mut request = vace_extend_request();
    let selection = selection(&prepared, MemoryStrategy::Resident);
    let contract = request_contract(&prepared, &request).unwrap();
    request.memory = contract.generation_memory(&selection);
    let receipt = request_evidence_revision(&prepared, &request).unwrap();
    let mut run_context = context(&prepared, &request, selection, receipt);
    validate_context(&prepared, &run_context).unwrap();

    run_context.mode = gen_core::MemoryMode::Other("image_to_video".to_owned());
    let error = validate_context(&prepared, &run_context)
        .unwrap_err()
        .to_string();
    assert!(error.contains("crossed Wan public memory mode"), "{error}");

    // The non-VACE VACE-Fun route keeps its public I2V contract.
    let (_tmp, spec) = vace_fun_fixture(WanI2vBackend::Candle, None);
    let fun = PreparedWanI2vMemory::prepare(
        &spec,
        WanI2vBackend::Candle,
        WanI2vRoute::VaceFun.provider_id(),
    )
    .unwrap();
    assert!(contract_for_mode_key(&fun, "image_to_video").is_ok());
}

/// sc-20799: the catch-all refusal must name the selector it actually enforces.
#[test]
fn the_unmatched_carrier_refusal_enumerates_every_public_wan_mode() {
    let (_tmp, spec) = vace_fixture(WanI2vBackend::Candle, None);
    let prepared = PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Candle, "wan_vace").unwrap();
    let mut request = vace_extend_request();
    request.video_mode = Some("not_a_wan_mode".to_owned());
    let error = validate_request(&prepared, &request)
        .unwrap_err()
        .to_string();
    for mode in [
        "image_to_video",
        "first_last_frame",
        "extend_clip",
        "video_bridge",
        "replace_person",
    ] {
        assert!(error.contains(mode), "{mode} missing from: {error}");
    }
}

/// Every A14B cell this authority seals publishes its own production identity, and no two
/// cells share one.
///
/// The cell set is **derived** from `WanI2vRoute::ALL` x both backends x the shipped tier
/// ladder rather than frozen as a count, so a new route or a new tier is swept the moment it
/// exists instead of silently sitting outside a literal.
#[test]
fn every_sealed_a14b_cell_publishes_a_distinct_production_identity() {
    let mut seen: Vec<(String, String)> = Vec::new();
    for route in WanI2vRoute::ALL {
        for backend in [WanI2vBackend::Mlx, WanI2vBackend::Candle] {
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                // Candle ships no dense tier under a tier suffix and MLX ships no untiered
                // root; the fixture writers mirror that, and VACE has its own receipt shape.
                if matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) {
                    continue;
                }
                let (_tmp, spec) = match backend {
                    WanI2vBackend::Mlx => mlx_fixture(route, quant),
                    WanI2vBackend::Candle => candle_fixture(route, quant),
                };
                let prepared = PreparedWanI2vMemory::prepare(&spec, backend, route.provider_id())
                    .unwrap_or_else(|error| {
                        panic!("{:?} {} {quant:?}: {error}", backend, route.provider_id())
                    });
                let expected = production_calibration_fingerprint(route, backend, prepared.tier);
                let published = prepared
                    .contract
                    .calibration
                    .as_ref()
                    .map(|identity| identity.fingerprint.clone());
                assert_eq!(
                    published,
                    expected,
                    "{:?} {} {quant:?} publishes the wrong identity",
                    backend,
                    route.provider_id()
                );
                if route == WanI2vRoute::Ti2v5b {
                    // Owned by the per-crate modules; this authority must stay silent.
                    assert!(
                        published.is_none(),
                        "Ti2V-5B must not mint a second production identity here"
                    );
                    continue;
                }
                let fingerprint = published.expect("an A14B cell publishes an identity");
                let cell = format!("{backend:?}/{}/{quant:?}", route.provider_id());
                for (other_cell, other) in &seen {
                    assert_ne!(
                        &fingerprint, other,
                        "{cell} and {other_cell} share one identity"
                    );
                }
                seen.push((cell, fingerprint));
            }
        }
    }
    // Two A14B routes x two backends x three tiers, all derived above.
    assert_eq!(seen.len(), 12, "swept cells: {seen:?}");
}

/// A root of one tier presented as another is refused, and the refusal names the tier.
#[test]
fn a_crossed_tier_root_is_refused_by_name() {
    for route in [WanI2vRoute::T2v14b, WanI2vRoute::I2v14b] {
        let (_tmp, q4) = mlx_fixture(route, Some(Quant::Q4));
        let mut crossed = q4.clone();
        crossed.quantize = Some(Quant::Q8);
        let error =
            PreparedWanI2vMemory::prepare(&crossed, WanI2vBackend::Mlx, route.provider_id())
                .expect_err("a q4 root selected as q8 must be refused")
                .to_string();
        assert!(
            error.contains("Q8") || error.contains("q8") || error.contains("Q8"),
            "{}: refusal does not name the tier: {error}",
            route.provider_id()
        );
    }
}

/// sc-22736 (epic sc-22723, E1): the weights-free registry surface is per-cell, and no string it
/// mints can ever be mistaken for a measured production identity.
///
/// Both sets are DERIVED from the same route x backend x tier grid rather than frozen, so a new
/// route or a new tier is swept the moment `WanI2vRoute::ALL` or the tier ladder moves.
///
/// Mutation that fails this: dropping the `-weights-free-conformance-` infix (the two sets then
/// collide cell for cell), or keying the weights-free string on fewer coordinates than the
/// production one (two cells then share one fixture identity).
#[test]
fn the_weights_free_a14b_namespace_is_per_cell_and_disjoint_from_production() {
    let mut production = BTreeSet::new();
    let mut weights_free = BTreeSet::new();
    let mut cells = 0_usize;
    for route in WanI2vRoute::ALL {
        for backend in [WanI2vBackend::Mlx, WanI2vBackend::Candle] {
            for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                let tier = MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                };
                let Some(free) = weights_free_calibration_fingerprint(route, backend, tier) else {
                    // A route this authority does not name has neither string.
                    assert_eq!(
                        production_calibration_fingerprint(route, backend, tier),
                        None,
                        "{} has a production string but no weights-free sibling",
                        route.provider_id()
                    );
                    continue;
                };
                let named = production_calibration_fingerprint(route, backend, tier)
                    .expect("a named cell has both strings");
                assert!(
                    weights_free.insert(free.clone()),
                    "{free} names more than one cell"
                );
                assert!(
                    production.insert(named.clone()),
                    "{named} names more than one cell"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 12, "two A14B routes x two backends x three tiers");
    assert!(
        production.is_disjoint(&weights_free),
        "the weights-free namespace overlaps production: {:?}",
        production.intersection(&weights_free).collect::<Vec<_>>()
    );
}

/// The weights-free contract touches no filesystem, keeps the route's declared shape, and
/// publishes the isolated identity at the spec's own load shape — which is exactly what a
/// registry catalog walk on a host with no weights consumes.
#[test]
fn the_weights_free_contract_is_filesystem_free_and_publishes_the_isolated_identity() {
    for route in [WanI2vRoute::T2v14b, WanI2vRoute::I2v14b] {
        for backend in [WanI2vBackend::Mlx, WanI2vBackend::Candle] {
            for (quant, spec_of) in [
                (
                    None,
                    LoadSpec::new(WeightsSource::Dir(
                        "/__sceneworks_memory_contract_surface__".into(),
                    )),
                ),
                (
                    Some(Quant::Q4),
                    LoadSpec::new(WeightsSource::Dir(
                        "/__sceneworks_memory_contract_surface__".into(),
                    ))
                    .with_quant(Quant::Q4),
                ),
                (
                    Some(Quant::Q8),
                    LoadSpec::new(WeightsSource::Dir(
                        "/__sceneworks_memory_contract_surface__".into(),
                    ))
                    .with_quant(Quant::Q8),
                ),
            ] {
                let contract = weights_free_contract(route.provider_id(), backend, &spec_of)
                    .unwrap_or_else(|error| {
                        panic!("{:?} {} {quant:?}: {error}", backend, route.provider_id())
                    });
                assert_eq!(contract.provider_id, route.provider_id());
                assert_eq!(contract.backend, backend.realization());
                assert_eq!(contract.load_shape, spec_of.load_shape);
                assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
                let tier = MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                };
                let identity = contract
                    .calibration
                    .as_ref()
                    .expect("a weights-free A14B contract is calibratable");
                assert_eq!(
                    identity.fingerprint,
                    weights_free_calibration_fingerprint(route, backend, tier).unwrap()
                );
                assert_eq!(identity.load_shape, spec_of.load_shape);
            }
        }
    }
    // Ti2V-5B and the VACE routes are owned elsewhere and must be refused rather than answered.
    for route in [WanI2vRoute::Ti2v5b, WanI2vRoute::Vace, WanI2vRoute::VaceFun] {
        let spec = LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert!(
            weights_free_contract(route.provider_id(), WanI2vBackend::Mlx, &spec).is_err(),
            "{} must not resolve a weights-free contract here",
            route.provider_id()
        );
    }
}

/// The constants the weights-free surface hands a behavior fixture are READ BACK through the
/// route's own menus, so a manifest/menu change cannot leave them naming a bucket or a rate the
/// route stopped admitting.
#[test]
fn a14b_weights_free_fixture_geometry_is_admissible() {
    let (width, height, frames) = A14B_FIXTURE_GEOMETRY;
    for route in [WanI2vRoute::T2v14b, WanI2vRoute::I2v14b] {
        assert!(
            route.public_geometries().contains(&(width, height)),
            "{}: {width}x{height} left the public bucket list",
            route.provider_id()
        );
        assert!(
            route.accepts_rate(A14B_PUBLIC_FPS, frames),
            "{}: {frames}@{A14B_PUBLIC_FPS} left the public rate menu",
            route.provider_id()
        );
        // The A14B menu really is single-rate; if that ever stops being true the weights-free
        // surface has to carry the rate instead of assuming it.
        for other in [8_u32, 24, 25, 30] {
            assert!(
                !route.accepts_rate(other, frames),
                "{}: the A14B rate menu gained {other} fps",
                route.provider_id()
            );
        }
        assert_eq!(
            a14b_public_mode_key(route),
            Some(if route == WanI2vRoute::T2v14b {
                "text_to_video"
            } else {
                "image_to_video"
            })
        );
    }
}
