use super::*;

/// Actual tokenizer/parser path only: no tensor, stream, model or admission.
/// Fresh paths ensure the successful per-path memo cannot mask source corruption.
#[test]
fn gemma_window_preserves_missing_and_malformed_tokenizer_failures() {
    let valid = tempfile::tempdir().unwrap();
    let path = valid.path().join("tokenizer.json");
    std::fs::write(
        &path,
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/gen-core/tests/fixtures/tiny_gemma3_tokenizer.json"
        )),
    )
    .unwrap();
    assert!(gemma_window(&path).unwrap() >= MAX_SEQUENCE_LENGTH as u64 - 2);
    for missing in [true, false] {
        let source = tempfile::tempdir().unwrap();
        let path = source.path().join("tokenizer.json");
        if !missing {
            std::fs::write(&path, b"not tokenizer json").unwrap();
        }
        assert!(matches!(gemma_window(&path), Err(Error::Msg(reason)) if reason.starts_with("tokenizer: ")),
            "missing={missing}: genuine tokenizer corruption must remain a parser failure, not Unsupported/no-evidence");
    }
}

#[test]
fn retired_projection_selects_real_subwindows_without_erasing_original_feature() {
    let cfg = DcAeConfig::sana_f32c32();
    for encode in [false, true] {
        let mut ordinary = Quanta::default();
        let mut retired = Quanta::default();
        dcae_block_quanta(
            &cfg,
            3,
            16,
            24,
            encode,
            SanaPricingRegime::Ordinary,
            &mut ordinary,
        );
        dcae_block_quanta(
            &cfg,
            3,
            16,
            24,
            encode,
            SanaPricingRegime::RetiredLifecycle,
            &mut retired,
        );
        assert_eq!(ordinary.0.len(), 1);
        assert_eq!(retired.0.len(), 2);
        let (attention, glu, combined) = if encode {
            (
                labels::ENCODER_ATTENTION,
                labels::ENCODER_GLU,
                labels::ENCODER_BLOCK,
            )
        } else {
            (
                labels::DECODER_ATTENTION,
                labels::DECODER_GLU,
                labels::DECODER_BLOCK,
            )
        };
        assert_eq!(ordinary.0[0].1, combined);
        assert_eq!(retired.0[0].1, attention);
        assert_eq!(retired.0[1].1, glu);
        assert_eq!(
            ordinary.0[0].0,
            retired.0[0].0 + retired.0[1].0,
            "ordinary single graph retains both halves; this is no renamed whole guard"
        );
        assert_eq!(
            retired.0[1].3,
            tensor_bytes(4, &[1, 16, 24, cfg.block_out_channels[3] as u64]),
            "original feature survives the attention join while GLU reads attended output"
        );
        assert_eq!(retired.0[0].2, Some((16, 24)));
        assert_eq!(retired.0[1].2, Some((16, 24)));
    }
}

/// Pure source recipe, not a measurement: every decode route must price its
/// real final conversion and ONE decoded input backing (the NCHW is a view).
#[test]
fn decoded_pixels_window_is_present_once_after_every_decode_route() {
    for no_tail in [false, true] {
        let mut cfg = DcAeConfig::sana_f32c32();
        if no_tail {
            cfg.block_types.fill(BlockType::EfficientVit);
        }
        for (lh, lw, tiling) in [
            (2, 3, None),
            (2, 3, Some((192, 48))),
            (8, 12, Some((192, 48))),
        ] {
            let mut quanta = Quanta::default();
            decode_quanta(&cfg, lh, lw, tiling, &mut quanta);
            let elements = 3 * lh * u64::from(LATENT_SCALE) * lw * u64::from(LATENT_SCALE);
            let terminal = quanta.0.last().expect("decode has a terminal window");
            assert_eq!(
                terminal.1,
                labels::DECODED_PIXELS,
                "conversion must not disappear behind an early decode return"
            );
            assert_eq!(
                quanta
                    .0
                    .iter()
                    .filter(|q| q.1 == labels::DECODED_PIXELS)
                    .count(),
                1
            );
            assert_eq!(
                terminal.0,
                mlx_gen::ios_quantum::decoded_to_image_bytes_from_metadata(elements)
            );
            assert_eq!(terminal.2, Some((lh * 32, lw * 32)));
            let unscaled = tensor_bytes(4, &[1, cfg.latent_channels as u64, lh, lw]);
            assert_eq!(
                terminal.3,
                tensor_bytes(4, &[elements]) + unscaled,
                "one decoded input (transpose alias), plus the still-owned unscaled latent"
            );
            assert_eq!(ios_quantum_phase(terminal.1), Some(IosPhase::Decode));
        }
    }
}

/// Every quantum label the crate's guards pass — and the foreign labels a SANA request runs (the
/// shared Gemma-2 encoder's, the curated sampler's step) — is attributed to a phase, and each
/// family lands in the phase it runs in.
#[test]
fn every_quantum_label_maps_to_its_phase() {
    for label in labels::ALL
        .iter()
        .chain(GEMMA_LABELS.iter())
        .chain([mlx_gen::sampler::SAMPLER_STEP_QUANTUM].iter())
    {
        assert!(
            ios_quantum_phase(label).is_some(),
            "{label} is not attributed to a phase"
        );
    }
    let phase = |l| ios_quantum_phase(l).unwrap();
    assert_eq!(phase(labels::HEAVY_LOAD), IosPhase::Load);
    assert_eq!(phase(GEMMA_LABELS[1]), IosPhase::Conditioning);
    assert_eq!(phase(labels::ENCODER_BLOCK), IosPhase::Encode);
    assert_eq!(phase(labels::REFERENCE_LATENT), IosPhase::Encode);
    assert_eq!(phase(labels::BLOCK), IosPhase::Denoise);
    assert_eq!(phase(labels::DECODER_BLOCK), IosPhase::Decode);
    assert_eq!(phase(labels::TILE_FINALIZE), IosPhase::Decode);
    assert_ne!(labels::ENCODER_BLOCK, labels::DECODER_BLOCK);
    let mut unique = labels::ALL.to_vec();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), labels::ALL.len(), "a label is shared");
    assert_eq!(ios_quantum_phase("Qwen VAE decode resnet"), None);
}

/// A request the generator refuses is refused by its price too, before any header is read.
#[test]
fn pricing_refuses_unknown_routes_and_unexecutable_levers() {
    use mlx_gen::gen_core::{
        GenerationMemory, LoadSpec, MemoryGeometry, MemoryNumericTier, MemorySelection,
        MemoryStrategy,
    };
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/sana".into()))
        .with_offload_policy(OffloadPolicy::Sequential);
    let selection = MemorySelection {
        strategy: MemoryStrategy::Resident,
        parameters: Default::default(),
        tier: MemoryNumericTier {
            precision: spec.precision,
            quant: None,
            component_precision_floors: &[],
        },
    };
    let req = |provider, mode, execution| {
        let geometry = MemoryGeometry {
            width: 512,
            height: 512,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        (provider, mode, geometry, execution)
    };
    for (provider, mode, geometry, execution) in [
        req("pid", MemoryMode::TextToImage, GenerationMemory::default()),
        req(
            crate::MODEL_ID,
            MemoryMode::Edit,
            GenerationMemory::default(),
        ),
        req(
            crate::MODEL_ID,
            MemoryMode::TextToImage,
            GenerationMemory {
                stream_transformer_blocks: true,
                stage_residency: true,
                ..Default::default()
            },
        ),
        req(
            crate::SPRINT_MODEL_ID,
            MemoryMode::TextToImage,
            GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(128),
                ..Default::default()
            },
        ),
    ] {
        let generation = mlx_gen::gen_core::GenerationRequest {
            width: geometry.width,
            height: geometry.height,
            count: geometry.batch,
            memory: Some(execution),
            ..Default::default()
        };
        let r = IosPricingRequest {
            provider,
            spec: &spec,
            mode: &mode,
            geometry,
            selection: &selection,
            execution: &execution,
            token_window: 0,
            generation: Some(&generation),
            transform: None,
            depth: None,
        };
        assert!(
            matches!(ios_pricing(&r), Err(Error::Unsupported(_))),
            "{provider} {mode:?} {execution:?} must be refused"
        );
    }
}

/// THE PROOF: the production modules run on a tiny snapshot under the CPU quantum fixture — the
/// loads, the Gemma-2 encoder, the bounded reference encode, the trunk's sampler loop at a chunked
/// cross-attention budget, the generator's latent batch and the tiled DC-AE decode — and the
/// largest CHECKED quantum of every phase is exactly the price's scratch and binding label, for base
/// SANA text-to-image and Sprint img2img. The pricing reads only the fixture's safetensors headers.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn observed_peak_quantum_per_phase_is_the_priced_one() {
    use crate::dc_ae::{DcAeDecoder, DcAeEncoder};
    use crate::pipeline::{SanaConditioning, SanaGenerateRequest, SanaHeavy};
    use crate::transformer::SanaTransformer;
    use mlx_gen::ios_quantum::QuantumEvent;
    use mlx_gen::weights::Weights;
    use mlx_rs::Array;

    const CHILD: &str = "SANA_IOS_PRICING_PROOF_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--include-ignored",
                "--exact",
                "ios_pricing::tests::observed_peak_quantum_per_phase_is_the_priced_one",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::test_metal_guard::forbid_metal());

    // Tiny but structurally real: a 6-stage x32 DC-AE ([R,R,R,E,E,E]) whose 32-channel latent is
    // the one `pipeline::create_noise` draws, a two-block trunk over it, a two-layer Gemma-2.
    let dcae = {
        use crate::config::BlockType::{EfficientVit as E, Res as R};
        DcAeConfig {
            in_channels: 3,
            latent_channels: 32,
            attention_head_dim: 8,
            block_out_channels: vec![8, 8, 16, 16, 32, 32],
            layers_per_block: vec![1; 6],
            encoder_layers_per_block: vec![1; 6],
            block_types: vec![R, R, R, E, E, E],
            qkv_multiscales: vec![5],
            upsample_interpolate: true,
            norm_eps: 1e-5,
            attn_eps: 1e-15,
            scaling_factor: 0.41407,
        }
    };
    let gemma = Gemma2Config {
        hidden_size: 64,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        intermediate_size: 128,
        ..Gemma2Config::gemma_2_2b()
    };
    let (vocab, window, caption) = (256i32, 24i32, 16i32);
    let edge = 256u32;
    // 64 tokens x 16 caption keys x 2 heads = 2048 scores: a 512-score budget chunks 4 ways.
    let attention = AttentionBudget::from_score_elements(512, true);
    let (decode_tile, decode_overlap, encode_core) = (128, 32, 64);

    let rand = |shape: &[i32], seed: u64, dtype: Dtype| {
        let a = mlx_rs::random::normal::<f32>(
            shape,
            None,
            None,
            Some(&mlx_rs::random::key(seed).unwrap()),
        )
        .unwrap()
        .multiply(Array::from_f32(0.05))
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
        a.eval().unwrap();
        a
    };
    let save = |dir: &Path, file: &str, map: &HashMap<String, Array>| {
        std::fs::create_dir_all(dir).unwrap();
        Array::save_safetensors(map.iter(), None, dir.join(file)).unwrap();
    };
    let gemma_map = || {
        let mut m = HashMap::new();
        let mut seed = 7000u64;
        let mut put = |k: String, shape: &[i32]| {
            seed += 1;
            m.insert(k, rand(shape, seed, Dtype::Bfloat16));
        };
        let (h, hd) = (gemma.hidden_size, gemma.head_dim);
        put("model.embed_tokens.weight".into(), &[vocab, h]);
        put("model.norm.weight".into(), &[h]);
        for i in 0..gemma.num_layers {
            let p = format!("model.layers.{i}");
            for n in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                put(format!("{p}.{n}.weight"), &[h]);
            }
            put(
                format!("{p}.self_attn.q_proj.weight"),
                &[gemma.num_heads * hd, h],
            );
            put(
                format!("{p}.self_attn.k_proj.weight"),
                &[gemma.num_kv_heads * hd, h],
            );
            put(
                format!("{p}.self_attn.v_proj.weight"),
                &[gemma.num_kv_heads * hd, h],
            );
            put(
                format!("{p}.self_attn.o_proj.weight"),
                &[h, gemma.num_heads * hd],
            );
            put(
                format!("{p}.mlp.gate_proj.weight"),
                &[gemma.intermediate_size, h],
            );
            put(
                format!("{p}.mlp.up_proj.weight"),
                &[gemma.intermediate_size, h],
            );
            put(
                format!("{p}.mlp.down_proj.weight"),
                &[h, gemma.intermediate_size],
            );
        }
        m
    };

    let tmp = tempfile::tempdir().unwrap();
    for (sprint, img2img) in [(false, false), (true, true)] {
        let root = tmp.path().join(format!("sprint{sprint}-img2img{img2img}"));
        let trunk_cfg = SanaTransformerConfig {
            in_channels: 32,
            out_channels: 32,
            num_layers: 2,
            caption_channels: gemma.hidden_size,
            ..crate::transformer::quantum_proofs::cfg(2, 16, 2.5, sprint)
        };
        let mut b = crate::transformer::quantum_proofs::Builder {
            map: HashMap::new(),
            seed: 100,
            dtype: Dtype::Float16,
        };
        b.trunk(&trunk_cfg);
        for i in 0..trunk_cfg.num_layers {
            b.block(&format!("transformer_blocks.{i}"), &trunk_cfg);
        }
        save(
            &root.join("transformer"),
            "diffusion_pytorch_model.safetensors",
            &b.map,
        );
        save(
            &root.join("vae"),
            "diffusion_pytorch_model.safetensors",
            &crate::dc_ae::quantum_proofs::weight_map(&dcae),
        );
        save(
            &root.join("text_encoder"),
            "gemma-2-2b-it.safetensors",
            &gemma_map(),
        );

        let facts = SanaPricingFacts {
            root: root.clone(),
            transformer: trunk_cfg.clone(),
            dc_ae: dcae.clone(),
            gemma: gemma.clone(),
            gemma_tokens: window as u64,
            caption_tokens: caption as u64,
            text_encoder: Inventory::read(&root.join("text_encoder")).unwrap(),
            trunk: Inventory::read(&root.join("transformer")).unwrap(),
            vae: Inventory::read(&root.join("vae")).unwrap(),
        };
        let plan = SanaPricingPlan {
            sprint,
            sequential: true,
            img2img,
            width: edge,
            height: edge,
            count: 1,
            decode_tiling: Some((decode_tile, decode_overlap)),
            encode_tile: Some(encode_core),
            attention,
        };
        let priced = price(&facts, &plan).unwrap();
        let quanta = priced_quanta(&facts, &plan).unwrap();

        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let sink = seen.clone();
        mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 36, || {
            mlx_gen::ios_quantum::with_cpu_diagnostic_observer(
                move |e| sink.borrow_mut().push(e.clone()),
                || {
                    let cancel = mlx_gen::CancelFlag::default();
                    // Loads: each checks its floor, then fails on the tiny (non-2b) tensors.
                    assert!(crate::model::load_text_encoder_component(&root).is_err());
                    if img2img {
                        assert!(crate::model::load_reference_encoder(&root).is_err());
                    }
                    assert!(crate::model::load_heavy(&root, sprint, false, false).is_err());

                    // Conditioning: the Gemma-2 forward over its padded window.
                    let w =
                        Weights::from_file(root.join("text_encoder/gemma-2-2b-it.safetensors"))?;
                    let encoder = crate::text_encoder::Gemma2::from_weights(&w, "model.", &gemma)?;
                    let ids: Vec<i32> = (0..window).map(|i| (i * 7) % vocab).collect();
                    let pad = vec![1i32; window as usize];
                    let hidden = encoder.forward(
                        &Array::from_slice(&ids, &[1, window]),
                        Some(&Array::from_slice(&pad, &[1, window])),
                    )?;
                    let cond =
                        mlx_gen::attention::slice_axis(&hidden, 1, window - caption, caption)?;
                    let mask = Array::from_slice(&vec![1i32; caption as usize], &[1, caption]);
                    mlx_rs::transforms::eval([&cond, &mask])?;
                    drop(encoder);
                    let conditioning = SanaConditioning {
                        cond: cond.clone(),
                        cond_mask: mask.clone(),
                        uncond: (!sprint).then(|| (cond.clone(), mask.clone())),
                    };

                    // Encode: the bounded reference encode + its scaled latent.
                    let image = mlx_gen::Image {
                        width: edge,
                        height: edge,
                        pixels: (0..edge * edge * 3).map(|i| (i % 251) as u8).collect(),
                    };
                    let req = SanaGenerateRequest {
                        width: edge,
                        height: edge,
                        steps: Some(2),
                        seed: Some(3),
                        init_image: img2img.then_some(&image),
                        strength: img2img.then_some(0.9),
                        ..SanaGenerateRequest::new("a fox")
                    };
                    let vae = Weights::from_dir(root.join("vae"))?;
                    let reference = if img2img {
                        let encoder = DcAeEncoder::from_weights(&vae, dcae.clone())?;
                        let r = crate::pipeline::prepare_reference_latent(
                            Some(&encoder),
                            &dcae,
                            sprint,
                            &req,
                            &cancel,
                            Some(encode_core),
                        )?;
                        assert!(r.is_some(), "the img2img case must encode");
                        r
                    } else {
                        None
                    };

                    // Denoise: the variant's sampler loop at the chunked budget, then the batch.
                    let trunk = SanaTransformer::from_weights(
                        &Weights::from_dir(root.join("transformer"))?,
                        trunk_cfg.clone(),
                    )?;
                    let decoder = DcAeDecoder::from_weights(&vae, dcae.clone())?;
                    let heavy = if sprint {
                        SanaHeavy::new_sprint_text_to_image(trunk, decoder, dcae.clone(), 0.1)
                    } else {
                        SanaHeavy::new_text_to_image(trunk, decoder, dcae.clone())
                    };
                    let latent = heavy.denoise_one_with_prepared_reference(
                        &conditioning,
                        &req,
                        4.5,
                        reference.as_ref(),
                        &cancel,
                        &mut |_| {},
                        &mlx_gen::PreviewSink::default(),
                        crate::transformer::SanaForwardPlan {
                            attention,
                            window: None,
                        },
                    )?;
                    crate::model::materialize_latent_batch(std::slice::from_ref(&latent))?;
                    drop(heavy);

                    // Decode: the tiled DC-AE decode.
                    let decoder = DcAeDecoder::from_weights(&vae, dcae.clone())?;
                    crate::pipeline::decode_to_image_with_tiling(
                        &decoder,
                        &dcae,
                        &latent,
                        &cancel,
                        Some(&TilingConfig::spatial_only(decode_tile, decode_overlap)),
                    )?;
                    Ok(())
                },
            )
        })
        .unwrap();

        // Largest checked quantum of each phase, every label attributed.
        let mut observed: BTreeMap<IosPhase, (u64, String)> = BTreeMap::new();
        for e in seen.borrow().iter() {
            if let QuantumEvent::Checked { label, scratch } = e {
                let phase =
                    ios_quantum_phase(label).unwrap_or_else(|| panic!("{label} has no phase"));
                let slot = observed.entry(phase).or_insert((0, String::new()));
                if *scratch > slot.0 {
                    *slot = (*scratch, label.clone());
                }
            }
        }
        // The largest PRICED quantum of each phase (the folded price binds the phase's worst
        // resident + scratch, which a quantum with less scratch but more resident can win: the
        // staged reference-encoder load holds the preprocessed image).
        let mut largest: BTreeMap<IosPhase, (u64, String)> = BTreeMap::new();
        for q in &quanta {
            let slot = largest.entry(q.phase).or_insert((0, String::new()));
            if q.scratch_bytes > slot.0 {
                *slot = (q.scratch_bytes, q.label.to_string());
            }
        }
        println!(
            "sprint={sprint} img2img={img2img}\n  observed {observed:?}\n  priced   {largest:?}"
        );
        assert_eq!(
            observed.keys().collect::<Vec<_>>(),
            priced.phases.iter().map(|p| &p.phase).collect::<Vec<_>>(),
            "the priced phases are the phases the request runs"
        );
        assert_eq!(observed, largest, "sprint={sprint} img2img={img2img}");
        // Each phase's binding quantum was checked at exactly its priced scratch.
        for p in &priced.phases {
            assert!(
                seen.borrow().iter().any(|e| matches!(
                    e,
                    QuantumEvent::Checked { label, scratch }
                        if *label == p.binding && *scratch == p.scratch_bytes
                )),
                "{:?}: binding {} at {} B was never checked at that scratch",
                p.phase,
                p.binding,
                p.scratch_bytes
            );
        }
        // Stronger than the phase maxima: every label's largest checked scratch is the largest
        // priced for it, and no label is priced that the request does not run (or vice versa).
        let mut observed: BTreeMap<String, u64> = BTreeMap::new();
        for e in seen.borrow().iter() {
            if let QuantumEvent::Checked { label, scratch } = e {
                let v = observed.entry(label.clone()).or_default();
                *v = (*v).max(*scratch);
            }
        }
        let mut priced: BTreeMap<String, u64> = BTreeMap::new();
        for q in &quanta {
            let v = priced.entry(q.label.to_string()).or_default();
            *v = (*v).max(q.scratch_bytes);
        }
        assert_eq!(
            observed, priced,
            "per label, sprint={sprint} img2img={img2img}"
        );
    }

    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED ios_pricing::tests::observed_peak_quantum_per_phase_is_the_priced_one");
}
