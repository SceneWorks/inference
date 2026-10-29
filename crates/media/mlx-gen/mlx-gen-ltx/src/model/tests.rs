use super::*;
// sc-19502: the derived stage-1 step count the descriptor advertises.
use crate::pipeline::NATIVE_STEPS;

const MODEL_SRC: &str = include_str!("../model.rs");

fn ltx25_request(prompt: &str) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.into(),
        width: 512,
        height: 512,
        frames: Some(17),
        seed: Some(0x5eed),
        ..Default::default()
    }
}

fn weights_free_ltx25(variant: TransformerVariant) -> Ltx25 {
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(".")));
    Ltx25 {
        descriptor: descriptor_25_for_variant(variant),
        memory_strategy: crate::memory_strategy_2_5::weights_free_memory_strategy_contract(&spec)
            .unwrap(),
        memory_tier: crate::memory_strategy_2_5::resolved_numeric_tier(&spec).unwrap(),
        spec,
        variant,
        enhancer_cache: Rc::new(RefCell::new(PrefixCache::with_budget(
            LTX25_ENHANCER_PREFIX_CACHE_BYTES,
        ))),
    }
}

#[test]
fn ltx25_split_audio_pipeline_loads_the_vocoder_component() {
    let start = MODEL_SRC.find("fn build_ltx25(").expect("LTX-2.5 builder");
    let end = MODEL_SRC[start..]
        .find("let spatial =")
        .map(|offset| start + offset)
        .expect("end of LTX-2.5 audio construction");
    let wiring = &MODEL_SRC[start..end];

    assert!(wiring.contains("bundle.require(LtxComponent::AudioVae)"));
    assert!(wiring.contains("audio.with_file_name(\"vocoder.safetensors\")"));
    assert!(wiring.contains("AudioDecoder::from_weights(&audio_w"));
    assert!(wiring.contains("LtxVocoder::from_weights(&vocoder_w"));
    assert!(!wiring.contains("LtxVocoder::from_weights(&audio_w"));
}

#[test]
fn ltx25_loaded_variant_narrows_steps_and_negative_conditioning() {
    let distilled = descriptor_25_for_variant(TransformerVariant::Distilled);
    assert_eq!(
        distilled.capabilities.supported_steps,
        StepSupport::Exact(vec![NATIVE_STEPS])
    );
    assert!(!distilled.capabilities.supports_negative_prompt);

    let dev = descriptor_25_for_variant(TransformerVariant::Dev);
    assert_eq!(
        dev.capabilities.supported_steps,
        StepSupport::Exact(vec![30])
    );
    assert!(dev.capabilities.supports_negative_prompt);
    assert!(dev.capabilities.samplers.is_empty());

    let dev_request = GenerationRequest {
        steps: Some(30),
        negative_prompt: Some("blurred motion".into()),
        ..ltx25_request("a red kite over the sea")
    };
    assert!(validate_request_for(MODEL_25_ID, &dev.capabilities, &dev_request).is_ok());
    assert!(validate_request_for(MODEL_25_ID, &distilled.capabilities, &dev_request).is_err());
    assert!(validate_request_for(
        MODEL_25_ID,
        &dev.capabilities,
        &GenerationRequest {
            steps: Some(NATIVE_STEPS),
            negative_prompt: None,
            ..ltx25_request("a red kite over the sea")
        }
    )
    .is_err());
}

/// Both alternate routes are ordinary provider calls, so the staged negative contexts and exact
/// checkpoint identity must cross the model boundary into DFR and IC-LoRA rather than only the
/// plain grid path. This deliberately names the two production call sites; removing either
/// argument or substituting a default variant makes the test fail.
#[test]
fn ltx25_provider_threads_dev_contexts_and_variant_to_dfr_and_iclora() {
    let start = MODEL_SRC
        .find("pub(crate) fn generate_av_from_embeddings(")
        .expect("model.rs must define the ordinary A/V provider route");
    let end = MODEL_SRC
        .find("\nmlx_gen::register_generators!")
        .expect("the provider route must precede registration");
    let body = &MODEL_SRC[start..end];
    let dfr_start = body
        .find("&DfrComponents {")
        .expect("ordinary DFR provider route must build DfrComponents");
    let dfr_end = body[dfr_start..]
        .find("},\n                &DfrRequest")
        .map(|offset| dfr_start + offset)
        .expect("DFR components must end before its request");
    let dfr = &body[dfr_start..dfr_end];
    for field in [
        "negative_video_ctx,",
        "negative_audio_ctx,",
        "variant: self.transformer_variant(),",
    ] {
        assert!(
            dfr.contains(field),
            "ordinary DFR route no longer threads `{field}` into its stage-one executor:\n{dfr}"
        );
    }

    let iclora_start = body
        .find("generate_av_latents_iclora(")
        .expect("extend_clip/video_bridge must call the IC-LoRA executor");
    let iclora_end = body[iclora_start..]
        .find(".map(|(video, audio)|")
        .map(|offset| iclora_start + offset)
        .expect("IC-LoRA call must return into the provider output mapping");
    let iclora = &body[iclora_start..iclora_end];
    for argument in [
        "negative_video_ctx,",
        "negative_audio_ctx,",
        "self.transformer_variant(),",
    ] {
        assert!(
                iclora.contains(argument),
                "ordinary IC-LoRA route no longer forwards `{argument}` to its stage-one executor:\n{iclora}"
            );
    }
}

/// This reaches the same request object `generate_impl` supplies to auto-duration, DFR,
/// seeded noise, and the render pipeline.  It catches the easy false green where enhancement
/// is computed/logged but the original prompt remains in downstream conditioning.
#[test]
fn ltx25_seeded_t2v_and_i2v_enhanced_requests_equal_explicit_prompt_requests() {
    let enhanced = "a low tracking shot through rain-lit streets".to_string();
    let t2v = ltx25_request("rainy city");
    let routed_t2v = request_with_effective_prompt(&t2v, Some(enhanced.clone()));
    assert_eq!(routed_t2v.prompt, enhanced);
    assert_eq!(routed_t2v.seed, t2v.seed);
    assert_eq!(routed_t2v.conditioning.len(), 0);

    let image = Image {
        width: 4,
        height: 4,
        pixels: vec![0; 4 * 4 * 3],
    };
    let i2v = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image,
            strength: Some(0.8),
        }],
        ..ltx25_request("rainy city")
    };
    let routed_i2v = request_with_effective_prompt(&i2v, Some(enhanced));
    assert_eq!(
        routed_i2v.prompt,
        "a low tracking shot through rain-lit streets"
    );
    assert_eq!(routed_i2v.seed, i2v.seed);
    assert!(matches!(
        routed_i2v.conditioning.as_slice(),
        [Conditioning::Reference {
            strength: Some(0.8),
            ..
        }]
    ));
}

#[test]
fn ltx25_production_enhancer_reaches_exact_gemma4_t2v_i2v_and_shared_decode_paths() {
    let start = MODEL_SRC
        .find("fn maybe_enhance_gemma4(")
        .expect("2.5 enhancer production method");
    let end = MODEL_SRC[start..]
        .find("\n}\n\n/// Prepare the actual Gemma-4 I2V prefill")
        .map(|offset| start + offset)
        .expect("enhancer method ends before I2V prefill helper");
    let route = &MODEL_SRC[start..end];
    for required in [
        "ltx25_enhancer_turn(&req.prompt, reference.is_some())",
        "ltx25_i2v_enhancer_prefill(&model, &mm, prompt_ids, image)?",
        "enhance::clamp_gemma4_max_tokens",
        "&mut cache.borrow_mut()",
    ] {
        assert!(
            route.contains(required),
            "missing `{required}` from:\n{route}"
        );
    }
    assert!(!route.contains("SampleParams::censored"));
    assert!(!route.contains("enhance::I2V_SYSTEM_PROMPT"));

    let vision_start = MODEL_SRC
        .find("fn ltx25_i2v_enhancer_prefill(")
        .expect("I2V enhancer prefill helper");
    let vision_end = MODEL_SRC[vision_start..]
        .find("\n}\n\n/// Build the enhancer chat renderer")
        .map(|offset| vision_start + offset)
        .expect("vision helper ends before template helper");
    let vision = &MODEL_SRC[vision_start..vision_end];
    for required in [
        "resize_bicubic_u8(",
        "gemma4_mm::patchify(",
        ".forward(&patches, (gh, gw))",
        "gemma4_mm::expand_framed_placeholders(",
        "model.splice_image_features(",
    ] {
        assert!(
            vision.contains(required),
            "missing `{required}` from:\n{vision}"
        );
    }

    let enhance_src = include_str!("../enhance.rs");
    for required in [
        "temperature: 0.0",
        "GEMMA4_NO_REPEAT_NGRAM",
        "generate_cached_with(",
        "generate_from_prefill(",
    ] {
        assert!(
            enhance_src.contains(required),
            "shared Gemma-4 decode path lost `{required}`"
        );
    }
}

#[test]
fn ltx25_gemma4_t2v_and_i2v_turns_match_upstream_contracts() {
    let (t2v_system, t2v_user) = ltx25_enhancer_turn("A red kite", false);
    assert_eq!(t2v_system, enhance::GEMMA4_T2V_SYSTEM_PROMPT);
    assert_eq!(t2v_user, "user prompt: A red kite");

    let (i2v_system, i2v_user) = ltx25_enhancer_turn("A red kite", true);
    assert_eq!(i2v_system, enhance::GEMMA4_I2V_SYSTEM_PROMPT);
    assert_eq!(i2v_user, "<|image|>User Raw Input Prompt: A red kite.");
}

#[test]
fn ltx25_provider_enhancement_is_registered_and_refuses_the_unlicensed_axis() {
    let generator = weights_free_ltx25(TransformerVariant::Distilled);
    let enhanced = GenerationRequest {
        enhance_prompt: true,
        ..ltx25_request("a paper kite")
    };
    assert!(Generator::validate(&generator, &enhanced).is_ok());
    let temperature = GenerationRequest {
        enhance_temperature: Some(0.7),
        ..enhanced.clone()
    };
    let err = Generator::validate(&generator, &temperature)
        .unwrap_err()
        .to_string();
    assert!(err.contains("greedy"), "{err}");
    let forbidden = GenerationRequest {
        use_uncensored_enhancer: true,
        ..enhanced
    };
    assert!(Generator::validate(&generator, &forbidden)
        .unwrap_err()
        .to_string()
        .contains("use_uncensored_enhancer"));
}

#[test]
fn ltx25_enhancer_asset_gate_fails_loudly_before_any_weight_or_network_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let err = validate_ltx25_enhancer_snapshot(dir.path()).unwrap_err();
    assert!(err.to_string().contains("offline google/gemma-4-12b-it"));

    std::fs::write(
        dir.path().join(LTX25_ENHANCER_MANIFEST_FILE),
        LTX25_ENHANCER_MANIFEST,
    )
    .unwrap();
    // Same public Gemma-4 unified identity strings, but not the exact stock config. Pad it to
    // the stock byte length so the content digest, rather than the size check, discriminates it.
    let mut wrong = r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"final_logit_softcapping":30.0}}"#.to_owned();
    wrong.push_str(&" ".repeat(4_423 - wrong.len()));
    std::fs::write(dir.path().join("config.json"), wrong).unwrap();
    let err = validate_ltx25_enhancer_snapshot(dir.path()).unwrap_err();
    assert!(err.to_string().contains("config.json` digest mismatch"));
}

#[test]
fn ltx25_enhancer_publication_manifest_has_the_sealed_offline_inventory_digest() {
    let manifest: serde_json::Value =
        serde_json::from_str(LTX25_ENHANCER_MANIFEST).expect("embedded manifest JSON");
    assert_eq!(
        manifest_string(&manifest, "source_repository").unwrap(),
        LTX25_ENHANCER_REPOSITORY
    );
    assert_eq!(
        manifest_string(&manifest, "source_revision").unwrap(),
        LTX25_ENHANCER_REVISION
    );
    assert_eq!(
        ltx25_enhancer_inventory_sha256(&manifest).unwrap(),
        LTX25_ENHANCER_INVENTORY_SHA256
    );
}

#[test]
fn ltx25_registered_route_enforces_geometry_and_temporal_constraints() {
    let generator = weights_free_ltx25(TransformerVariant::Distilled);
    let request = GenerationRequest {
        prompt: "a quiet moonlit harbor".into(),
        width: 512,
        height: 512,
        frames: Some(17),
        ..Default::default()
    };
    Generator::validate(&generator, &request).expect("64px / 1+8*k request must pass");

    let mut geometry = request.clone();
    geometry.width = 672; // divisible by 32, but not the provider's 64px stage-1 stride.
    assert!(Generator::validate(&generator, &geometry)
        .unwrap_err()
        .to_string()
        .contains("divisible by 64"));

    let mut temporal = request;
    temporal.frames = Some(16);
    assert!(Generator::validate(&generator, &temporal)
        .unwrap_err()
        .to_string()
        .contains("1 + 8"));
}

#[test]
fn ltx25_provider_admits_only_advanced_axes_it_executes() {
    let generator = weights_free_ltx25(TransformerVariant::Distilled);
    let request = GenerationRequest {
        prompt: "a quiet moonlit harbor".into(),
        width: 512,
        height: 512,
        frames: Some(17),
        num_generated_keyframes: Some(1),
        ..Default::default()
    };
    assert!(Generator::validate(&generator, &request).is_ok());
    let mut temporal = request.clone();
    temporal.num_generated_keyframes = None;
    temporal.temporal_upsample_rounds = Some(1);
    assert!(Generator::validate(&generator, &temporal).is_ok());
    let mut automatic = request;
    automatic.num_generated_keyframes = None;
    automatic.auto_duration = Some(mlx_gen::gen_core::duration_head::AutoDurationRange {
        min_seconds: 1.0,
        max_seconds: 2.0,
    });
    assert!(Generator::validate(&generator, &automatic).is_ok());
    assert!(
        generator
            .descriptor()
            .capabilities
            .supports_diffusion_decoder
    );
}

/// The DurationHead predictor is not merely admitted by the descriptor: its value becomes the
/// request's actual frame plan before the provider allocates video/audio noise.  The injected
/// spy is the production seam (`DurationHead::predict_seconds`) and makes a deleted/bypassed
/// call fail this test instead of leaving a default-frame false green.
#[test]
fn ltx25_auto_duration_prediction_drives_the_provider_frame_plan() {
    let request = GenerationRequest {
        prompt: "a slow orbit around a lighthouse".into(),
        width: 512,
        height: 512,
        fps: Some(24),
        auto_duration: Some(mlx_gen::gen_core::duration_head::AutoDurationRange {
            min_seconds: 2.0,
            max_seconds: 8.0,
        }),
        ..Default::default()
    };
    let calls = std::cell::Cell::new(0usize);
    let planned = apply_predicted_frames(&request, 24.0, &mut || {
        calls.set(calls.get() + 1);
        Ok(3.0)
    })
    .expect("duration head result must resolve to provider frames");
    assert_eq!(
        calls.get(),
        1,
        "the real predictor seam must be called once"
    );
    assert_ne!(
        planned.frames, request.frames,
        "prediction must affect the plan"
    );
    assert_eq!(planned.frames, Some(65));
    assert_ne!(Ltx::latent_dims(&planned).0, Ltx::latent_dims(&request).0);

    let explicit = GenerationRequest {
        frames: Some(17),
        ..request
    };
    let explicit_plan = apply_predicted_frames(&explicit, 24.0, &mut || {
        panic!("explicit frames must bypass the duration-head predictor")
    })
    .expect("explicit frame count wins");
    assert_eq!(explicit_plan.frames, Some(17));
}

/// This is the request-side DFR plan consumed by the ordinary provider's
/// `generate_dfr_av_latents` branch.  A temporal request changes both the noise canvas and the
/// trim contract; a count request changes the actual slot positions.  Either call being
/// deleted or replaced with the plain two-stage plan makes one of these assertions fail.
#[test]
fn ltx25_dfr_request_plan_drives_temporal_and_generated_slot_execution() {
    let temporal = GenerationRequest {
        prompt: "fast tracking shot through a market".into(),
        width: 512,
        height: 512,
        frames: Some(153),
        temporal_upsample_rounds: Some(2),
        ..Default::default()
    };
    let temporal_plan = plan_dfr_request(&temporal)
        .expect("DFR temporal plan")
        .expect("temporal request must select DFR");
    assert_eq!(temporal_plan.requested_frames, 153);
    assert_eq!(
        temporal_plan.canvas_frames, 161,
        "DFR must pad before denoise"
    );
    assert_eq!(temporal_plan.temporal_upsample_rounds, 2);
    assert_eq!(temporal_plan.keyframe_positions, vec![32, 64, 96, 128, 160]);

    let dfr_calls = std::cell::Cell::new(0usize);
    let plain_calls = std::cell::Cell::new(0usize);
    let mut ignored_progress = |_| {};
    let branch = dispatch_dfr(
        Some(&temporal_plan),
        &mut ignored_progress,
        |plan, _| {
            dfr_calls.set(dfr_calls.get() + 1);
            Ok((plan.canvas_frames, plan.temporal_upsample_rounds))
        },
        |_| {
            plain_calls.set(plain_calls.get() + 1);
            Ok((0, 0))
        },
    )
    .expect("a DFR plan must dispatch to the DFR execution branch");
    assert_eq!(branch, (161, 2));
    assert_eq!(dfr_calls.get(), 1);
    assert_eq!(plain_calls.get(), 0);

    let generated = GenerationRequest {
        num_generated_keyframes: Some(3),
        frames: Some(121),
        ..temporal
    };
    let generated_plan = plan_dfr_request(&generated)
        .expect("generated-slot plan")
        .expect("generated keyframes must select DFR");
    assert_eq!(generated_plan.keyframe_positions, vec![30, 60, 90]);
    assert_eq!(
        mlx_gen::gen_core::ltx_dfr::dfr_target_frames(
            i64::from(temporal_plan.requested_frames),
            temporal_plan.temporal_upsample_rounds,
        ),
        609,
        "the DFR trim contract must survive two real temporal rounds"
    );
}

/// A staged diffusion VAE is a provider execution choice, not an extra file the conv path may
/// quietly ignore.  The spy observes the exact budget/mode passed into the same dispatch that
/// `decode_video` uses, so deleting that call or replacing it with `decode_seeded` goes red.
#[test]
fn ltx25_staged_diffvae_executes_the_budgeted_decoder_mode() {
    let conv = LoadSpec::new(WeightsSource::Dir(PathBuf::from(".")));
    assert_eq!(ltx25_decoder_selection(&conv), Ltx25DecoderSelection::Conv);

    let diffusion = conv.with_component(
        LtxComponent::DiffusionVideoVae.id(),
        WeightsSource::File(PathBuf::from("/tmp/vae_diffusion_decoder.safetensors")),
    );
    assert_eq!(
        ltx25_decoder_selection(&diffusion),
        Ltx25DecoderSelection::DiffusionBudgeted(DEFAULT_DIFFVAE_MODE)
    );
    assert_eq!(
        ltx25_video_component(&diffusion),
        LtxComponent::DiffusionVideoVae
    );

    let observed = std::cell::Cell::new(None);
    let output = decode_diffvae_budgeted(DEFAULT_DIFFVAE_MODE, |mode| {
        observed.set(Some(mode));
        Ok("budgeted-diffvae")
    })
    .expect("the selected decoder must be invoked");
    assert_eq!(output, "budgeted-diffvae");
    assert_eq!(observed.get(), Some(DiffVaeMode::ChunkedEager));
}

#[test]
fn calibration_fault_is_request_local_and_phase_exact() {
    use mlx_gen::gen_core::{GenerationMemory, MemoryPhase};

    let plain = GenerationRequest::default();
    for phase in [
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ] {
        assert_eq!(
            finish_calibration_phase(&plain, phase, || Ok(17)).unwrap(),
            17
        );

        let phase_without_authorization = GenerationRequest {
            memory: Some(GenerationMemory {
                calibration_error_phase: Some(phase),
                calibration_fault_harness_authorized: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            finish_calibration_phase(&phase_without_authorization, phase, || Ok(19)).unwrap(),
            19,
            "the hidden phase carrier is inert without explicit harness authorization"
        );

        let authorization_without_phase = GenerationRequest {
            memory: Some(GenerationMemory {
                calibration_error_phase: None,
                calibration_fault_harness_authorized: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            finish_calibration_phase(&authorization_without_phase, phase, || Ok(21)).unwrap(),
            21,
            "authorization alone must not invent a fault phase"
        );

        let mut memory = GenerationMemory::default();
        memory.authorize_calibration_fault(phase);
        let injected = GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        let error = finish_calibration_phase(&injected, phase, || Ok(17))
            .expect_err("the selected physical phase must return a typed error")
            .to_string();
        assert!(error.contains(MODEL_ID), "got: {error}");
        assert!(
            error.contains("injected memory-strategy calibration error"),
            "got: {error}"
        );
        assert!(error.contains(&format!("{phase:?}")), "got: {error}");

        let other = if phase == MemoryPhase::Decode {
            MemoryPhase::Denoise
        } else {
            MemoryPhase::Decode
        };
        assert_eq!(
            finish_calibration_phase(&injected, other, || Ok(23)).unwrap(),
            23,
            "a request-local fault must not spill into another phase"
        );
    }
}

#[test]
fn calibration_fault_drops_completed_phase_state_before_warm_recovery() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug)]
    struct DropWitness(Arc<AtomicUsize>);
    impl Drop for DropWitness {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let mut memory = mlx_gen::gen_core::GenerationMemory::default();
    memory.authorize_calibration_fault(mlx_gen::gen_core::MemoryPhase::Denoise);
    let injected = GenerationRequest {
        memory: Some(memory),
        ..Default::default()
    };
    finish_calibration_phase(&injected, mlx_gen::gen_core::MemoryPhase::Denoise, || {
        Ok(DropWitness(drops.clone()))
    })
    .expect_err("the injected request must fail after phase state exists");
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    let warm = finish_calibration_phase(
        &GenerationRequest::default(),
        mlx_gen::gen_core::MemoryPhase::Denoise,
        || Ok(DropWitness(drops.clone())),
    )
    .expect("an unmodified warm request must recover");
    drop(warm);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn calibrated_scope_finishes_injected_error_and_a_fresh_scope_recovers() {
    use mlx_gen::gen_core::{
        LoadSpec, MemoryPhase, MemoryRunOutcome, MemoryStrategy, Quant, WeightsSource,
    };

    fn fixture() -> (
        LoadSpec,
        mlx_gen::gen_core::MemoryProviderContract,
        mlx_gen::gen_core::MemoryBehaviorFixture,
    ) {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-fixture".into()))
            .with_quant(Quant::Q8);
        let contract = crate::memory_strategy::weights_free_memory_strategy_contract(&spec)
            .expect("the provider-owned weights-free contract fixture");
        let fixture = crate::memory_strategy::registered_valid_fixtures(
            &spec,
            &contract,
            MemoryStrategy::StagedResidency,
        )
        .expect("the provider-owned behavior fixture")
        .pop()
        .expect("one staged-residency fixture");
        (spec, contract, fixture)
    }

    let (spec, contract, fixture) = fixture();
    let mut scope =
        crate::memory_strategy::registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .expect("the LTX provider must open a calibrated request scope");
    let mut injected = fixture.request.clone();
    scope.configure_request(&mut injected).unwrap();
    injected
        .memory
        .as_mut()
        .expect("the selected staged rung installs its request carrier")
        .authorize_calibration_fault(MemoryPhase::Denoise);
    scope.enter_phase(MemoryPhase::Conditioning).unwrap();
    scope.leave_phase(MemoryPhase::Conditioning).unwrap();
    scope.enter_phase(MemoryPhase::Denoise).unwrap();
    let error = finish_calibration_phase(&injected, MemoryPhase::Denoise, || Ok(()))
        .expect_err("the authorized provider fault must escape the physical boundary");
    scope.leave_phase(MemoryPhase::Denoise).unwrap();
    scope
        .finish(MemoryRunOutcome::Error {
            message: error.to_string(),
        })
        .expect("an injected provider error must complete scope cleanup");
    assert!(
        scope.finish(MemoryRunOutcome::Complete).is_err(),
        "the error outcome must terminally finish the request scope"
    );

    let mut recovery =
        crate::memory_strategy::registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .expect("a fresh provider request scope must open after the injected error");
    let mut recovered_request = fixture.request;
    recovery.configure_request(&mut recovered_request).unwrap();
    recovery.enter_phase(MemoryPhase::Conditioning).unwrap();
    recovery.leave_phase(MemoryPhase::Conditioning).unwrap();
    recovery.enter_phase(MemoryPhase::Denoise).unwrap();
    finish_calibration_phase(&recovered_request, MemoryPhase::Denoise, || Ok(()))
        .expect("the subsequent unmodified provider request must recover");
    recovery.leave_phase(MemoryPhase::Denoise).unwrap();
    recovery.enter_phase(MemoryPhase::Decode).unwrap();
    finish_calibration_phase(&recovered_request, MemoryPhase::Decode, || Ok(()))
        .expect("recovery must remain healthy through decode");
    recovery.leave_phase(MemoryPhase::Decode).unwrap();
    recovery.finish(MemoryRunOutcome::Complete).unwrap();
}

#[test]
fn replacement_mask_rejects_wrapping_u32_dimensions_without_panicking() {
    let frame = Image {
        width: 65_536,
        height: 65_536,
        pixels: Vec::new(),
    };
    let mask = frame.clone();

    let err = apply_replacement_mask(&frame, &mask, 1.0)
        .expect_err("wrapped dimensions must return a typed error");
    assert!(
        err.to_string().contains("frame/mask must be RGB8"),
        "unexpected error: {err}"
    );
}

/// F-050 (sc-11133): drive `StageProgressFold` over the exact event stream a curated 2nd-order
/// solver produces — each stage's σ-derived `current` forwarded TWICE per step (predictor +
/// corrector) — and assert the folded bar is monotone, reaches `TOTAL_STEPS`, and never overruns.
/// Before the fix the blind counter reached 22 against `total: 11`.
#[test]
fn stage_progress_fold_dedupes_multi_eval_and_reaches_total() {
    let mut fold = StageProgressFold::new(TOTAL_STEPS);
    let mut emitted = Vec::new();
    // Stage 1: 8 steps, each forwarded value seen twice (heun predictor+corrector), monotone
    // per-stage 1..=8. Stage 2: 3 steps, restarts at 1, forwarded 1..=3 twice each.
    let stage1: Vec<u32> = (1..=8).flat_map(|c| [c, c]).collect();
    let stage2: Vec<u32> = (1..=3).flat_map(|c| [c, c]).collect();
    for fwd in stage1.into_iter().chain(stage2) {
        if let Some(cur) = fold.observe(fwd) {
            emitted.push(cur);
        }
    }
    assert_eq!(
        emitted,
        (1..=11).collect::<Vec<_>>(),
        "the folded bar must be exactly 1..=11 (monotone, complete, no overrun)"
    );
}

/// The native distilled path forwards `i+1` once per step (no multi-eval); the fold must still
/// produce the same clean `1..=11` sequence.
#[test]
fn stage_progress_fold_handles_single_eval_native_path() {
    let mut fold = StageProgressFold::new(TOTAL_STEPS);
    let mut emitted = Vec::new();
    for fwd in (1..=8).chain(1..=3) {
        if let Some(cur) = fold.observe(fwd) {
            emitted.push(cur);
        }
    }
    assert_eq!(emitted, (1..=11).collect::<Vec<_>>());
    assert_eq!(TOTAL_STEPS, 11);
}

#[test]
fn descriptor_advertises_curated_samplers_no_scheduler() {
    // epic 7114 (sc-7122): LTX exposes the curated SAMPLER axis (the joint two-stream T2V+A denoise)
    // but NO scheduler — it keeps its baked distilled σ schedule. A non-empty scheduler list would be
    // a false capability.
    let caps = descriptor().capabilities;
    for s in [
        "euler",
        "euler_ancestral",
        "heun",
        "dpmpp_2m",
        "dpmpp_sde",
        "uni_pc",
        "ddim",
    ] {
        assert!(
            caps.samplers.contains(&s),
            "curated sampler {s:?} should be advertised"
        );
    }
    assert!(
        caps.schedulers.is_empty(),
        "LTX is sampler-only (no scheduler axis — the distilled schedule is baked)"
    );
}

#[test]
fn descriptor_declares_unconditional_staging_without_selectable_sequential_control() {
    let caps = descriptor().capabilities;
    assert!(!caps.supports_sequential_offload);
    assert_eq!(
        caps.staged_residency_availability(),
        mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged
    );
}

#[test]
fn resolve_gemma_dir_rejects_nonexistent_spec_override() {
    // sc-8827/sc-13664: the `LoadSpec::text_encoder` slot is the only TE source, and a nonexistent
    // override path errors with the spec-side message up front — the worker drives the TE location
    // through the spec (no `$LTX_GEMMA_DIR` / HF-cache scan any more).
    let src = WeightsSource::Dir("/nonexistent/ltx_gemma".into());
    let err = resolve_gemma_dir(Some(&src)).unwrap_err().to_string();
    assert!(err.contains("LoadSpec text_encoder"), "got: {err}");
    assert!(err.contains("does not exist"), "got: {err}");
}

/// sc-13664: `LoadSpec::text_encoder` is now **required** — an absent slot is a load-time,
/// actionable error naming the slot (not a silent `$LTX_GEMMA_DIR` / HF-cache fallback).
#[test]
fn resolve_gemma_dir_requires_text_encoder_slot() {
    let err = resolve_gemma_dir(None).unwrap_err().to_string();
    assert!(err.contains("LoadSpec::text_encoder"), "got: {err}");
    assert!(!err.contains("LTX_GEMMA_DIR"), "got: {err}");
}

/// sc-13664: LTX's only recognized component is `uncensored_enhancer`; an unknown component key is
/// rejected at load (typed `Unsupported`), and the enhancer path is validated up front when staged.
#[test]
fn load_rejects_unknown_component_and_validates_enhancer() {
    let bogus = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
        "bogus_component",
        WeightsSource::File("/x.safetensors".into()),
    );
    assert!(matches!(
        load(&bogus).err().expect("err"),
        Error::Unsupported(_)
    ));
    // A staged-but-nonexistent uncensored_enhancer errors up front, naming the component.
    let missing = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
        "uncensored_enhancer",
        WeightsSource::Dir("/nope/amoral".into()),
    );
    let err = load(&missing).err().expect("err").to_string();
    assert!(err.contains("uncensored_enhancer"), "got: {err}");
    assert!(err.contains("does not exist"), "got: {err}");
}

#[test]
fn latent_dims_matches_reference_formula() {
    // 256×256, 9 frames: latent_frames = 1+(9-1)/8 = 2; stage1 = H/2/32 = 4; stage2 = H/32 = 8.
    let req = GenerationRequest {
        width: 256,
        height: 256,
        frames: Some(9),
        ..Default::default()
    };
    assert_eq!(Ltx::latent_dims(&req), (2, 4, 4, 8, 8));
    // 512×768, 1 frame: latent_frames = 1; stage1 = 8×12; stage2 = 16×24.
    let req = GenerationRequest {
        width: 768,
        height: 512,
        frames: Some(1),
        ..Default::default()
    };
    assert_eq!(Ltx::latent_dims(&req), (1, 8, 12, 16, 24));
}

/// sc-18789: the DFR knobs are refused on the 2.3 checkpoint with a TYPED `Unsupported` — its
/// transformer has no learned keyframe-slot marker, so slots would denoise as unmarked tokens
/// (wasted compute, no conditioning). `0`/`None` stay accepted (off).
#[test]
fn validate_refuses_dfr_knobs_on_2_3_typed() {
    let caps = descriptor().capabilities;
    let base = GenerationRequest {
        prompt: "p".into(),
        width: 640,
        height: 384,
        frames: Some(25),
        ..Default::default()
    };
    for (label, req) in [
        (
            "num_generated_keyframes",
            GenerationRequest {
                num_generated_keyframes: Some(3),
                ..base.clone()
            },
        ),
        (
            "temporal_upsample_rounds",
            GenerationRequest {
                temporal_upsample_rounds: Some(1),
                ..base.clone()
            },
        ),
    ] {
        let err = validate_request(&caps, &req).expect_err(label);
        assert!(matches!(err, Error::Unsupported(_)), "{label}: {err}");
        assert!(err.to_string().contains("2.5"), "{label}: {err}");
    }
    // Zero / unset are the off position, not a refusal.
    let off = GenerationRequest {
        num_generated_keyframes: Some(0),
        temporal_upsample_rounds: Some(0),
        ..base
    };
    validate_request(&caps, &off).expect("0 is off");
}

#[test]
fn request_position_grids_share_the_candle_time_axis_at_public_fps() {
    // Candle's LTX grids use the request `fps` for both video stages and derive audio frames
    // as Python `round(frames / fps * 25)`. Exercise the MLX request seam at the non-default
    // public rates so a fixed 24-fps secondary grid cannot pass unnoticed.
    for fps in [25u32, 30] {
        let req = GenerationRequest {
            width: 256,
            height: 256,
            frames: Some(9),
            fps: Some(fps),
            ..Default::default()
        };
        let (lf, h1, w1, h2, w2) = Ltx::latent_dims(&req);
        let fps = Ltx::fps(&req);
        let stage1 = crate::positions::create_position_grid_data_with(
            1,
            lf,
            h1,
            w1,
            TEMPORAL_SCALE as i64,
            SPATIAL_SCALE as i64,
            fps,
            true,
        );
        let stage2 = crate::positions::create_position_grid_data_with(
            1,
            lf,
            h2,
            w2,
            TEMPORAL_SCALE as i64,
            SPATIAL_SCALE as i64,
            fps,
            true,
        );
        let time_end = |grid: &[f32], tokens: usize| grid[2 * tokens - 1];
        let expected_video_end = 9.0 / fps;
        assert!(
            (time_end(&stage1, lf * h1 * w1) - expected_video_end).abs() < 1e-7,
            "stage-1 {fps}-fps grid used the wrong clock"
        );
        assert!(
            (time_end(&stage2, lf * h2 * w2) - expected_video_end).abs() < 1e-7,
            "stage-2 {fps}-fps grid used the wrong clock"
        );

        let appended = crate::conditioning::keyframe_append_position_data(
            1,
            1,
            1,
            0,
            TEMPORAL_SCALE as i64,
            SPATIAL_SCALE as i64,
            fps,
        );
        assert!(
            (appended[1] - 1.0 / fps).abs() < 1e-7,
            "appended clip used the wrong {fps}-fps time axis"
        );

        let audio_frames = Ltx::audio_frames(&req);
        let expected_audio_frames = compute_audio_frames(9, fps as f64);
        assert_eq!(
            audio_frames, expected_audio_frames,
            "audio {fps}-fps frame count"
        );
        let video_seconds = 9.0 / fps as f64;
        let audio_seconds = audio_frames as f64 / crate::positions::AUDIO_LATENTS_PER_SECOND;
        assert!(
            (audio_seconds - video_seconds).abs()
                <= 0.5 / crate::positions::AUDIO_LATENTS_PER_SECOND + 1e-12,
            "audio/video duration disagreement at {fps} fps: {audio_seconds} vs {video_seconds}"
        );
    }
}

#[test]
fn default_request_position_grids_remain_the_24_fps_fixture() {
    let req = GenerationRequest {
        width: 256,
        height: 256,
        frames: Some(9),
        ..Default::default()
    };
    assert_eq!(Ltx::fps(&req), DEFAULT_FPS);
    let fixture = crate::positions::create_position_grid_data_with(
        1,
        1,
        1,
        1,
        TEMPORAL_SCALE as i64,
        SPATIAL_SCALE as i64,
        Ltx::fps(&req),
        true,
    );
    assert_eq!(fixture, vec![0.0, 1.0 / 24.0, 0.0, 32.0, 0.0, 32.0]);
}

#[test]
fn validate_request_enforces_constraints() {
    let caps = descriptor().capabilities;
    let base = GenerationRequest {
        prompt: "a".into(),
        width: 512,
        height: 512,
        frames: Some(33),
        ..Default::default()
    };
    assert!(validate_request(&caps, &base).is_ok());
    let zero_fps = validate_request(
        &caps,
        &GenerationRequest {
            fps: Some(0),
            ..base.clone()
        },
    )
    .expect_err("plain MLX LTX validation must reject zero fps before position/audio math");
    assert!(matches!(zero_fps, Error::Msg(_)), "got: {zero_fps:?}");
    assert!(zero_fps
        .to_string()
        .contains("fps must be greater than zero"));
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            prompt: String::new(),
            ..base.clone()
        }
    )
    .is_err());
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            width: 500,
            ..base.clone()
        }
    )
    .is_err());
    // The pinned stride is `SIZE_MULTIPLE` (= 64 = 2×SPATIAL_SCALE), NOT the bare 32× VAE scale:
    // a size that is a multiple of SPATIAL_SCALE but not SIZE_MULTIPLE must still be rejected, and
    // the error must name the stride (sc-12587 — this is the value SceneWorks ties to).
    assert_eq!(SIZE_MULTIPLE, 64);
    let off_stride = validate_request(
        &caps,
        &GenerationRequest {
            width: 480, // 15×32 — a multiple of SPATIAL_SCALE but not SIZE_MULTIPLE
            ..base.clone()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(
        off_stride.contains("divisible by 64"),
        "expected the stride error, got: {off_stride}"
    );
    // A different in-range multiple of SIZE_MULTIPLE is accepted.
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            width: 448, // 7×64
            ..base.clone()
        }
    )
    .is_ok());
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(32),
            ..base.clone()
        }
    )
    .is_err());
    // sc-4053: an unbounded frame count that still satisfies `% 8 == 1` is rejected by the max.
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(8_000_001), // 8_000_001 % 8 == 1, but far beyond MAX_FRAMES
            ..base.clone()
        }
    )
    .is_err());
    // The maximum itself is allowed; the next valid form above it is not.
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(MAX_FRAMES),
            ..base.clone()
        }
    )
    .is_ok());
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            frames: Some(MAX_FRAMES + 8),
            ..base.clone()
        }
    )
    .is_err());
}

/// sc-19502 — this lane used to accept ANY `req.steps` and render the baked 8-step schedule
/// regardless, while candle refused the same request. Both now refuse it.
///
/// Goes through `validate_request`, the function the generator actually calls
/// (`impl_generator!`'s `validate` arm), rather than asserting on `descriptor().capabilities`
/// directly — a declaration that no request path consults is exactly the inert-key failure this
/// story exists to fix, and reading the field back would prove only that a struct literal
/// contains what it contains.
#[test]
fn validate_request_refuses_an_off_schedule_step_count() {
    let caps = descriptor().capabilities;
    let base = GenerationRequest {
        prompt: "a".into(),
        width: 512,
        height: 512,
        frames: Some(9),
        ..Default::default()
    };
    let at = |steps: Option<u32>| {
        validate_request(
            &caps,
            &GenerationRequest {
                steps,
                ..base.clone()
            },
        )
    };

    // The advertised count and "the model picks" are both admitted — the common path (the
    // catalog's `defaults.steps` is 8) must not regress into a rejection.
    assert!(at(Some(NATIVE_STEPS)).is_ok());
    assert!(at(None).is_ok());

    // 30 is the case the story names: previously accepted here and silently ignored, refused on
    // candle. 1 and 4 cover the under side, which a FLOOR-shaped key would have admitted.
    for steps in [1u32, 4, 7, 9, 30, 50] {
        let err = at(Some(steps))
            .expect_err("an off-schedule step count must be refused, not silently ignored")
            .to_string();
        assert!(
            err.contains(MODEL_ID) && err.contains(&format!("steps={steps}")),
            "the refusal must name the model and the request: {err}"
        );
        assert!(
            err.contains(&NATIVE_STEPS.to_string()),
            "the refusal must name the legal value: {err}"
        );
    }
}

/// sc-19502 — the advertised step surface is DERIVED from the σ table this engine actually runs,
/// so re-baking the schedule cannot leave a stale advertised count behind.
///
/// The cross-lane half (that candle advertises the same 8) cannot be asserted here: no crate
/// depends on both `mlx-gen-ltx` and `candle-gen-ltx`, and mlx is macOS-only. SceneWorks owns
/// that guard, where one catalog entry demonstrably drives both backends.
#[test]
fn advertised_steps_are_derived_from_the_baked_schedule() {
    assert_eq!(
        NATIVE_STEPS as usize,
        crate::pipeline::STAGE1_SIGMAS.len() - 1
    );
    assert_eq!(NATIVE_STEPS, 8, "the distilled stage-1 schedule is 8 steps");
    assert_eq!(
        descriptor().capabilities.supported_steps,
        StepSupport::Exact(vec![NATIVE_STEPS]),
        "the descriptor must advertise exactly the baked schedule"
    );
}

#[test]
fn validate_request_conditioning() {
    let caps = descriptor().capabilities;
    let base = GenerationRequest {
        prompt: "a".into(),
        width: 512,
        height: 512,
        frames: Some(9),
        ..Default::default()
    };
    let img = Image {
        width: 4,
        height: 4,
        pixels: vec![0u8; 4 * 4 * 3],
    };
    // A single I2V `Reference` is accepted.
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img.clone(),
                strength: Some(0.8),
            }],
            ..base.clone()
        }
    )
    .is_ok());
    // Unsupported conditioning (e.g. Depth) is rejected.
    assert!(validate_request(
        &caps,
        &GenerationRequest {
            conditioning: vec![Conditioning::Depth { image: img.clone() }],
            ..base.clone()
        }
    )
    .is_err());
    // More than one `Reference` is rejected during weight-free preflight (single-image I2V only).
    let two = GenerationRequest {
        conditioning: vec![
            Conditioning::Reference {
                image: img.clone(),
                strength: None,
            },
            Conditioning::Reference {
                image: img.clone(),
                strength: None,
            },
        ],
        ..base.clone()
    };
    let err = validate_request(&caps, &two).unwrap_err().to_string();
    assert!(err.contains("multiple reference images"), "{err}");

    let control = Conditioning::ControlClip {
        frames: vec![img.clone()],
        mask: vec![img.clone()],
        masking_strength: 0.8,
        start_frame: 0,
        mode: mlx_gen::ReplacementMode::FaceOnly,
    };
    // The full 1–4 ordered replace-person surface is admitted before weights load.
    for reference_count in 1..=4 {
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                conditioning: vec![
                    control.clone(),
                    Conditioning::MultiReference {
                        images: vec![img.clone(); reference_count],
                    },
                ],
                ..base.clone()
            }
        )
        .is_ok());
    }
    for conditioning in [
        vec![control.clone()],
        vec![Conditioning::MultiReference {
            images: vec![img.clone()],
        }],
        vec![
            control.clone(),
            Conditioning::Reference {
                image: img.clone(),
                strength: None,
            },
            Conditioning::MultiReference {
                images: vec![img.clone()],
            },
        ],
        vec![
            control.clone(),
            Conditioning::MultiReference {
                images: vec![img.clone(); 5],
            },
        ],
    ] {
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                conditioning,
                ..base.clone()
            }
        )
        .is_err());
    }

    // The render consumes `control_clip()` (the first match), so duplicates must fail rather
    // than silently discard the second after text encoding.
    let err = validate_request(
        &caps,
        &GenerationRequest {
            conditioning: vec![control.clone(), control],
            ..base
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("at most one ControlClip"), "{err}");
}

#[test]
fn validate_request_rejects_late_conditioning_shape_errors_preflight() {
    let caps = descriptor().capabilities;
    let base = GenerationRequest {
        prompt: "a".into(),
        width: 512,
        height: 512,
        frames: Some(49), // seven latent frames: accepted indices -7..=6.
        ..Default::default()
    };
    let img = Image {
        width: 4,
        height: 4,
        pixels: vec![0u8; 4 * 4 * 3],
    };
    let rejects = |conditioning| {
        validate_request(
            &caps,
            &GenerationRequest {
                conditioning,
                ..base.clone()
            },
        )
        .unwrap_err()
        .to_string()
    };

    let err = rejects(vec![Conditioning::Keyframe {
        image: img.clone(),
        frame_idx: 7,
        strength: 1.0,
    }]);
    assert!(err.contains("keyframe latent frame index 7"), "{err}");

    let err = rejects(vec![Conditioning::VideoClip {
        frames: vec![],
        frame_idx: 0,
        strength: 1.0,
    }]);
    assert!(err.contains("video conditioning clip is empty"), "{err}");
    let err = rejects(vec![Conditioning::VideoClip {
        frames: vec![img.clone()],
        frame_idx: -8,
        strength: 1.0,
    }]);
    assert!(err.contains("clip latent frame index -8"), "{err}");

    let err = rejects(vec![
        Conditioning::ControlClip {
            frames: vec![],
            mask: vec![],
            masking_strength: 1.0,
            start_frame: 0,
            mode: mlx_gen::ReplacementMode::FaceOnly,
        },
        Conditioning::MultiReference {
            images: vec![img.clone()],
        },
    ]);
    assert!(err.contains("control clip is empty"), "{err}");
    let err = rejects(vec![
        Conditioning::ControlClip {
            frames: vec![img.clone()],
            mask: vec![],
            masking_strength: 1.0,
            start_frame: 0,
            mode: mlx_gen::ReplacementMode::FaceOnly,
        },
        Conditioning::MultiReference {
            images: vec![img.clone()],
        },
    ]);
    assert!(err.contains("frame count 1 != mask count 0"), "{err}");
    let err = rejects(vec![
        Conditioning::ControlClip {
            frames: vec![img.clone()],
            mask: vec![img.clone()],
            masking_strength: 1.0,
            start_frame: 7,
            mode: mlx_gen::ReplacementMode::FaceOnly,
        },
        Conditioning::MultiReference { images: vec![img] },
    ]);
    assert!(err.contains("replace_person latent frame index 7"), "{err}");
}
