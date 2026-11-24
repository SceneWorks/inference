use super::*;
use crate::{
    AdapterKind, AdapterSpec, Capabilities, Conditioning, ControlKind, GenerationMemory,
    GenerationPhase, Image, MemoryGeometry, MemoryRunOutcome, Modality, PhaseAdapter,
    WeightsSource,
};

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "base_image_fixture",
        backend: "mock",
        family: "test",
        modality: Modality::Image,
        capabilities: Capabilities::default(),
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
    }
}

fn plan(request: &GenerationRequest) -> Result<RenderPlan> {
    RenderPlan::base_image(&descriptor(), request, pass(vec![1.0, 0.5, 0.0], 2, 0.0))
}

fn pass(sigmas: Vec<f32>, executed_steps: usize, capture_sigma: f32) -> PackedImagePass<'static> {
    PackedImagePass {
        pixel_stride: 16,
        token_width: 64,
        sampler: "euler",
        guidance: 3.5,
        sigmas,
        executed_steps,
        capture_sigma,
    }
}

fn adapter_phase_request() -> GenerationRequest {
    GenerationRequest {
        count: 2,
        seed: Some(11),
        phases: Some(vec![
            GenerationPhase {
                steps: 1,
                guidance: Some(0.0),
                adapters: vec![],
            },
            GenerationPhase {
                steps: 2,
                guidance: None,
                adapters: vec![
                    PhaseAdapter {
                        adapter: 1,
                        weight: Some(-0.25),
                    },
                    PhaseAdapter {
                        adapter: 0,
                        weight: None,
                    },
                    PhaseAdapter {
                        adapter: 1,
                        weight: Some(1.25),
                    },
                ],
            },
            GenerationPhase {
                steps: 1,
                guidance: Some(0.0),
                adapters: vec![],
            },
        ]),
        ..Default::default()
    }
}

fn adapter_phase_descriptor() -> ModelDescriptor {
    let mut descriptor = descriptor();
    descriptor.capabilities.supports_phases = true;
    descriptor
}

fn adapter_phase_spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir("/base".into())).with_adapters(vec![
        AdapterSpec::new("/a".into(), 0.6, AdapterKind::Lora),
        AdapterSpec::new("/b".into(), 0.8, AdapterKind::Lokr),
    ])
}

#[test]
fn krea_plan_lowers_ordered_phase_arguments_and_binds_the_request_and_load() {
    let descriptor = adapter_phase_descriptor();
    let spec = adapter_phase_spec();
    let req = adapter_phase_request();
    let sigmas = vec![1.0, 0.75, 0.5, 0.25, 0.0];
    let plan = RenderPlan::krea_multiphase(&descriptor, &spec, &req, pass(sigmas, 4, 0.0)).unwrap();
    assert_eq!(plan.denoise_spans(), [0..1, 1..3, 3..4]);
    let [base, adapted, final_base] = plan.adapter_phases() else {
        panic!("expected base → adapted → base");
    };
    assert_eq!(base.arguments(), final_base.arguments());
    assert!(base.arguments().adapters().is_empty());
    assert!(!base.arguments().uses_classifier_free_guidance());
    assert_eq!(adapted.arguments().guidance(), 3.5);
    assert!(adapted.arguments().uses_classifier_free_guidance());
    assert_eq!(
        adapted.arguments().adapters(),
        [
            PhaseAdapterScale::new(1, -0.25),
            PhaseAdapterScale::new(0, 0.6),
            PhaseAdapterScale::new(1, 1.25),
        ]
    );
    assert!(plan.adapter_phases_need_negative());
    let json = serde_json::to_value(&plan).unwrap();
    assert_eq!(
        json["pass"]["adapter_phases"]["installation"],
        "immutable_residuals"
    );
    assert_eq!(
        json["pass"]["adapter_phases"]["sampling"],
        "reset_solver_and_reuse_image_seed"
    );
    assert!(plan.check_request(&descriptor, &req).is_ok());
    assert!(plan.check_load_spec(&spec).is_ok());
    assert!(plan.clone().with_inert_split_at(2).is_err());
    assert!(RenderPlan::packed_image_trajectory(
        &descriptor,
        &req,
        pass(plan.sigmas().to_vec(), 4, 0.0)
    )
    .is_err());

    let mut swapped = spec.clone();
    swapped.adapters.swap(0, 1);
    assert!(plan.check_load_spec(&swapped).is_err());
    let mut changed = req.clone();
    changed.phases.as_mut().unwrap()[1].adapters.swap(0, 1);
    assert!(plan.check_request(&descriptor, &changed).is_err());
    changed = req.clone();
    changed.pag = Some(crate::PagConfig::default());
    assert!(plan.check_request(&descriptor, &changed).is_err());
    changed = req;
    changed.text_style_gain = Some(0.75);
    assert!(plan.check_request(&descriptor, &changed).is_err());
}

#[test]
fn krea_plan_rebinds_placement_without_replacing_compiled_load_identity() {
    use crate::{LoadShape, LoadShapeDeclarationResult, OffloadPolicy, Precision};

    let descriptor = adapter_phase_descriptor();
    let spec = adapter_phase_spec();
    let req = adapter_phase_request();
    let plan = RenderPlan::krea_multiphase(
        &descriptor,
        &spec,
        &req,
        pass(vec![1.0, 0.75, 0.5, 0.25, 0.0], 4, 0.0),
    )
    .unwrap();
    let mut placed = spec.clone().with_offload_policy(OffloadPolicy::Sequential);
    placed.load_shape = LoadShape::DeferredMaterialization;
    placed.load_shape_declaration_result = LoadShapeDeclarationResult::Eligible;
    assert!(plan.check_load_spec(&placed).is_err());
    let rebound = plan.clone().rebind_load_placement(&placed).unwrap();
    assert_eq!(rebound.adapter_phases(), plan.adapter_phases());
    assert!(rebound.check_load_spec(&placed).is_ok());
    assert!(rebound.check_load_spec(&spec).is_err());

    let mut rescaled = placed.clone();
    rescaled.adapters[0].scale = 0.75;
    let mut replaced_adapter = placed.clone();
    replaced_adapter.adapters[0].path = "/different-adapter".into();
    let mut replaced_weights = placed.clone();
    replaced_weights.weights = WeightsSource::Dir("/different-weights".into());
    let mut retyped = placed;
    retyped.precision = Precision::Fp32;
    for other in [rescaled, replaced_adapter, replaced_weights, retyped] {
        assert!(matches!(
            plan.clone().rebind_load_placement(&other),
            Err(Error::Unsupported(message)) if message.contains("different model or adapter stack")
        ));
        let bypass = plan.clone().bind_load(&other);
        assert!(bypass.check_load_spec(&other).is_err());
        assert!(bypass
            .check_load(&RenderLoadBinding::from_spec(&other))
            .is_err());
        assert!(bypass.check_load_spec(&spec).is_ok());
    }
}

#[test]
fn krea_phase_plan_rejects_a_new_pin_for_the_same_adapter_path() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("adapter.safetensors");
    std::fs::write(&path, [1_u8]).unwrap();
    let mut ordinary = adapter_phase_spec();
    ordinary.adapters[0].path = path.clone();
    ordinary.adapters[1].path = path.clone();
    let mut sealed = ordinary.clone();
    sealed.prepare_file_sources().unwrap();
    let plan = RenderPlan::krea_multiphase(
        &adapter_phase_descriptor(),
        &sealed,
        &adapter_phase_request(),
        pass(vec![1.0, 0.75, 0.5, 0.25, 0.0], 4, 0.0),
    )
    .unwrap();
    assert!(plan.check_load_spec(&sealed.clone()).is_ok());
    assert!(plan.clone().rebind_load_placement(&ordinary).is_err());
    std::fs::write(path, [2_u8, 3]).unwrap();
    ordinary.prepare_file_sources().unwrap();
    assert!(plan.clone().rebind_load_placement(&ordinary).is_err());
    assert!(plan
        .bind_load(&ordinary)
        .check_load_spec(&ordinary)
        .is_err());
}

#[test]
fn krea_plan_preserves_authored_stops_across_short_and_long_native_schedules() {
    let descriptor = adapter_phase_descriptor();
    let spec = adapter_phase_spec();
    let mut req = adapter_phase_request();
    let short =
        RenderPlan::krea_multiphase(&descriptor, &spec, &req, pass(vec![1.0, 0.65, 0.0], 2, 0.0))
            .unwrap();
    assert_eq!(short.denoise_spans(), [0..1, 1..2]);
    assert_eq!(short.capture_sigma(), 0.0);
    req.phases.as_mut().unwrap()[2].adapters.push(PhaseAdapter {
        adapter: 2,
        weight: None,
    });
    assert!(
        RenderPlan::krea_multiphase(
            &descriptor,
            &spec,
            &req,
            pass(short.sigmas().to_vec(), 2, 0.0),
        )
        .is_err(),
        "an adapter in a clipped phase is still validated"
    );
    req.phases.as_mut().unwrap()[2].adapters.clear();
    let long = RenderPlan::krea_multiphase(
        &descriptor,
        &spec,
        &req,
        pass(vec![1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.0], 6, 0.0),
    )
    .unwrap();
    assert_eq!(long.denoise_spans(), [0..1, 1..3, 3..4]);
    assert_eq!(long.capture_sigma(), 0.4);
    assert_eq!(long.denoise_sigmas(), &[1.0, 0.85, 0.7, 0.55, 0.4]);
    assert_eq!(long.sigmas().len(), 7);
}

#[test]
fn krea_pag_receipt_counts_the_prefix_on_the_full_ddim_uniform_grid() {
    let req = GenerationRequest {
        scheduler: Some("ddim_uniform".into()),
        phases: Some(vec![
            GenerationPhase {
                steps: 16,
                ..Default::default()
            },
            GenerationPhase {
                steps: 15,
                ..Default::default()
            },
        ]),
        ..Default::default()
    };
    let sigmas =
        crate::sampling::flow_schedule_with_shift(req.scheduler.as_deref(), 0.0, None, 31, &[])
            .unwrap();
    let plan = RenderPlan::krea_multiphase(
        &adapter_phase_descriptor(),
        &adapter_phase_spec(),
        &req,
        pass(sigmas, 32, 0.0),
    )
    .unwrap();
    let executed_steps = plan.executed_steps();
    let schedule_steps = plan.sigmas().len() - 1;
    assert_eq!((executed_steps, schedule_steps), (31, 32));
    assert_eq!(plan.denoise_spans(), [0..16, 16..31]);

    let beyond_prefix = crate::PagResolved {
        scale: 2.0,
        adaptive_scale: 0.0,
        layers: vec![2],
        start: 0.96,
        stop: 1.0,
    };
    let active_edge = crate::PagResolved {
        start: 0.9375,
        stop: 0.95,
        ..beyond_prefix.clone()
    };
    assert!(beyond_prefix.step_scale(31, schedule_steps) > 0.0);
    assert_eq!(beyond_prefix.active_steps(executed_steps as u32), 1);
    assert!(active_edge.step_scale(30, schedule_steps) > 0.0);
    assert_eq!(active_edge.active_steps(executed_steps as u32), 0);

    for (pag, expected_steps) in [(beyond_prefix, 0), (active_edge, 1)] {
        let evaluated_steps = plan
            .adapter_phases()
            .iter()
            .flat_map(AdapterDenoisePhase::span)
            .filter(|&step| pag.step_scale(step, schedule_steps) > 0.0)
            .count() as u32;
        let receipt = pag.execution_report_for_prefix(executed_steps as u32, schedule_steps as u32);
        assert_eq!(receipt.total_steps, 31);
        assert_eq!(receipt.active_steps, expected_steps);
        assert_eq!(receipt.active_steps, evaluated_steps);
    }
}

#[test]
fn packed_image_plan_freezes_reference_and_control_content_before_load() {
    let image = Image {
        width: 16,
        height: 16,
        pixels: vec![7; 16 * 16 * 3],
    };
    for conditioning in [
        Conditioning::Reference {
            image: image.clone(),
            strength: Some(0.5),
        },
        Conditioning::Control {
            image: image.clone(),
            kind: ControlKind::Pose,
            scale: Some(0.7),
        },
    ] {
        let mut request = GenerationRequest {
            conditioning: vec![conditioning],
            ..Default::default()
        };
        let compiled = RenderPlan::packed_image_trajectory(
            &descriptor(),
            &request,
            pass(vec![1.0, 0.5, 0.0], 2, 0.0),
        )
        .unwrap();
        assert!(compiled.check_request(&descriptor(), &request).is_ok());
        match &mut request.conditioning[0] {
            Conditioning::Reference { image, .. } | Conditioning::Control { image, .. } => {
                image.pixels[0] ^= 1;
            }
            _ => unreachable!(),
        }
        assert!(compiled.check_request(&descriptor(), &request).is_err());
        match &mut request.conditioning[0] {
            Conditioning::Reference { image, strength } => {
                image.pixels[0] ^= 1;
                *strength = Some(0.75);
            }
            Conditioning::Control { image, scale, .. } => {
                image.pixels[0] ^= 1;
                *scale = Some(0.25);
            }
            _ => unreachable!(),
        }
        assert!(compiled.check_request(&descriptor(), &request).is_err());
    }
}

#[test]
fn packed_image_plan_refuses_multiple_or_unsupported_conditioning() {
    let image = Image {
        width: 16,
        height: 16,
        pixels: vec![0; 16 * 16 * 3],
    };
    for conditioning in [
        vec![
            Conditioning::Reference {
                image: image.clone(),
                strength: None,
            };
            2
        ],
        vec![
            Conditioning::Reference {
                image: image.clone(),
                strength: None,
            },
            Conditioning::Control {
                image: image.clone(),
                kind: ControlKind::Pose,
                scale: Some(0.7),
            },
        ],
        vec![Conditioning::Depth { image }],
    ] {
        let request = GenerationRequest {
            conditioning,
            ..Default::default()
        };
        assert!(RenderPlan::packed_image_trajectory(
            &descriptor(),
            &request,
            pass(vec![1.0, 0.5, 0.0], 2, 0.0),
        )
        .is_err());
    }
    let malformed = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: 16,
                height: 16,
                pixels: vec![0; 3],
            },
            strength: None,
        }],
        ..Default::default()
    };
    assert!(RenderPlan::packed_image_trajectory(
        &descriptor(),
        &malformed,
        pass(vec![1.0, 0.5, 0.0], 2, 0.0),
    )
    .is_err());
}

#[derive(Default)]
struct Scope(Vec<(bool, MemoryPhase)>);

impl MemoryRequestScope for Scope {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
        panic!("caller owns configuration")
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
        panic!("caller owns finish")
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.0.push((true, phase));
        Ok(())
    }
    fn enter_render_phase(&mut self, phase: MemoryPhase, id: &RenderPhaseIdentity) -> Result<()> {
        assert_eq!((id.pass, id.phase), (0, 0));
        assert_eq!(
            id.geometry,
            MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0
            }
        );
        self.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.0.push((false, phase));
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
        unreachable!()
    }
    fn configure_attention(&mut self, _: u32) -> Result<()> {
        unreachable!()
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
        unreachable!()
    }
}

#[test]
fn split_conditioning_covers_text_handoff_and_source_preparation_before_denoise() {
    use std::{cell::RefCell, rc::Rc};

    struct TraceScope(Rc<RefCell<Vec<&'static str>>>);
    impl MemoryRequestScope for TraceScope {
        fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
            unreachable!()
        }
        fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
            unreachable!()
        }
        fn enter_phase(&mut self, _: MemoryPhase) -> Result<()> {
            unreachable!()
        }
        fn enter_render_phase(
            &mut self,
            phase: MemoryPhase,
            _: &RenderPhaseIdentity,
        ) -> Result<()> {
            self.0.borrow_mut().push(match phase {
                MemoryPhase::Conditioning => "conditioning-enter",
                MemoryPhase::Denoise => "denoise-enter",
                MemoryPhase::Decode => "decode-enter",
            });
            Ok(())
        }
        fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
            self.0.borrow_mut().push(match phase {
                MemoryPhase::Conditioning => "conditioning-leave",
                MemoryPhase::Denoise => "denoise-leave",
                MemoryPhase::Decode => "decode-leave",
            });
            Ok(())
        }
        fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
            unreachable!()
        }
        fn configure_attention(&mut self, _: u32) -> Result<()> {
            unreachable!()
        }
        fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
            unreachable!()
        }
    }

    let request = GenerationRequest::default();
    let plan = plan(&request).unwrap();
    let trace = Rc::new(RefCell::new(Vec::new()));
    let mut scope = TraceScope(Rc::clone(&trace));
    let run = RenderRun::new(&plan, &request.cancel, Some(&mut scope));
    run.begin_conditioning().unwrap();
    assert!(run.begin_conditioning().is_err());
    trace.borrow_mut().push("text-encode");
    trace.borrow_mut().push("text-drop");
    trace.borrow_mut().push("source-encode");
    run.end_conditioning().unwrap();
    assert!(run.end_conditioning().is_err());
    run.images(
        &mut |_| {},
        |_| Ok::<_, Error>(()),
        |_, _, _| Ok::<_, Error>(()),
        |_, _, _| Ok::<_, Error>(()),
    )
    .unwrap();
    assert_eq!(
        *trace.borrow(),
        [
            "conditioning-enter",
            "text-encode",
            "text-drop",
            "source-encode",
            "conditioning-leave",
            "denoise-enter",
            "denoise-leave",
            "decode-enter",
            "decode-leave",
        ]
    );
}

#[test]
fn sequential_images_wrap_seed_and_drive_real_scope_per_operation() {
    let req = GenerationRequest {
        seed: Some(u64::MAX),
        count: 2,
        ..Default::default()
    };
    let plan = plan(&req).unwrap();
    let mut scope = Scope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    let encoded = run.encode(|| Ok::<_, Error>(3)).unwrap();
    let out = run
        .images(
            &mut |_| {},
            |seed| Ok::<_, Error>((encoded, seed)),
            |seed, latent, _| {
                assert_eq!(latent.1, seed);
                Ok(latent)
            },
            |seed, latent, _| {
                assert_eq!(latent.0, 3);
                Ok(seed)
            },
        )
        .unwrap();
    assert_eq!(out, [u64::MAX, 0]);
    use MemoryPhase::*;
    assert_eq!(
        scope.0,
        [
            (true, Conditioning),
            (false, Conditioning),
            (true, Denoise),
            (false, Denoise),
            (true, Decode),
            (false, Decode),
            (true, Denoise),
            (false, Denoise),
            (true, Decode),
            (false, Decode),
        ]
    );
}

#[test]
fn cancel_during_denoise_prevents_decode_and_all_later_images() {
    let req = GenerationRequest {
        count: 2,
        ..Default::default()
    };
    let plan = plan(&req).unwrap();
    let mut scope = Scope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    run.encode(|| Ok::<_, Error>(())).unwrap();
    let result = run.images(
        &mut |_| {},
        |_| Ok::<_, Error>(()),
        |_, (), _| {
            req.cancel.cancel();
            Ok(())
        },
        |_, (), _| -> Result<()> { panic!("decode after cancellation") },
    );
    assert!(matches!(result, Err(Error::Canceled)));
    assert_eq!(
        scope.0,
        [
            (true, MemoryPhase::Conditioning),
            (false, MemoryPhase::Conditioning),
            (true, MemoryPhase::Denoise),
            (false, MemoryPhase::Denoise)
        ]
    );
}

#[test]
fn plan_freezes_schedule_decode_count_and_preparation_inputs() {
    let mut req = GenerationRequest {
        width: 640,
        height: 480,
        prompt: "original".into(),
        ..Default::default()
    };
    let spec = LoadSpec::new(WeightsSource::Dir("/fixture/model".into()));
    let plan = plan(&req).unwrap().bind_load(&spec);
    let snapshot = serde_json::to_value(&plan).unwrap();
    assert_eq!(
        snapshot["pass"]["sigmas"],
        serde_json::json!([1.0, 0.5, 0.0])
    );
    assert_eq!(snapshot["pass"]["phase"]["stream"]["tokens"], 1200);
    assert_eq!(
        snapshot["pass"]["stages"],
        serde_json::json!([
            "Schedule",
            "PromptPrep",
            "Encode",
            "Init",
            "Enter",
            "Denoise",
            "Exit",
            "PostLatent",
            "Decode",
            "Output"
        ])
    );
    assert_eq!(plan.denoise_sigmas(), plan.sigmas());
    assert!(snapshot["pass"].get("inert_splits").is_none());
    req.memory = Some(GenerationMemory::default());
    plan.check_request(&descriptor(), &req).unwrap();
    req.prompt += " changed";
    assert!(plan.check_request(&descriptor(), &req).is_err());
    plan.check_load(&RenderLoadBinding::from_spec(&spec))
        .unwrap();
    let other = LoadSpec::new(WeightsSource::Dir("/fixture/other".into()));
    assert!(plan
        .check_load(&RenderLoadBinding::from_spec(&other))
        .is_err());
    let mut changed_control = spec.clone();
    changed_control.control = Some(WeightsSource::File("/fixture/control".into()));
    assert!(plan
        .check_load(&RenderLoadBinding::from_spec(&changed_control))
        .is_err());
    let mut changed_policy = spec.clone();
    changed_policy.offload_policy = crate::OffloadPolicy::Sequential;
    assert!(plan
        .check_load(&RenderLoadBinding::from_spec(&changed_policy))
        .is_err());
    let mut changed_component = spec.clone();
    changed_component.components.insert(
        "flux_ip_image_encoder".into(),
        WeightsSource::Dir("/fixture/encoder".into()),
    );
    assert!(plan
        .check_load(&RenderLoadBinding::from_spec(&changed_component))
        .is_err());
    let mut changed_shape = spec.clone();
    changed_shape.load_shape = crate::LoadShape::DeferredMaterialization;
    assert!(plan
        .check_load(&RenderLoadBinding::from_spec(&changed_shape))
        .is_err());
}

#[test]
fn inert_phase_boundaries_cover_executed_steps_without_changing_request() {
    let req = GenerationRequest::default();
    let plan = RenderPlan::base_image(
        &descriptor(),
        &req,
        pass(vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.0], 5, 0.0),
    )
    .unwrap();
    let one_span = 0..5;
    assert_eq!(plan.denoise_spans(), std::slice::from_ref(&one_span));
    assert!(plan.clone().with_inert_split_at(0).is_err());
    assert!(plan.clone().with_inert_split_at(5).is_err());
    let split = plan
        .with_inert_split_at(4)
        .unwrap()
        .with_inert_split_at(2)
        .unwrap()
        .with_inert_split_at(2)
        .unwrap();
    assert_eq!(split.denoise_spans(), [0..2, 2..4, 4..5]);
    assert_eq!(split.denoise_sigmas(), &[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]);
    assert_eq!(split.guidance(), 3.5);
    split.check_request(&descriptor(), &req).unwrap();
    assert_eq!(
        serde_json::to_value(&split).unwrap()["pass"]["inert_splits"],
        serde_json::json!([2, 4])
    );
}

#[derive(Default)]
struct PhaseScope {
    events: Vec<(bool, MemoryPhase, usize)>,
    active: Option<usize>,
    terminal: Vec<MemoryRunOutcome>,
}

impl MemoryRequestScope for PhaseScope {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
        unreachable!()
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> Result<()> {
        self.active = None; // owner-side teardown after a failed phase
        self.terminal.push(outcome);
        Ok(())
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.events.push((true, phase, 0));
        Ok(())
    }
    fn enter_render_phase(&mut self, phase: MemoryPhase, id: &RenderPhaseIdentity) -> Result<()> {
        if phase == MemoryPhase::Denoise {
            assert!(self.active.replace(id.phase).is_none());
        }
        assert_eq!(
            (id.pass, id.geometry.width, id.geometry.height),
            (0, 1024, 1024)
        );
        self.events.push((true, phase, id.phase));
        Ok(())
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        let index = if phase == MemoryPhase::Denoise {
            self.active.take().expect("denoise phase was entered")
        } else {
            0
        };
        self.events.push((false, phase, index));
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
        unreachable!()
    }
    fn configure_attention(&mut self, _: u32) -> Result<()> {
        unreachable!()
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
        unreachable!()
    }
}

#[test]
fn krea_runner_uses_contiguous_slices_and_the_original_seed_in_each_scope() {
    let req = adapter_phase_request();
    let plan = RenderPlan::krea_multiphase(
        &adapter_phase_descriptor(),
        &adapter_phase_spec(),
        &req,
        pass(vec![1.0, 0.75, 0.5, 0.25, 0.0], 4, 0.0),
    )
    .unwrap();
    let mut scope = PhaseScope::default();
    let mut trace = Vec::new();
    let output = {
        let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
        run.encode(|| Ok::<_, Error>(())).unwrap();
        run.images_with_denoise_phases(
            &mut |_| {},
            |_| Ok::<_, Error>(1.0),
            |seed, initial, _| {
                run.denoise_adapter_phases(seed, initial, |phase, sigmas, phase_seed, latent| {
                    assert_eq!(latent, sigmas[0]);
                    run.with_active_scope(|_| Ok(()))?
                        .expect("body runs with the admitted scope");
                    trace.push((phase.index(), phase_seed, sigmas.to_vec()));
                    Ok(*sigmas.last().unwrap())
                })
            },
            |_, latent, _| Ok(latent),
        )
        .unwrap()
    };
    assert_eq!(output, [0.0, 0.0]);
    assert_eq!(
        trace,
        [
            (0, 11, vec![1.0, 0.75]),
            (1, 11, vec![0.75, 0.5, 0.25]),
            (2, 11, vec![0.25, 0.0]),
            (0, 12, vec![1.0, 0.75]),
            (1, 12, vec![0.75, 0.5, 0.25]),
            (2, 12, vec![0.25, 0.0]),
        ]
    );
    assert_eq!(
        scope
            .events
            .iter()
            .filter(|event| event.1 == MemoryPhase::Denoise)
            .map(|&(enter, _, phase)| (enter, phase))
            .collect::<Vec<_>>(),
        [
            (true, 0),
            (false, 0),
            (true, 1),
            (false, 1),
            (true, 2),
            (false, 2),
            (true, 0),
            (false, 0),
            (true, 1),
            (false, 1),
            (true, 2),
            (false, 2),
        ]
    );
}

#[test]
fn krea_runner_leaves_failed_phase_for_owner_teardown() {
    let req = adapter_phase_request();
    let plan = RenderPlan::krea_multiphase(
        &adapter_phase_descriptor(),
        &adapter_phase_spec(),
        &req,
        pass(vec![1.0, 0.75, 0.5, 0.25, 0.0], 4, 0.0),
    )
    .unwrap();
    let mut scope = PhaseScope::default();
    {
        let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
        run.encode(|| Ok::<_, Error>(())).unwrap();
        let result = run.images_with_denoise_phases(
            &mut |_| {},
            |_| Ok(()),
            |seed, (), _| {
                run.denoise_adapter_phases(seed, (), |phase, _, _, ()| match phase.index() {
                    0 => Ok(()),
                    1 => Err(Error::Canceled),
                    _ => panic!("a later phase ran after cancellation"),
                })
            },
            |_, (), _| -> Result<()> { panic!("decode started after cancellation") },
        );
        assert!(matches!(result, Err(Error::Canceled)));
    }
    assert_eq!(scope.active, Some(1));
    scope.finish(MemoryRunOutcome::Canceled).unwrap();
    assert_eq!(scope.terminal, [MemoryRunOutcome::Canceled]);
    assert!(scope.active.is_none());
}

#[test]
#[allow(clippy::drop_non_drop)] // End RenderRun's scope borrow before inspecting the mock.
fn two_inert_phases_enter_distinct_scopes_in_order_for_each_image() {
    let req = GenerationRequest {
        count: 2,
        seed: Some(11),
        ..Default::default()
    };
    let plan = plan(&req).unwrap().with_inert_split_at(1).unwrap();
    let mut scope = PhaseScope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    run.encode(|| Ok::<_, Error>(())).unwrap();
    let images = run
        .images_with_denoise_phases(
            &mut |_| {},
            Ok::<_, Error>,
            |_, seed, _| {
                run.begin_denoise_phase(0)?;
                run.end_denoise_phase(0)?;
                run.begin_denoise_phase(1)?;
                run.end_denoise_phase(1)?;
                Ok(seed)
            },
            |_, seed, _| Ok(seed),
        )
        .unwrap();
    assert_eq!(images, [11, 12]);
    drop(run);
    use MemoryPhase::*;
    assert_eq!(
        scope.events,
        [
            (true, Conditioning, 0),
            (false, Conditioning, 0),
            (true, Denoise, 0),
            (false, Denoise, 0),
            (true, Denoise, 1),
            (false, Denoise, 1),
            (true, Decode, 0),
            (false, Decode, 0),
            (true, Denoise, 0),
            (false, Denoise, 0),
            (true, Denoise, 1),
            (false, Denoise, 1),
            (true, Decode, 0),
            (false, Decode, 0),
        ]
    );
    assert!(scope.active.is_none());
}

#[test]
#[allow(clippy::drop_non_drop)] // End RenderRun's scope borrow before inspecting the mock.
fn one_span_phase_path_keeps_one_denoise_scope() {
    let req = GenerationRequest::default();
    let plan = plan(&req).unwrap();
    let mut scope = PhaseScope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    run.encode(|| Ok::<_, Error>(())).unwrap();
    run.images_with_denoise_phases(
        &mut |_| {},
        |_| Ok::<_, Error>(()),
        |_, (), _| {
            run.begin_denoise_phase(0)?;
            run.end_denoise_phase(0)?;
            Ok(())
        },
        |_, (), _| Ok(()),
    )
    .unwrap();
    drop(run);
    use MemoryPhase::*;
    assert_eq!(
        scope.events,
        [
            (true, Conditioning, 0),
            (false, Conditioning, 0),
            (true, Denoise, 0),
            (false, Denoise, 0),
            (true, Decode, 0),
            (false, Decode, 0),
        ]
    );
}

#[test]
#[allow(clippy::drop_non_drop)] // End RenderRun's scope borrow before owner teardown.
fn cancellation_during_phase_preserves_cancel_for_owner_teardown() {
    let req = GenerationRequest::default();
    let plan = plan(&req).unwrap().with_inert_split_at(1).unwrap();
    let mut scope = PhaseScope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    run.encode(|| Ok::<_, Error>(())).unwrap();
    let failure = run.images_with_denoise_phases(
        &mut |_| {},
        |_| Ok::<_, Error>(()),
        |_, (), _| {
            run.begin_denoise_phase(0)?;
            req.cancel.cancel();
            Err::<(), Error>(Error::Canceled)
        },
        |_, (), _| -> Result<()> { panic!("decode after cancellation") },
    );
    assert!(matches!(failure, Err(Error::Canceled)));
    drop(run);
    assert_eq!(scope.active, Some(0));
    scope.finish(MemoryRunOutcome::Canceled).unwrap();
    assert!(scope.active.is_none());
    assert_eq!(scope.terminal.len(), 1);
}

#[test]
#[allow(clippy::drop_non_drop)] // End RenderRun's scope borrow before owner teardown.
fn failed_phase_keeps_original_error_for_owner_teardown() {
    let req = GenerationRequest::default();
    let plan = plan(&req).unwrap().with_inert_split_at(1).unwrap();
    let mut scope = PhaseScope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    run.encode(|| Ok::<_, Error>(())).unwrap();
    let failure = run.images_with_denoise_phases(
        &mut |_| {},
        |_| Ok::<_, Error>(()),
        |_, (), _| {
            run.begin_denoise_phase(0)?;
            Err::<(), Error>(Error::Msg("original provider fault".into()))
        },
        |_, (), _| -> Result<()> { panic!("decode after failed phase") },
    );
    assert!(
        matches!(failure, Err(Error::Msg(ref message)) if message == "original provider fault")
    );
    drop(run);
    assert_eq!(scope.active, Some(0));
    assert_eq!(scope.events.last(), Some(&(true, MemoryPhase::Denoise, 0)));
    scope
        .finish(MemoryRunOutcome::Error {
            message: "original provider fault".into(),
        })
        .unwrap();
    assert!(scope.active.is_none());
    assert_eq!(scope.terminal.len(), 1);
}

#[test]
fn invalid_schedules_and_unaligned_geometry_refuse_before_any_stage() {
    let req = GenerationRequest::default();
    for (nodes, steps, capture) in [
        (vec![0.0], 0, 0.0),
        (vec![1.0, 0.0], 0, 0.0),
        (vec![1.0, 0.0], 2, 0.0),
        (vec![0.5, 0.6, 0.0], 2, 0.0),
        (vec![1.0, f32::NAN, 0.0], 2, 0.0),
        (vec![1.0, 0.25, 0.0], 1, 0.5),
    ] {
        assert!(RenderPlan::base_image(&descriptor(), &req, pass(nodes, steps, capture)).is_err());
    }
    assert!(plan(&GenerationRequest { width: 17, ..req }).is_err());

    // Reference / source-image img2img terminal zero-step endpoint (`sigmas == [0.0]`,
    // `executed_steps == 0`) is permitted on `packed_image_trajectory` for VAE round-trips.
    let ref_req = GenerationRequest {
        conditioning: vec![reference(16, 16, Some(1.0))],
        ..Default::default()
    };
    let terminal_plan =
        RenderPlan::packed_image_trajectory(&descriptor(), &ref_req, pass(vec![0.0], 0, 0.0))
            .expect("terminal zero-step reference endpoint plan");
    assert_eq!(terminal_plan.executed_steps(), 0);
    assert_eq!(terminal_plan.denoise_sigmas(), &[0.0]);
    assert_eq!(terminal_plan.denoise_spans(), vec![0..0]);
}

#[test]
fn owner_refuses_a_second_encode_or_out_of_order_trajectory() {
    let req = GenerationRequest::default();
    let plan = plan(&req).unwrap();
    let run = RenderRun::new(&plan, &req.cancel, None);
    let images = || {
        run.images(
            &mut |_| {},
            |_| Ok::<_, Error>(()),
            |_, (), _| Ok(()),
            |_, (), _| Ok(()),
        )
    };
    assert!(images().is_err());
    run.encode(|| Ok::<_, Error>(())).unwrap();
    assert!(run.encode(|| Ok::<_, Error>(())).is_err());
    assert_eq!(images().unwrap(), [()]);
    assert!(images().is_err());
}

#[test]
fn failed_init_does_not_enter_the_denoise_scope() {
    let req = GenerationRequest::default();
    let plan = plan(&req).unwrap();
    let mut scope = Scope::default();
    let run = RenderRun::new(&plan, &req.cancel, Some(&mut scope));
    run.encode(|| Ok::<_, Error>(())).unwrap();
    let failure = run
        .images(
            &mut |_| {},
            |_| Err::<(), _>(Error::Msg("init failed".into())),
            |_, (), _| -> Result<()> { panic!("denoise after failed init") },
            |_, (), _| -> Result<()> { panic!("decode after failed init") },
        )
        .unwrap_err();
    assert_eq!(failure.to_string(), "init failed");
    assert_eq!(
        scope.0,
        [
            (true, MemoryPhase::Conditioning),
            (false, MemoryPhase::Conditioning)
        ]
    );
}

fn trajectory(request: &GenerationRequest) -> Result<Option<RenderPlan>> {
    RenderPlan::packed_image_trajectory(&descriptor(), request, pass(vec![1.0, 0.5, 0.0], 2, 0.0))
        .map(|plan| Some(plan.bind_load(&crate::LoadSpec::new(WeightsSource::Dir("/m".into())))))
}

fn hires_requests() -> (GenerationRequest, GenerationRequest) {
    let base = GenerationRequest {
        prompt: "a fox".into(),
        width: 32,
        height: 32,
        seed: Some(5),
        ..Default::default()
    };
    let refine = GenerationRequest {
        width: 64,
        height: 48,
        steps: Some(3),
        ..base.clone()
    };
    (base, refine)
}

fn produced(width: u32, height: u32) -> ProducedReference {
    ProducedReference {
        from_pass: 0,
        width,
        height,
        strength: Some(0.3),
    }
}

fn reference(width: u32, height: u32, strength: Option<f32>) -> Conditioning {
    Conditioning::Reference {
        image: Image {
            width,
            height,
            pixels: vec![9; (width * height * 3) as usize],
        },
        strength,
    }
}

#[derive(Default)]
struct RenderPhaseEntries(Vec<(MemoryPhase, RenderPhaseIdentity)>);

impl MemoryRequestScope for RenderPhaseEntries {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
        Ok(())
    }
    fn enter_phase(&mut self, _: MemoryPhase) -> Result<()> {
        Ok(())
    }
    fn enter_render_phase(&mut self, phase: MemoryPhase, id: &RenderPhaseIdentity) -> Result<()> {
        self.0.push((phase, *id));
        Ok(())
    }
    fn leave_phase(&mut self, _: MemoryPhase) -> Result<()> {
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
        unreachable!()
    }
    fn configure_attention(&mut self, _: u32) -> Result<()> {
        unreachable!()
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
        unreachable!()
    }
}

#[test]
fn multi_pass_plan_compiles_before_load_and_binds_produced_reference_by_shape() {
    let (base, refine) = hires_requests();
    let plan = MultiPassRenderPlan::compile(
        &[
            PlannedPassRequest {
                request: &base,
                produced_reference: None,
            },
            PlannedPassRequest {
                request: &refine,
                produced_reference: Some(produced(64, 48)),
            },
        ],
        trajectory,
    )
    .unwrap()
    .expect("both passes have pre-load plans");
    let [first, second] = plan.passes() else {
        panic!("two passes");
    };
    assert_eq!((first.pass_index(), second.pass_index()), (0, 1));
    assert_eq!(
        (first.output_size(), second.output_size()),
        ((32, 32), (64, 48))
    );
    assert_eq!(second.produced_reference(), Some(produced(64, 48)));
    assert_eq!(first.produced_reference(), None);
    assert_eq!(plan.total_executed_steps(), 4);
    first.check_request(&descriptor(), &base).unwrap();
    plan.check_load(&RenderLoadBinding::from_spec(&crate::LoadSpec::new(
        WeightsSource::Dir("/m".into()),
    )))
    .unwrap();

    // Any pixels of the declared size and strength stand in for the produced image.
    let mut live = refine.clone();
    live.conditioning = vec![reference(64, 48, Some(0.3))];
    second.check_request(&descriptor(), &live).unwrap();
    live.conditioning = vec![reference(64, 48, Some(0.3))];
    if let Conditioning::Reference { image, .. } = &mut live.conditioning[0] {
        image.pixels.fill(200);
    }
    second.check_request(&descriptor(), &live).unwrap();
    for wrong in [
        vec![reference(32, 32, Some(0.3))],
        vec![reference(64, 48, Some(0.5))],
        vec![reference(64, 48, None)],
        Vec::new(),
    ] {
        live.conditioning = wrong;
        assert!(second.check_request(&descriptor(), &live).is_err());
    }
    // Other output-affecting inputs remain exact.
    live.conditioning = vec![reference(64, 48, Some(0.3))];
    live.steps = Some(4);
    assert!(second.check_request(&descriptor(), &live).is_err());
    live.steps = refine.steps;
    live.source_image = Some(Image {
        width: 16,
        height: 16,
        pixels: vec![0; 16 * 16 * 3],
    });
    assert!(second.check_request(&descriptor(), &live).is_err());
    live.source_image = None;
    if let Conditioning::Reference { image, .. } = &mut live.conditioning[0] {
        image.pixels.pop().unwrap();
    }
    assert!(second.check_request(&descriptor(), &live).is_err());

    // The phase identity of the second pass names its own pass and geometry.
    let mut scope = RenderPhaseEntries::default();
    let cancel = CancelFlag::new();
    {
        let run = RenderRun::new(second, &cancel, Some(&mut scope));
        run.encode(|| Ok::<_, Error>(())).unwrap();
    }
    let [(phase, identity)] = scope.0.as_slice() else {
        panic!("expected one phase");
    };
    assert_eq!(*phase, MemoryPhase::Conditioning);
    assert_eq!(identity.pass, 1);
    assert_eq!(
        (identity.geometry.width, identity.geometry.height),
        (64, 48)
    );
    assert_eq!(identity.geometry.reference_count, 1);
}

#[test]
fn exact_multi_pass_plan_retains_pins_and_terminal_reference_controls_across_placement() {
    use crate::{LoadShape, LoadShapeDeclarationResult, OffloadPolicy};

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dit.safetensors");
    std::fs::write(&path, [1]).unwrap();
    let mut spec = LoadSpec::new(WeightsSource::File(path.clone()));
    let unprepared = spec.clone();
    spec.prepare_file_sources().unwrap();
    let (base, refine) = hires_requests();
    let produced = ProducedReference {
        strength: Some(1.0),
        ..produced(refine.width, refine.height)
    };
    let passes = [
        PlannedPassRequest {
            request: &base,
            produced_reference: None,
        },
        PlannedPassRequest {
            request: &refine,
            produced_reference: Some(produced),
        },
    ];
    let compile = |spec: &LoadSpec, req: &GenerationRequest| {
        let schedule = if req.conditioning.is_empty() {
            pass(vec![1.0, 0.5, 0.0], 2, 0.0)
        } else {
            pass(vec![0.0], 0, 0.0)
        };
        RenderPlan::packed_image_trajectory_with_exact_load(&descriptor(), spec, req, schedule)
            .map(Some)
    };
    assert!(matches!(
        compile(&unprepared, &base),
        Err(Error::Unsupported(message)) if message.contains("finalized file source pins")
    ));
    assert!(trajectory(&base)
        .unwrap()
        .unwrap()
        .check_load_spec(&spec)
        .is_err());
    let plan = MultiPassRenderPlan::compile(&passes, |req| compile(&spec, req))
        .unwrap()
        .unwrap();
    assert_eq!(plan.total_executed_steps(), 2);
    plan.check_load_spec(&spec).unwrap();
    assert!(plan
        .check_load(&RenderLoadBinding::from_spec(&spec))
        .is_err());
    let mut placed = spec.clone().with_offload_policy(OffloadPolicy::Sequential);
    placed.load_shape = LoadShape::DeferredMaterialization;
    placed.load_shape_declaration_result = LoadShapeDeclarationResult::Eligible;
    assert!(plan.check_load_spec(&placed).is_err());
    let rebound = plan.clone().rebind_load_placement(&placed).unwrap();
    rebound.check_load_spec(&placed).unwrap();
    assert!(rebound.check_load_spec(&spec).is_err());

    let [first, second] = rebound.passes() else {
        panic!("expected two passes");
    };
    assert_eq!((first.reference_count(), second.reference_count()), (0, 1));
    assert_eq!(second.produced_reference(), Some(produced));
    assert_eq!(second.denoise_sigmas(), &[0.0]);
    assert_eq!(second.executed_steps(), 0);
    let mut scope = RenderPhaseEntries::default();
    let cancel = CancelFlag::new();
    {
        let run = RenderRun::new(second, &cancel, Some(&mut scope));
        run.encode(|| Ok::<_, Error>(())).unwrap();
        let images = run
            .images(
                &mut |_| panic!("terminal reference should not report denoise progress"),
                Ok::<_, Error>,
                |seed, initial, _| {
                    assert_eq!(initial, seed);
                    Ok(initial)
                },
                |seed, terminal, _| {
                    assert_eq!(terminal, seed);
                    Ok(seed)
                },
            )
            .unwrap();
        assert_eq!(images, [second.base_seed()]);
    }
    assert_eq!(
        scope
            .0
            .into_iter()
            .map(|(phase, id)| (phase, id.pass))
            .collect::<Vec<_>>(),
        [(MemoryPhase::Conditioning, 1), (MemoryPhase::Decode, 1)]
    );
    let live = GenerationRequest {
        conditioning: vec![reference(refine.width, refine.height, Some(1.0))],
        ..refine.clone()
    };
    second.check_request(&descriptor(), &live).unwrap();
    for changed in [
        GenerationRequest {
            strength: Some(0.8),
            ..live.clone()
        },
        GenerationRequest {
            text_style_gain: Some(0.8),
            ..live.clone()
        },
        GenerationRequest {
            pag: Some(crate::PagConfig::default()),
            ..live.clone()
        },
        GenerationRequest {
            output_channels: crate::OutputChannels::Rgba,
            ..live
        },
    ] {
        assert!(second.check_request(&descriptor(), &changed).is_err());
    }

    std::fs::write(path, [2, 3]).unwrap();
    let mut repinned = unprepared;
    repinned.prepare_file_sources().unwrap();
    assert!(plan.clone().rebind_load_placement(&repinned).is_err());
    assert!(first
        .clone()
        .bind_load(&repinned)
        .check_load_spec(&repinned)
        .is_err());
    assert!(
        MultiPassRenderPlan::compile(&passes, |req| if req.conditioning.is_empty() {
            compile(&spec, req)
        } else {
            compile(&repinned, req)
        })
        .is_err()
    );
}

#[test]
fn multi_pass_plan_refuses_invalid_pass_shapes_before_load() {
    let (base, refine) = hires_requests();
    let compile =
        |passes: &[PlannedPassRequest<'_>]| MultiPassRenderPlan::compile(passes, trajectory);
    let first = PlannedPassRequest {
        request: &base,
        produced_reference: None,
    };
    // A single pass is not a multi-pass plan.
    assert!(compile(&[first]).is_err());
    // Pass 2 geometry is compiled and refused before load (off the 16-pixel grid).
    let off_grid = GenerationRequest {
        width: 65,
        ..refine.clone()
    };
    assert!(compile(&[
        first,
        PlannedPassRequest {
            request: &off_grid,
            produced_reference: Some(produced(65, 48)),
        },
    ])
    .is_err());
    // A pass cannot consume itself or a later pass, nor carry its own conditioning too.
    assert!(compile(&[
        first,
        PlannedPassRequest {
            request: &refine,
            produced_reference: Some(ProducedReference {
                from_pass: 1,
                ..produced(64, 48)
            }),
        },
    ])
    .is_err());
    let conditioned = GenerationRequest {
        conditioning: vec![reference(64, 48, Some(0.3))],
        ..refine.clone()
    };
    assert!(compile(&[
        first,
        PlannedPassRequest {
            request: &conditioned,
            produced_reference: Some(produced(64, 48)),
        },
    ])
    .is_err());
    let with_source_image = GenerationRequest {
        source_image: Some(Image {
            width: 16,
            height: 16,
            pixels: vec![0; 16 * 16 * 3],
        }),
        ..refine.clone()
    };
    assert!(compile(&[
        first,
        PlannedPassRequest {
            request: &with_source_image,
            produced_reference: Some(produced(64, 48)),
        },
    ])
    .is_err());
    assert!(matches!(
        compile(&[
            first,
            PlannedPassRequest {
                request: &refine,
                produced_reference: Some(ProducedReference {
                    strength: Some(f32::NAN),
                    ..produced(64, 48)
                }),
            },
        ]),
        Err(Error::Unsupported(message)) if message.contains("produced reference strength")
    ));
    // An oversized placeholder and a prompt batch are refused.
    assert!(compile(&[
        first,
        PlannedPassRequest {
            request: &refine,
            produced_reference: Some(produced(8192, 8192)),
        },
    ])
    .is_err());
    let batch = GenerationRequest {
        count: 2,
        ..base.clone()
    };
    assert!(compile(&[
        PlannedPassRequest {
            request: &batch,
            produced_reference: None,
        },
        PlannedPassRequest {
            request: &refine,
            produced_reference: Some(produced(64, 48)),
        },
    ])
    .is_err());
    // A provider without a pre-load plan yields no plan.
    assert_eq!(
        MultiPassRenderPlan::compile(
            &[
                first,
                PlannedPassRequest {
                    request: &refine,
                    produced_reference: Some(produced(64, 48)),
                },
            ],
            |_| Ok(None),
        )
        .unwrap(),
        None
    );
}

#[test]
fn single_pass_plans_keep_pass_zero_and_digest_binding() {
    let request = GenerationRequest {
        conditioning: vec![reference(16, 16, Some(0.5))],
        ..Default::default()
    };
    let compiled = trajectory(&request).unwrap().unwrap();
    assert_eq!(compiled.pass_index(), 0);
    assert_eq!(compiled.produced_reference(), None);
    assert!(!serde_json::to_string(&compiled)
        .unwrap()
        .contains("pass_index"));
    let mut changed = request.clone();
    if let Conditioning::Reference { image, .. } = &mut changed.conditioning[0] {
        image.pixels[0] ^= 1;
    }
    assert!(compiled.check_request(&descriptor(), &changed).is_err());
}

#[test]
fn video_denoise_borrows_the_admitted_scope_once_and_preserves_failure_ownership() {
    use super::video::{VideoDenoisePhase, VideoModelRule};
    let phase = VideoDenoisePhase {
        identity: RenderPhaseIdentity {
            pass: 0,
            phase: 0,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                frames: 17,
                batch: 1,
                reference_count: 1,
            },
        },
        latent_frames: 5,
        solver: crate::sampling::Solver::Heun,
        sigmas: vec![0.99, 0.6, 0.0],
        seed: 11,
        models: VideoModelRule::Fixed(()),
    };
    let cancel = CancelFlag::new();
    let mut scope = PhaseScope::default();
    phase
        .run(&cancel, Some(&mut scope), || Ok::<_, Error>(()))
        .unwrap();
    assert_eq!(
        scope.events,
        [
            (true, MemoryPhase::Denoise, 0),
            (false, MemoryPhase::Denoise, 0)
        ]
    );
    assert!(phase
        .run(&cancel, Some(&mut scope), || Err::<(), _>(
            Error::ProcessPoisoned("device".into())
        ))
        .is_err());
    assert_eq!(scope.active, Some(0));
    cancel.cancel();
    let events = scope.events.clone();
    let result: Result<()> = phase.run(&cancel, Some(&mut scope), || {
        panic!("cancelled before phase entry")
    });
    assert!(matches!(result, Err(Error::Canceled)));
    assert_eq!(scope.events, events);
    assert!(scope.terminal.is_empty());
}

#[test]
fn video_phase_accepts_the_native_schedule_direction_and_selects_exact_sigma_thresholds() {
    use super::video::{VideoDenoisePhase, VideoModelCoordinate, VideoModelRule};
    let phase = VideoDenoisePhase {
        identity: RenderPhaseIdentity {
            pass: 0,
            phase: 0,
            geometry: MemoryGeometry {
                width: 128,
                height: 96,
                frames: 17,
                batch: 1,
                reference_count: 0,
            },
        },
        latent_frames: 5,
        solver: (),
        sigmas: vec![0.0, 0.4, 1.0],
        seed: 11,
        models: VideoModelRule::AtOrAbove {
            coordinate: VideoModelCoordinate::Sigma,
            threshold: 0.4,
            high: true,
            low: false,
        },
    };
    phase.validate().unwrap();
    assert!(!phase.models.slot(0.4f32.next_down()));
    assert!(phase.models.slot(0.4));
}
