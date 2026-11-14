use super::*;
use crate::{
    Capabilities, Conditioning, ControlKind, GenerationMemory, Image, MemoryGeometry,
    MemoryRunOutcome, Modality, WeightsSource,
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
        (vec![1.0, 0.0], 0, 0.0),
        (vec![1.0, 0.0], 2, 0.0),
        (vec![0.5, 0.6, 0.0], 2, 0.0),
        (vec![1.0, f32::NAN, 0.0], 2, 0.0),
        (vec![1.0, 0.25, 0.0], 1, 0.5),
    ] {
        assert!(RenderPlan::base_image(&descriptor(), &req, pass(nodes, steps, capture)).is_err());
    }
    assert!(plan(&GenerationRequest { width: 17, ..req }).is_err());
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

    // The phase identity of the second pass names its own pass and geometry.
    struct Identities(Vec<RenderPhaseIdentity>);
    impl MemoryRequestScope for Identities {
        fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
            Ok(())
        }
        fn enter_phase(&mut self, _: MemoryPhase) -> Result<()> {
            Ok(())
        }
        fn enter_render_phase(&mut self, _: MemoryPhase, id: &RenderPhaseIdentity) -> Result<()> {
            self.0.push(*id);
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
    let mut scope = Identities(Vec::new());
    let cancel = CancelFlag::new();
    {
        let run = RenderRun::new(second, &cancel, Some(&mut scope));
        run.encode(|| Ok::<_, Error>(())).unwrap();
    }
    assert_eq!(scope.0[0].pass, 1);
    assert_eq!(
        (scope.0[0].geometry.width, scope.0[0].geometry.height),
        (64, 48)
    );
    assert_eq!(scope.0[0].geometry.reference_count, 1);
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
