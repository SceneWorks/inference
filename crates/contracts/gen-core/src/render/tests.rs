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
