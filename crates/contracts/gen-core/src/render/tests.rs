use super::*;
use crate::{
    Capabilities, GenerationMemory, MemoryGeometry, MemoryRunOutcome, Modality, WeightsSource,
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
    RenderPlan::base_image(
        &descriptor(),
        request,
        "euler",
        3.5,
        vec![1.0, 0.5, 0.0],
        2,
        0.0,
    )
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
        [(true, MemoryPhase::Denoise), (false, MemoryPhase::Denoise)]
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
    assert_eq!(snapshot["pass"]["stages"][7], "Exit");
    assert_eq!(plan.denoise_sigmas(), plan.sigmas());
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
        assert!(
            RenderPlan::base_image(&descriptor(), &req, "euler", 0.0, nodes, steps, capture)
                .is_err()
        );
    }
    assert!(plan(&GenerationRequest { width: 17, ..req }).is_err());
}
