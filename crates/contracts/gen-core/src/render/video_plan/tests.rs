use super::*;
use crate::generator::CompletedGeneration;
use crate::{
    Capabilities, GenerationOutput, Generator, Image, Modality, ModelRegistration, OffloadPolicy,
    Progress, ProviderRegistryBuilder, WeightsSource,
};
use std::cell::Cell;

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "video_layout_test",
        backend: "mock",
        family: "test",
        modality: Modality::Video,
        capabilities: Capabilities::default(),
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
    }
}
fn request() -> GenerationRequest {
    GenerationRequest {
        width: 16,
        height: 8,
        frames: Some(5),
        fps: Some(8),
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: 2,
                height: 1,
                pixels: vec![43; 6],
            },
            strength: Some(1.0),
        }],
        ..Default::default()
    }
}
fn compile(_: &LoadSpec, req: &GenerationRequest) -> Result<Option<VideoPlanLayout>> {
    let geometry = MemoryGeometry {
        width: req.width,
        height: req.height,
        frames: 5,
        batch: 1,
        reference_count: 1,
    };
    Ok(Some(VideoPlanLayout {
        packed_components: vec![],
        output_conversion: VideoOutputConversion {
            device_bytes: 700,
            host_bytes: 500,
        },
        output: geometry,
        admission: geometry,
        domain: MemoryBudgetDomain::DeviceVram,
        references: vec![VideoReferenceLayout {
            role: "start",
            width: req.width,
            height: req.height,
            frames: 1,
        }],
        phases: vec![VideoPhaseLayout {
            identity: RenderPhaseIdentity {
                pass: 0,
                phase: 0,
                geometry,
            },
            latent_frames: 2,
            latent_bytes: 256,
            retained_bytes: 64,
            decoded_tensor_bytes: 350,
            decoded_frames: 5,
            decode_chunk_frames: 5,
            source: 0..5,
            history_frames: 0,
        }],
    }))
}
fn registry() -> crate::ProviderRegistry {
    ProviderRegistryBuilder::new()
        .register_generator(ModelRegistration {
            descriptor,
            load: |s| Ok(Box::new(Mock::new(s))),
            footprint: None,
        })
        .register_video_plan(VideoPlanRegistration {
            provider_id: descriptor().id,
            compile,
        })
        .build()
        .unwrap()
}
struct Mock {
    desc: ModelDescriptor,
    load: RenderLoadBinding,
    forward_calls: Cell<u32>,
    latent_frames: u32,
    output_case: usize,
    contract: Option<crate::MemoryProviderContract>,
}
impl Mock {
    fn new(spec: &LoadSpec) -> Self {
        Self {
            desc: descriptor(),
            load: RenderLoadBinding::from_video_spec(spec),
            forward_calls: Cell::new(0),
            latent_frames: 2,
            output_case: 0,
            contract: None,
        }
    }
}
impl Generator for Mock {
    fn memory_strategy_contract(&self) -> Option<&crate::MemoryProviderContract> {
        self.contract.as_ref()
    }
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }
    fn video_load_binding(&self) -> Option<&RenderLoadBinding> {
        Some(&self.load)
    }
    fn validate(&self, _: &GenerationRequest) -> Result<()> {
        Ok(())
    }
    fn generate(
        &self,
        _: &GenerationRequest,
        _: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.forward_calls.set(self.forward_calls.get() + 1);
        let mut frame = Image {
            width: 16,
            height: 8,
            pixels: vec![0; 16 * 8 * 3],
        };
        let mut fps = 8;
        let mut count = 5;
        match self.output_case {
            1 => return Ok(GenerationOutput::Images(vec![frame])),
            2 => frame.width = 8,
            3 => {
                frame.pixels.pop();
            }
            4 => count = 4,
            5 => fps = 12,
            _ => {}
        }
        Ok(GenerationOutput::Video {
            frames: vec![frame; count],
            fps,
            audio: None,
        })
    }
    fn generate_with_scoped_execution_report(
        &self,
        req: &GenerationRequest,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        let scope = scope.expect("normal planned runner supplies a phase scope");
        let identity = RenderPhaseIdentity {
            pass: 0,
            phase: 0,
            geometry: MemoryGeometry {
                width: req.width,
                height: req.height,
                frames: 5,
                batch: 1,
                reference_count: 1,
            },
        };
        scope.enter_video_phase(&identity, self.latent_frames)?;
        let report = self.generate_with_execution_report(req, on_progress)?;
        scope.leave_phase(MemoryPhase::Denoise)?;
        Ok(report)
    }
}

#[test]
fn cached_placement_and_real_pixels_are_bound_before_provider_work() {
    let req = request();
    let mut spec = LoadSpec::new(WeightsSource::Dir("cache-A".into()));
    let plan = registry()
        .compile_video_plan(descriptor().id, &spec, &req)
        .unwrap()
        .unwrap();
    let previous = Mock::new(&spec);
    spec.offload_policy = OffloadPolicy::Sequential;
    let mut actual = Mock::new(&spec);
    plan.check_provider(actual.descriptor(), actual.video_load_binding())
        .unwrap();
    let admitted = plan.bind_admission(&spec, None, &req).unwrap();
    let execute = |model: &Mock, req: &GenerationRequest| {
        model.generate_video_planned_with_execution_report(req, &admitted, None, &mut |_| {})
    };
    assert!(execute(&previous, &req).is_err());
    actual.desc.backend = "other";
    assert!(execute(&actual, &req).is_err());
    actual.desc = descriptor();
    let mut changed = request();
    let Conditioning::Reference { image, .. } = &mut changed.conditioning[0] else {
        unreachable!()
    };
    image.pixels[0] ^= 1;
    assert!(execute(&actual, &changed).is_err());
    changed = request();
    changed.low_res_start = Some(crate::LowResStartSpec::new(1, None));
    assert!(registry()
        .compile_video_plan(descriptor().id, &spec, &changed)
        .is_err());
    assert!(execute(&actual, &changed).is_err());
    changed.low_res_start = None;
    changed.conditioning_fps = Some(6);
    assert!(execute(&actual, &changed).is_err());
    actual.latent_frames = 1;
    assert!(execute(&actual, &req).is_err());
    assert_eq!(actual.forward_calls.get(), 0);
    actual.latent_frames = 2;
    assert!(execute(&actual, &req).is_ok());
    assert_eq!(actual.forward_calls.get(), 1);
    for case in 1..=5 {
        actual.output_case = case;
        assert!(execute(&actual, &req).is_err());
    }
    assert!(admitted.bind_admission(&spec, None, &req).is_err());
    spec.weights = WeightsSource::Dir("cache-B".into());
    assert!(plan.bind_admission(&spec, None, &req).is_err());
    spec.weights = WeightsSource::Dir("cache-A".into());
    changed = request();
    changed.memory = Some(GenerationMemory {
        stage_residency: true,
        ..Default::default()
    });
    assert!(plan.bind_admission(&spec, None, &changed).is_err());
}

#[test]
fn denoise_retained_tensor_and_terminal_rgb_use_distinct_budgets() {
    let req = request();
    let spec = LoadSpec::new(WeightsSource::Dir("cache".into()));
    let mut plan = registry()
        .compile_video_plan(descriptor().id, &spec, &req)
        .unwrap()
        .unwrap();
    assert_eq!(
        plan.minimum_denoise_live_bytes(MemoryBudgetDomain::DeviceVram)
            .unwrap(),
        320
    );
    assert_eq!(
        plan.minimum_live_bytes(MemoryBudgetDomain::HostRam)
            .unwrap(),
        6 + 500 + 16 * 8 * 3 * 5
    );
    assert_eq!(
        plan.minimum_live_bytes(MemoryBudgetDomain::UnifiedMemory)
            .unwrap(),
        0
    );
    assert_eq!(
        plan.minimum_decode_live_bytes(MemoryBudgetDomain::DeviceVram)
            .unwrap(),
        700
    );
    plan.layout.domain = MemoryBudgetDomain::UnifiedMemory;
    assert_eq!(
        plan.minimum_denoise_live_bytes(MemoryBudgetDomain::UnifiedMemory)
            .unwrap(),
        326
    );
    assert_eq!(
        plan.minimum_live_bytes(MemoryBudgetDomain::HostRam)
            .unwrap(),
        0
    );
    let output_bytes = 16 * 8 * 3 * 5;
    let budget = output_bytes + 700;
    assert!(output_bytes + 6 < budget && 700 + 500 < budget);
    assert_eq!(
        plan.minimum_decode_live_bytes(MemoryBudgetDomain::UnifiedMemory)
            .unwrap(),
        output_bytes + 6 + 700 + 500
    );
    assert!(
        plan.minimum_live_bytes(MemoryBudgetDomain::UnifiedMemory)
            .unwrap()
            > budget
    );
}

#[test]
fn resident_defaults_preserve_optional_scope_and_optimized_requires_one() {
    use crate::{
        LoadShape, MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategy, Precision,
    };
    let req = request();
    let spec = LoadSpec::new(WeightsSource::Dir("fixtures".into()));
    let plan = registry()
        .compile_video_plan(descriptor().id, &spec, &req)
        .unwrap()
        .unwrap();
    let mut context = MemoryRunContext {
        selection: MemorySelection {
            strategy: MemoryStrategy::Resident,
            parameters: Default::default(),
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        },
        optimization_authority: Default::default(),
        calibration_abi: crate::MEMORY_CALIBRATION_ABI,
        calibration_fingerprint: String::new(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::Other("image_to_video".into()),
        has_reference: true,
        use_pid: false,
        has_phases: false,
        geometry: plan.geometry(),
        reference_resources: None,
        overlay: None,
        budget: MemoryBudget {
            total_bytes: 10000,
            committed_bytes: 100,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 2000,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "test".into(),
        low_res_start: None,
    };
    let mock = Mock::new(&spec);
    let bound = plan.bind_admission(&spec, Some(&context), &req).unwrap();
    assert!(mock
        .begin_memory_strategy_request(&context)
        .unwrap()
        .is_none());
    assert!(mock
        .generate_video_planned_with_execution_report(&req, &bound, None, &mut |_| {})
        .is_ok());
    context.selection.strategy = MemoryStrategy::StagedResidency;
    let forged = plan.bind_admission(&spec, Some(&context), &req).unwrap();
    assert!(mock
        .generate_video_planned_with_execution_report(&req, &forged, None, &mut |_| {})
        .is_err());
    assert_eq!(mock.forward_calls.get(), 1);
    let mut mock = mock;
    let mut contract = crate::MemoryProviderContract::compatibility_default(
        mock.desc.id,
        crate::MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract
        .strategies
        .iter_mut()
        .find(|r| r.strategy == MemoryStrategy::StagedResidency)
        .unwrap()
        .support = crate::MemoryStrategySupport::Implemented;
    let mut req = req;
    let selected = contract.generation_memory(&context.selection);
    req.memory = selected;
    mock.contract = Some(contract);
    let bound = plan.bind_admission(&spec, Some(&context), &req).unwrap();
    assert!(mock
        .generate_video_planned_with_execution_report(&req, &bound, None, &mut |_| {})
        .is_err());
    let mut borrowed = BorrowedScope::default();
    assert!(mock
        .generate_video_planned_with_execution_report(
            &req,
            &bound,
            Some(&mut borrowed),
            &mut |_| {}
        )
        .is_ok());
    assert_eq!(borrowed.events, vec![true, false]);
    req.memory = None;
    let incorrectly_configured = plan.bind_admission(&spec, Some(&context), &req).unwrap();
    assert!(mock
        .generate_video_planned_with_execution_report(
            &req,
            &incorrectly_configured,
            Some(&mut borrowed),
            &mut |_| {}
        )
        .is_err());
    assert_eq!(borrowed.events, vec![true, false]);
}

#[derive(Default)]
struct BorrowedScope {
    events: Vec<bool>,
}
impl MemoryRequestScope for BorrowedScope {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
        panic!("caller owns configure")
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
        panic!("caller owns finish")
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        assert_eq!(phase, MemoryPhase::Denoise);
        self.events.push(true);
        Ok(())
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        assert_eq!(phase, MemoryPhase::Denoise);
        self.events.push(false);
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
