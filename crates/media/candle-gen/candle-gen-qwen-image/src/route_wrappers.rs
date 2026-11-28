//! Registered `Generator` wrappers for Candle Qwen-Image-Edit aliases and Qwen-Image Fun Control
//! (`SW-6982`, `REG-6`, `B1`, `E4`).

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec,
    ModelDescriptor, Progress, WeightsSource,
};

use crate::control_fun::{
    QwenFunControl, QwenFunControlPaths, QwenFunControlRequest, DEFAULT_CONTROL_SCALE,
};
use crate::edit::{QwenEdit, QwenEditPaths, QwenEditRequest};
use crate::{DEFAULT_GUIDANCE, MODEL_ID, SIZE_MULTIPLE};

const LIGHTNING_DEFAULT_STEPS: usize = 4;
const LIGHTNING_DEFAULT_GUIDANCE: f32 = 1.0;

fn extract_edit_references(
    request: &GenerationRequest,
    route_id: &str,
) -> gen_core::Result<Vec<Image>> {
    let mut refs = Vec::new();
    for cond in &request.conditioning {
        match cond {
            Conditioning::Reference { image, .. } => refs.push(image.clone()),
            Conditioning::MultiReference { images } => refs.extend(images.iter().cloned()),
            _ => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{route_id} only accepts Reference or MultiReference conditioning"
                )));
            }
        }
    }
    if refs.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{route_id} requires at least one Reference or MultiReference conditioning image"
        )));
    }
    Ok(refs)
}

fn extract_control_input<'a>(
    request: &'a GenerationRequest,
    route_id: &str,
) -> gen_core::Result<(&'a Image, Option<f32>)> {
    match request.conditioning.as_slice() {
        [Conditioning::Control { image, scale, .. }] => Ok((image, *scale)),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{route_id} requires exactly one Control conditioning image"
        ))),
    }
}

fn build_edit_request(
    request: &GenerationRequest,
    steps: usize,
    guidance: f32,
    seed: u64,
    lightning: bool,
) -> QwenEditRequest {
    QwenEditRequest {
        prompt: request.prompt.clone(),
        negative: request.negative_prompt.clone().unwrap_or_default(),
        width: request.width,
        height: request.height,
        steps,
        sampler: request.sampler.as_deref().map(str::to_owned),
        scheduler: request.scheduler.as_deref().map(str::to_owned),
        scheduler_shift: request.scheduler_shift,
        guidance,
        seed,
        lightning,
        stage_residency: request.memory.is_some_and(|m| m.stage_residency),
        memory: request.memory,
        cancel: request.cancel.clone(),
        preview: request.preview.clone(),
        pag: request.features.pag.clone(),
    }
}

struct QwenEditGenerator {
    descriptor: ModelDescriptor,
    loaded_spec: LoadSpec,
    model: QwenEdit,
    lightning: bool,
    memory_contract: Option<gen_core::MemoryProviderContract>,
}

impl Generator for QwenEditGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_contract.as_ref()
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("qwen/edit_route", self.descriptor.id.into());
            candle_gen::execution::policy("qwen/lightning", format!("{}", self.lightning).into());
        })
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let _run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        self.generate(req, on_progress)
            .map(gen_core::GenerationReport::from_output)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_planned_with(self, req, plan, scope, on_progress, || {
            candle_gen::execution::policy("qwen/edit_route", self.descriptor.id.into());
            candle_gen::execution::policy("qwen/lightning", format!("{}", self.lightning).into());
        })
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if request.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        if !request.width.is_multiple_of(SIZE_MULTIPLE)
            || !request.height.is_multiple_of(SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "{}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                self.descriptor.id, request.width, request.height
            )));
        }
        let _ = extract_edit_references(request, self.descriptor.id)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let references = extract_edit_references(request, self.descriptor.id)?;
        let default_steps = if self.lightning {
            LIGHTNING_DEFAULT_STEPS
        } else {
            crate::resolve_steps(None)
        };
        let default_guidance = if self.lightning {
            LIGHTNING_DEFAULT_GUIDANCE
        } else {
            DEFAULT_GUIDANCE
        };
        let steps = request.steps.map(|s| s as usize).unwrap_or(default_steps);
        let guidance = request
            .true_cfg
            .or(request.guidance)
            .unwrap_or(default_guidance);
        let base_seed = request.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let edit_req = build_edit_request(
                request,
                steps,
                guidance,
                base_seed.wrapping_add(idx as u64),
                self.lightning,
            );
            images.push(
                self.model
                    .generate(&edit_req, &references, on_progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn load_edit_with(
    descriptor: ModelDescriptor,
    lightning: bool,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], descriptor.id)?;
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} does not accept control, identity, PiD, or IP-Adapter overlays",
            descriptor.id
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory",
                descriptor.id
            )));
        }
    };
    let paths = QwenEditPaths {
        root,
        text_encoder: spec.text_encoder.clone(),
        adapters: spec.adapters.clone(),
        offload_policy: spec.offload_policy,
    };
    let memory_contract = crate::memory_strategy::provider_contract("qwen_image_edit", spec).ok();
    let model = QwenEdit::load_with_spec(&paths, spec).map_err(gen_core::Error::from)?;
    Ok(Box::new(QwenEditGenerator {
        descriptor,
        loaded_spec: spec.clone(),
        model,
        lightning,
        memory_contract,
    }))
}

pub(crate) fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_with(crate::edit_descriptor(), false, spec)
}

pub(crate) fn load_edit_2509(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_with(crate::edit_2509_descriptor(), false, spec)
}

pub(crate) fn load_edit_2511(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_with(crate::edit_2511_descriptor(), false, spec)
}

pub(crate) fn load_edit_2511_lightning(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_with(crate::edit_2511_lightning_descriptor(), true, spec)
}

fn compile_edit_2509_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    crate::compile_edit_render_plan_for(crate::edit_2509_descriptor(), false, spec, req)
}

fn compile_edit_2511_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    crate::compile_edit_render_plan_for(crate::edit_2511_descriptor(), false, spec, req)
}

fn compile_edit_2511_lightning_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    crate::compile_edit_render_plan_for(crate::edit_2511_lightning_descriptor(), true, spec, req)
}

fn route_edit_alias(
    route_id: &'static str,
    descriptor: fn() -> ModelDescriptor,
    load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
    compile_plan: fn(
        &LoadSpec,
        &GenerationRequest,
    ) -> gen_core::Result<Option<gen_core::RenderPlan>>,
    lightning: bool,
) -> gen_core::RouteRecord {
    let mut route = crate::route_base();
    route.route_id = route_id;
    route.structure.conditioning.kinds = vec![
        gen_core::ConditioningKind::Reference,
        gen_core::ConditioningKind::MultiReference,
    ];
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresEdit;
    if lightning {
        route.defaults.steps = Some(LIGHTNING_DEFAULT_STEPS as u32);
        route.defaults.guidance = Some(LIGHTNING_DEFAULT_GUIDANCE);
    }
    route.bindings = gen_core::RouteBindings::generator(descriptor, load)
        .with_feature_route(Some(MODEL_ID), descriptor)
        .with_render_plan(compile_plan);
    route
}

pub(crate) fn route_edit_2509() -> gen_core::RouteRecord {
    route_edit_alias(
        "qwen_image_edit_2509",
        crate::edit_2509_descriptor,
        load_edit_2509,
        compile_edit_2509_render_plan,
        false,
    )
}

pub(crate) fn route_edit_2511() -> gen_core::RouteRecord {
    route_edit_alias(
        "qwen_image_edit_2511",
        crate::edit_2511_descriptor,
        load_edit_2511,
        compile_edit_2511_render_plan,
        false,
    )
}

pub(crate) fn route_edit_2511_lightning() -> gen_core::RouteRecord {
    route_edit_alias(
        "qwen_image_edit_2511_lightning",
        crate::edit_2511_lightning_descriptor,
        load_edit_2511_lightning,
        compile_edit_2511_lightning_render_plan,
        true,
    )
}

struct QwenFunControlGenerator {
    descriptor: ModelDescriptor,
    loaded_spec: LoadSpec,
    model: QwenFunControl,
}

impl Generator for QwenFunControlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("qwen/control_route", self.descriptor.id.into());
        })
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let _run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        self.generate(req, on_progress)
            .map(gen_core::GenerationReport::from_output)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_planned_with(self, req, plan, scope, on_progress, || {
            candle_gen::execution::policy("qwen/control_route", self.descriptor.id.into());
        })
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if request.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        if !request.width.is_multiple_of(SIZE_MULTIPLE)
            || !request.height.is_multiple_of(SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "{}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                self.descriptor.id, request.width, request.height
            )));
        }
        let _ = extract_control_input(request, self.descriptor.id)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (control_image, cond_scale) = extract_control_input(request, self.descriptor.id)?;
        let steps = crate::resolve_steps(request.steps);
        let guidance = request
            .true_cfg
            .or(request.guidance)
            .unwrap_or(DEFAULT_GUIDANCE);
        let control_scale = cond_scale
            .or(request.control_scale)
            .unwrap_or(DEFAULT_CONTROL_SCALE);
        let base_seed = request.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let control_req = QwenFunControlRequest {
                prompt: request.prompt.clone(),
                negative: request.negative_prompt.clone().unwrap_or_default(),
                width: request.width,
                height: request.height,
                steps,
                sampler: request.sampler.as_deref().map(str::to_owned),
                scheduler: request.scheduler.as_deref().map(str::to_owned),
                scheduler_shift: request.scheduler_shift,
                guidance,
                control_scale,
                seed: base_seed.wrapping_add(idx as u64),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
                pag: request.features.pag.clone(),
            };
            images.push(
                self.model
                    .generate(&control_req, control_image, on_progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub(crate) fn load_control(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], crate::control_descriptor().id)?;
    if !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "qwen_image_control does not accept extra control, identity, PiD, or IP-Adapter overlays"
                .into(),
        ));
    }
    let qwen_base = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "qwen_image_control requires a base snapshot directory".into(),
            ));
        }
    };
    let controlnet = match &spec.control {
        Some(WeightsSource::File(path) | WeightsSource::Dir(path)) => path.clone(),
        None => {
            return Err(gen_core::Error::Msg(
                "qwen_image_control requires spec.control".into(),
            ));
        }
    };
    let paths = QwenFunControlPaths {
        qwen_base,
        text_encoder: spec.text_encoder.clone(),
        controlnet,
        adapters: spec.adapters.clone(),
    };
    let model = QwenFunControl::load_with_spec(&paths, spec).map_err(gen_core::Error::from)?;
    Ok(Box::new(QwenFunControlGenerator {
        descriptor: crate::control_descriptor(),
        loaded_spec: spec.clone(),
        model,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_edit_request_forwards_generation_memory() {
        let memory = gen_core::GenerationMemory {
            stage_residency: true,
            decode_tile_edge: Some(512),
            attention_chunk_size: Some(65_536),
            ..Default::default()
        };
        let request = GenerationRequest {
            prompt: "relight in warm sunset".into(),
            width: 512,
            height: 512,
            memory: Some(memory),
            ..Default::default()
        };
        let edit_req = build_edit_request(&request, 4, 1.0, 42, true);
        assert!(edit_req.stage_residency);
        assert_eq!(edit_req.memory, Some(memory));
    }

    #[test]
    fn all_qwen_edit_aliases_and_control_compile_route_specific_render_plans() {
        let registry = crate::provider_registry().expect("qwen registry builds");
        let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "/tmp/qwen-dir",
        )));
        let ref_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };
        let edit_req = GenerationRequest {
            prompt: "relight portrait".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Reference {
                image: ref_img.clone(),
                strength: None,
            }],
            ..Default::default()
        };
        let default_steps = crate::resolve_steps(None);
        for (route_id, expected_steps) in [
            ("qwen_image_edit", default_steps),
            ("qwen_image_edit_2509", default_steps),
            ("qwen_image_edit_2511", default_steps),
            ("qwen_image_edit_2511_lightning", LIGHTNING_DEFAULT_STEPS),
        ] {
            let plan = registry
                .compile_render_plan(route_id, &spec, &edit_req)
                .unwrap_or_else(|e| panic!("{route_id} plan compile failed: {e}"))
                .unwrap_or_else(|| panic!("{route_id} must return a RenderPlan"));
            let desc = registry
                .generator_descriptor(route_id)
                .expect("descriptor registered");
            plan.check_request(&desc, &edit_req)
                .unwrap_or_else(|e| panic!("{route_id} plan check_request failed: {e}"));
            plan.check_load(&gen_core::RenderLoadBinding::from_spec(&spec))
                .unwrap_or_else(|e| panic!("{route_id} plan check_load failed: {e}"));
            assert_eq!(plan.executed_steps(), expected_steps, "{route_id}");
        }

        let control_spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "/tmp/qwen-dir",
        )))
        .with_control(WeightsSource::File(std::path::PathBuf::from(
            "/tmp/qwen-control.safetensors",
        )));
        let control_req = GenerationRequest {
            prompt: "studio pose".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Control {
                image: ref_img,
                kind: gen_core::ControlKind::Pose,
                scale: Some(0.8),
            }],
            ..Default::default()
        };
        let control_plan = registry
            .compile_render_plan("qwen_image_control", &control_spec, &control_req)
            .expect("qwen_image_control plan compiles")
            .expect("qwen_image_control returns RenderPlan");
        let control_desc = registry
            .generator_descriptor("qwen_image_control")
            .expect("control descriptor registered");
        control_plan
            .check_request(&control_desc, &control_req)
            .expect("control plan check_request succeeds");
        control_plan
            .check_load(&gen_core::RenderLoadBinding::from_spec(&control_spec))
            .expect("control plan check_load succeeds");
    }
}
