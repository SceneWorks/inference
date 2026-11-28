//! Registered `Generator` wrappers for Candle Qwen-Image-Edit aliases and Qwen-Image Fun Control
//! (`SW-6982`, `REG-6`, `B1`, `E4`).

use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec,
    MemoryGeometry, MemoryPhase, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
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

struct ActiveEditAdmission {
    context: MemoryRunContext,
    model: QwenEdit,
}

struct QwenEditMemoryScope {
    inner: crate::memory_strategy::QwenMemoryScope,
    active_admission: Arc<Mutex<Option<ActiveEditAdmission>>>,
}

impl QwenEditMemoryScope {
    fn clear_admission(&self) {
        *candle_gen::lock_recover(&self.active_admission) = None;
    }
}

impl Drop for QwenEditMemoryScope {
    fn drop(&mut self) {
        self.clear_admission();
    }
}

impl MemoryRequestScope for QwenEditMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.inner.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.inner.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.inner.configure_decode(tile_edge, overlap, geometry)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.inner.configure_attention(chunk_size)
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.inner
            .materialize_transformer_window(first_block, block_count)
    }

    fn finish(&mut self, outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.clear_admission();
        self.inner.finish(outcome)
    }
}

struct QwenEditGenerator {
    descriptor: ModelDescriptor,
    paths: QwenEditPaths,
    loaded_spec: LoadSpec,
    model: QwenEdit,
    lightning: bool,
    loaded_quant: Option<gen_core::Quant>,
    memory_contract: Option<gen_core::MemoryProviderContract>,
    active_admission: Arc<Mutex<Option<ActiveEditAdmission>>>,
}

impl QwenEditGenerator {
    fn generate_report_inner(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
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
        let admitted_guard = candle_gen::lock_recover(&self.active_admission);
        if let Some(admitted) = admitted_guard.as_ref() {
            if request.count != 1 {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: admitted memory context requires count=1 (got {})",
                    self.descriptor.id, request.count
                )));
            }
            let edit_req = build_edit_request(request, steps, guidance, base_seed, self.lightning);
            return admitted.model.generate_with_memory_context_with_report(
                &admitted.context,
                &edit_req,
                &references,
                on_progress,
            );
        }
        drop(admitted_guard);
        let mut images = Vec::with_capacity(request.count as usize);
        let mut pag = None;
        for idx in 0..request.count {
            let edit_req = build_edit_request(
                request,
                steps,
                guidance,
                base_seed.wrapping_add(idx as u64),
                self.lightning,
            );
            let report = self
                .model
                .generate_with_report(&edit_req, &references, on_progress)?;
            if pag.is_none() {
                pag = report.pag;
            }
            if let Some(GenerationOutput::Images(mut imgs)) = report.output {
                images.append(&mut imgs);
            }
        }
        Ok(gen_core::GenerationReport {
            pag,
            ..gen_core::GenerationReport::from_output(GenerationOutput::Images(images))
        })
    }
}

impl Generator for QwenEditGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_contract.as_ref()
    }

    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        let Some(contract) = self.memory_contract.as_ref() else {
            return MemorySafetyDecision::Accept;
        };
        let validated = crate::edit::validate_memory_load_spec(&self.loaded_spec)
            .and_then(|()| {
                let recipe = crate::edit::validate_memory_artifact_recipe(&self.loaded_spec)?;
                crate::edit::validate_memory_recipe_context(recipe, context)?;
                Ok(())
            })
            .map_err(|error| gen_core::Error::Unsupported(error.to_string()))
            .and_then(|()| {
                crate::memory_strategy::validate_edit_context(
                    contract,
                    context,
                    self.loaded_quant,
                    !self.loaded_spec.adapters.is_empty(),
                )
            });
        MemorySafetyDecision::from_validation(validated)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_contract.as_ref() else {
            return Ok(None);
        };
        let mut slot = candle_gen::lock_recover(&self.active_admission);
        if slot.is_some() {
            return Err(gen_core::Error::Msg(format!(
                "{}: an admitted memory request scope is already active on this generator",
                self.descriptor.id
            )));
        }
        let admitted_model =
            QwenEdit::load_with_memory_context(&self.paths, &self.loaded_spec, context)
                .map_err(gen_core::Error::from)?;
        let device = admitted_model.device().clone();
        *slot = Some(ActiveEditAdmission {
            context: context.clone(),
            model: admitted_model,
        });
        drop(slot);
        let inner = crate::memory_strategy::QwenMemoryScope::new(
            self.descriptor.id,
            device,
            contract,
            context,
        );
        Ok(Some(Box::new(QwenEditMemoryScope {
            inner,
            active_admission: Arc::clone(&self.active_admission),
        })))
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
        self.generate_report_inner(req, on_progress)
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
        let report = self.generate_report_inner(request, on_progress)?;
        report.output.ok_or_else(|| {
            gen_core::Error::Msg(format!("{}: produced no output", self.descriptor.id))
        })
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
    let loaded_quant = crate::memory_strategy::snapshot_quant_tier(spec, "qwen_image_edit")
        .ok()
        .flatten();
    let model = QwenEdit::load_with_spec(&paths, spec).map_err(gen_core::Error::from)?;
    Ok(Box::new(QwenEditGenerator {
        descriptor,
        paths,
        loaded_spec: spec.clone(),
        model,
        lightning,
        loaded_quant,
        memory_contract,
        active_admission: Arc::new(Mutex::new(None)),
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
            plan.check_request(desc, &edit_req)
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
            .check_request(control_desc, &control_req)
            .expect("control plan check_request succeeds");
        control_plan
            .check_load(&gen_core::RenderLoadBinding::from_spec(&control_spec))
            .expect("control plan check_load succeeds");
    }

    #[test]
    fn registry_loaded_qwen_edit_executes_staged_residency_admission_lifecycle() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp
            .path()
            .join(crate::edit::EDIT_2511_REPO_DIR)
            .join("snapshots")
            .join(crate::edit::EDIT_2511_REVISION)
            .join("bf16");
        std::fs::create_dir_all(&base).expect("create snapshot base");
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &base.join("text_encoder"),
            crate::ENCODER_CONTRACT,
            crate::VISION_ENCODER_CONTRACT,
        )
        .expect("write multimodal encoder fixture");

        let spec = LoadSpec::new(WeightsSource::Dir(base))
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
            .with_resolved_route(crate::edit::EDIT_2511_BASE_ROUTE);
        let registry = crate::provider_registry().expect("qwen registry builds");
        let generator = registry
            .load("qwen_image_edit_2511", &spec)
            .expect("registry loads qwen_image_edit_2511 trait object");
        let contract = generator
            .memory_strategy_contract()
            .expect("resolved snapshot publishes edit memory strategy contract")
            .clone();
        let context = gen_core::standard_memory_behavior_context(
            &contract,
            gen_core::MemoryStrategy::StagedResidency,
            gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            gen_core::MemoryBehaviorRoute {
                mode: gen_core::MemoryMode::Edit,
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .expect("build staged-residency edit memory context");

        assert_eq!(
            generator.memory_strategy_safety_check(&context),
            MemorySafetyDecision::Accept
        );

        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .expect("begin_memory_strategy_request succeeds")
            .expect("staged-residency edit contract opens a request scope");

        let ref_img = Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        };
        let cancel = gen_core::CancelFlag::default();
        cancel.cancel();
        let mut req = GenerationRequest {
            prompt: "relight portrait".into(),
            width: context.geometry.width,
            height: context.geometry.height,
            count: context.geometry.batch,
            conditioning: vec![Conditioning::Reference {
                image: ref_img,
                strength: None,
            }],
            cancel,
            ..Default::default()
        };
        scope
            .configure_request(&mut req)
            .expect("scope configures staged-residency request");
        let staged_memory = req
            .memory
            .expect("scope attaches staged-residency GenerationMemory");
        assert!(staged_memory.stage_residency);

        // Both direct generate() and generate_planned() execute through the admitted
        // QwenEdit::generate_with_memory_context_with_report path, passing memory-authority
        // validation and hitting the early cancel fence inside generate_inner.
        let direct_err = generator
            .generate(&req, &mut |_| {})
            .expect_err("pre-cancelled request should stop at generate_inner cancel fence");
        assert!(
            matches!(direct_err, gen_core::Error::Canceled),
            "expected Canceled after passing admitted memory authority, got: {direct_err}"
        );

        let plan = registry
            .compile_render_plan("qwen_image_edit_2511", &spec, &req)
            .expect("plan compiles")
            .expect("plan returned");
        let planned_err = generator
            .generate_planned(&req, &plan, Some(&mut *scope), &mut |_| {})
            .expect_err("pre-cancelled planned request should stop at generate_inner cancel fence");
        assert!(
            matches!(planned_err, gen_core::Error::Canceled),
            "expected Canceled from planned admitted path, got: {planned_err}"
        );

        // Tampering with memory controls inside an active scope is still rejected by
        // QwenEdit's concrete validate_memory_request check.
        req.memory = None;
        let drifted_err = generator
            .generate(&req, &mut |_| {})
            .expect_err("memory control drift inside active scope must fail closed");
        assert!(
            drifted_err
                .to_string()
                .contains("request memory controls changed after admission"),
            "unexpected drift error: {drifted_err}"
        );

        // Finishing the scope releases the admitted context; constrained memory without a
        // scope is once again rejected by validate_memory_authority.
        scope
            .finish(gen_core::MemoryRunOutcome::Canceled)
            .expect("finish scope succeeds");
        drop(scope);
        req.memory = Some(staged_memory);
        let unadmitted_err = generator
            .generate(&req, &mut |_| {})
            .expect_err("constrained memory outside a scope must be rejected");
        assert!(
            unadmitted_err
                .to_string()
                .contains("constrained memory requires an exact admitted context"),
            "unexpected unadmitted error: {unadmitted_err}"
        );
    }
}
