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

fn extract_edit_references(request: &GenerationRequest, route_id: &str) -> gen_core::Result<Vec<Image>> {
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

struct QwenEditGenerator {
    descriptor: ModelDescriptor,
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
            let edit_req = QwenEditRequest {
                prompt: request.prompt.clone(),
                negative: request.negative_prompt.clone().unwrap_or_default(),
                width: request.width,
                height: request.height,
                steps,
                sampler: request.sampler.as_deref().map(str::to_owned),
                scheduler: request.scheduler.as_deref().map(str::to_owned),
                scheduler_shift: request.scheduler_shift,
                guidance,
                seed: base_seed.wrapping_add(idx as u64),
                lightning: self.lightning,
                stage_residency: request.memory.is_some_and(|m| m.stage_residency),
                memory: None,
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
                pag: request.features.pag.clone(),
            };
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

fn route_edit_alias(
    route_id: &'static str,
    descriptor: fn() -> ModelDescriptor,
    load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
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
        .with_render_plan(crate::compile_edit_render_plan);
    route
}

pub(crate) fn route_edit_2509() -> gen_core::RouteRecord {
    route_edit_alias(
        "qwen_image_edit_2509",
        crate::edit_2509_descriptor,
        load_edit_2509,
        false,
    )
}

pub(crate) fn route_edit_2511() -> gen_core::RouteRecord {
    route_edit_alias(
        "qwen_image_edit_2511",
        crate::edit_2511_descriptor,
        load_edit_2511,
        false,
    )
}

pub(crate) fn route_edit_2511_lightning() -> gen_core::RouteRecord {
    route_edit_alias(
        "qwen_image_edit_2511_lightning",
        crate::edit_2511_lightning_descriptor,
        load_edit_2511_lightning,
        true,
    )
}

struct QwenFunControlGenerator {
    descriptor: ModelDescriptor,
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
        model,
    }))
}
