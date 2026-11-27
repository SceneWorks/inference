//! Registered `Generator` wrappers for Candle Kolors IP-Adapter (`kolors_ip_adapter`) and
//! Kolors ControlNet Pose (`kolors_control`) (`SW-6982`, `REG-6`, `PIPE-18`).

use candle_gen::gen_core::{
    self, Conditioning, ControlKind, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, ModelDescriptor, Progress, WeightsSource,
};

use crate::control::{KolorsControl, KolorsControlPaths, KolorsControlRequest};
use crate::ip_provider::{IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest};
use crate::{DEFAULT_CONTROL_SCALE, DEFAULT_IP_ADAPTER_SCALE, SIZE_MULTIPLE};

const DEFAULT_STEPS: usize = 50;
const DEFAULT_GUIDANCE: f32 = 5.0;

fn extract_reference<'a>(
    request: &'a GenerationRequest,
    route_id: &str,
) -> gen_core::Result<(&'a Image, Option<f32>)> {
    match request.conditioning.as_slice() {
        [Conditioning::Reference { image, strength }] => Ok((image, *strength)),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{route_id} requires exactly one Reference conditioning image"
        ))),
    }
}

fn extract_control<'a>(
    request: &'a GenerationRequest,
    route_id: &str,
) -> gen_core::Result<(&'a Image, Option<f32>)> {
    match request.conditioning.as_slice() {
        [Conditioning::Control {
            image,
            kind: ControlKind::Pose,
            scale,
        }] => Ok((image, *scale)),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{route_id} requires exactly one Pose Control conditioning image"
        ))),
    }
}

pub(crate) fn route_ip_adapter() -> gen_core::RouteRecord {
    let mut route = crate::route_base();
    route.route_id = "kolors_ip_adapter";
    route.structure.conditioning.kinds = vec![gen_core::ConditioningKind::Reference];
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresReference;
    route.structure.conditioning.modules = vec!["kolors_ip_adapter"];
    route.structure.weight_plan.requires_ip_adapter_overlay = true;
    route.structure.denoiser =
        gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new("kolors_sdxl_unet"));
    route.defaults.strength = Some(DEFAULT_IP_ADAPTER_SCALE);
    route.bindings = gen_core::RouteBindings::generator(ip_adapter_descriptor, load_ip_adapter)
        .with_feature_route(Some(crate::MODEL_ID), ip_adapter_descriptor);
    route
}

pub fn ip_adapter_descriptor() -> ModelDescriptor {
    route_ip_adapter().compute_descriptor()
}

struct KolorsIpAdapterGenerator {
    descriptor: ModelDescriptor,
    model: std::cell::RefCell<IpAdapterKolors>,
}

impl Generator for KolorsIpAdapterGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("kolors/route", self.descriptor.id.into());
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
                "{}: width/height must be multiples of {SIZE_MULTIPLE}",
                self.descriptor.id
            )));
        }
        let _ = extract_reference(request, self.descriptor.id)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (reference, strength) = extract_reference(request, self.descriptor.id)?;
        let steps = request.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
        let guidance = request.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let ip_adapter_scale = strength.unwrap_or(DEFAULT_IP_ADAPTER_SCALE);
        let base_seed = request.seed.unwrap_or(0);
        let mut model = self.model.borrow_mut();
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let ip_req = IpAdapterKolorsRequest {
                prompt: request.prompt.clone(),
                negative: request.negative_prompt.clone().unwrap_or_default(),
                width: request.width,
                height: request.height,
                steps,
                guidance,
                ip_adapter_scale,
                sampler: request.sampler.as_deref().map(str::to_owned),
                scheduler: request.scheduler.as_deref().map(str::to_owned),
                seed: base_seed.wrapping_add(idx as u64),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
            };
            images.push(
                model
                    .generate(&ip_req, reference, on_progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub(crate) fn load_ip_adapter(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], ip_adapter_descriptor().id)?;
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.pid.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "kolors_ip_adapter does not accept control, identity, or PiD overlays".into(),
        ));
    }
    let kolors_base = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "kolors_ip_adapter requires a Kolors snapshot directory".into(),
            ));
        }
    };
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(path) | WeightsSource::File(path)) => path.clone(),
        None => {
            return Err(gen_core::Error::Msg(
                "kolors_ip_adapter requires spec.ip_adapter".into(),
            ));
        }
    };
    let paths = IpAdapterKolorsPaths {
        kolors_base,
        ip_adapter,
        adapters: spec.adapters.clone(),
    };
    let model = IpAdapterKolors::load(&paths).map_err(gen_core::Error::from)?;
    Ok(Box::new(KolorsIpAdapterGenerator {
        descriptor: ip_adapter_descriptor(),
        model: std::cell::RefCell::new(model),
    }))
}

pub(crate) fn route_control() -> gen_core::RouteRecord {
    let mut route = crate::route_base();
    route.route_id = "kolors_control";
    route.structure.conditioning.kinds = vec![gen_core::ConditioningKind::Control];
    route.structure.conditioning.control_kinds = Some(gen_core::AcceptedControlKinds::Only(vec![
        ControlKind::Pose,
    ]));
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresControl;
    route.structure.conditioning.modules = vec!["kolors_controlnet_pose"];
    route.structure.weight_plan.requires_control_overlay = true;
    route.structure.denoiser =
        gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new("kolors_sdxl_unet"));
    route.defaults.strength = Some(DEFAULT_CONTROL_SCALE);
    route.bindings = gen_core::RouteBindings::generator(control_descriptor, load_control)
        .with_feature_route(Some(crate::MODEL_ID), control_descriptor);
    route
}

pub fn control_descriptor() -> ModelDescriptor {
    route_control().compute_descriptor()
}

struct KolorsControlGenerator {
    descriptor: ModelDescriptor,
    model: KolorsControl,
}

impl Generator for KolorsControlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("kolors/route", self.descriptor.id.into());
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
                "{}: width/height must be multiples of {SIZE_MULTIPLE}",
                self.descriptor.id
            )));
        }
        let _ = extract_control(request, self.descriptor.id)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (control_image, cond_scale) = extract_control(request, self.descriptor.id)?;
        let steps = request.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
        let guidance = request.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let control_scale = cond_scale
            .or(request.control_scale)
            .unwrap_or(DEFAULT_CONTROL_SCALE);
        let base_seed = request.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let control_req = KolorsControlRequest {
                prompt: request.prompt.clone(),
                negative: request.negative_prompt.clone().unwrap_or_default(),
                width: request.width,
                height: request.height,
                steps,
                guidance,
                control_scale,
                sampler: request.sampler.as_deref().map(str::to_owned),
                scheduler: request.scheduler.as_deref().map(str::to_owned),
                seed: base_seed.wrapping_add(idx as u64),
                use_pid: request.use_pid(),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
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
    gen_core::reject_unknown_components(spec, &[], control_descriptor().id)?;
    if !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "kolors_control does not accept extra control, identity, or IP-Adapter overlays".into(),
        ));
    }
    let kolors_base = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "kolors_control requires a Kolors snapshot directory".into(),
            ));
        }
    };
    let controlnet = match &spec.control {
        Some(WeightsSource::File(path) | WeightsSource::Dir(path)) => path.clone(),
        None => {
            return Err(gen_core::Error::Msg(
                "kolors_control requires spec.control".into(),
            ));
        }
    };
    let paths = KolorsControlPaths {
        kolors_base,
        controlnet,
        adapters: spec.adapters.clone(),
    };
    let mut model = KolorsControl::load(&paths).map_err(gen_core::Error::from)?;
    if let Some(pid) = &spec.pid {
        model = model.with_pid(pid).map_err(gen_core::Error::from)?;
    }
    Ok(Box::new(KolorsControlGenerator {
        descriptor: control_descriptor(),
        model,
    }))
}
