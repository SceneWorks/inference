//! Registered `Generator` wrappers for Candle Z-Image Control (`z_image_turbo_control`,
//! `z_image_control`) (`SW-6982`, `REG-6`, `PIPE-18`).

use candle_gen::gen_core::{
    self, Conditioning, GenerationMemory, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, ModelDescriptor, Progress, WeightsSource,
};

use crate::control::{
    ZImageControl, ZImageControlPaths, ZImageControlRequest, DEFAULT_CONTROL_SCALE,
};

const TURBO_DEFAULT_STEPS: usize = 4;
const BASE_DEFAULT_STEPS: usize = 50;

fn extract_control<'a>(
    request: &'a GenerationRequest,
    route_id: &str,
) -> gen_core::Result<(&'a Image, Option<f32>)> {
    let mut control = None;
    for cond in &request.conditioning {
        if let Conditioning::Control { image, scale, .. } = cond {
            if control.replace((image, *scale)).is_some() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{route_id} accepts at most one Control conditioning image"
                )));
            }
        }
    }
    control.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{route_id} requires a Control conditioning image"
        ))
    })
}

struct ZImageControlGenerator {
    descriptor: ModelDescriptor,
    base: bool,
    model: ZImageControl,
}

impl Generator for ZImageControlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("z_image/control_route", self.descriptor.id.into());
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
        let _ = extract_control(request, self.descriptor.id)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (skeleton, cond_scale) = extract_control(request, self.descriptor.id)?;
        let default_steps = if self.base {
            BASE_DEFAULT_STEPS
        } else {
            TURBO_DEFAULT_STEPS
        };
        let steps = request.steps.map(|s| s as usize).unwrap_or(default_steps);
        let control_scale = cond_scale
            .or(request.control_scale)
            .unwrap_or(DEFAULT_CONTROL_SCALE);
        let base_seed = request.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let control_req = ZImageControlRequest {
                prompt: request.prompt.clone(),
                width: request.width,
                height: request.height,
                steps,
                control_scale,
                guidance: request.guidance,
                scheduler_shift: request.scheduler_shift,
                sampler: request.sampler.as_deref().map(str::to_owned),
                scheduler: request.scheduler.as_deref().map(str::to_owned),
                negative_prompt: request.negative_prompt.clone(),
                seed: base_seed.wrapping_add(idx as u64),
                use_pid: request.use_pid(),
                memory: request.memory.unwrap_or_default(),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
            };
            images.push(
                self.model
                    .generate(&control_req, skeleton, on_progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn load_control_variant(
    descriptor: ModelDescriptor,
    base: bool,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], descriptor.id)?;
    if !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} does not accept extra control, identity, or IP-Adapter overlays",
            descriptor.id
        )));
    }
    let snapshot = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} requires a Z-Image snapshot directory",
                descriptor.id
            )));
        }
    };
    let control = match &spec.control {
        Some(WeightsSource::File(path) | WeightsSource::Dir(path)) => path.clone(),
        None => {
            return Err(gen_core::Error::Msg(format!(
                "{} requires spec.control",
                descriptor.id
            )));
        }
    };
    let paths = ZImageControlPaths {
        snapshot,
        text_encoder: spec.text_encoder.clone(),
        control,
        adapters: spec.adapters.clone(),
        base,
    };
    let mut model =
        ZImageControl::load_with_memory_spec(&paths, spec, GenerationMemory::default())
            .map_err(gen_core::Error::from)?;
    if let Some(pid) = &spec.pid {
        model = model.with_pid(pid).map_err(gen_core::Error::from)?;
    }
    Ok(Box::new(ZImageControlGenerator {
        descriptor,
        base,
        model,
    }))
}

pub(crate) fn load_turbo_control(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_control_variant(crate::turbo_control_descriptor(), false, spec)
}

pub(crate) fn load_base_control(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_control_variant(crate::base_control_descriptor(), true, spec)
}
