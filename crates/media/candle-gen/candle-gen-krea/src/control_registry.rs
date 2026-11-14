//! Registry bridge for the native Krea pose loader, including imported transformers.
use crate::control_provider::{Krea2Control, Krea2ControlPaths, Krea2ControlRequest};
use candle_gen::gen_core::{
    self, Conditioning, ConditioningKind, ControlKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, ModelDescriptor, Progress, Quant, WeightsSource, BASE_SNAPSHOT_COMPONENT,
};

pub(crate) fn descriptor() -> ModelDescriptor {
    let mut descriptor = crate::descriptor();
    descriptor.id = "krea_2_turbo_control";
    descriptor.capabilities.conditioning = vec![ConditioningKind::Control];
    descriptor.control_kinds = Some(gen_core::control::AcceptedControlKinds::Only(vec![
        ControlKind::Pose,
    ]));
    descriptor.capabilities.supported_quants = &[Quant::Q4, Quant::Q8];
    descriptor
}

struct ControlGenerator {
    descriptor: ModelDescriptor,
    model: std::cell::RefCell<Option<(bool, Krea2Control)>>,
    paths: Krea2ControlPaths,
    spec: LoadSpec,
    #[cfg(feature = "cuda")]
    device: candle_gen::candle_core::Device,
    memory_contract: gen_core::MemoryProviderContract,
    loaded_quant: Option<Quant>,
}

fn control_input(request: &GenerationRequest) -> gen_core::Result<(&gen_core::Image, Option<f32>)> {
    match request.conditioning.as_slice() {
        [Conditioning::Control {
            image,
            kind: ControlKind::Pose,
            scale,
        }] => Ok((image, *scale)),
        _ => Err(gen_core::Error::Unsupported(
            "Krea pose control requires exactly one pose control image".into(),
        )),
    }
}

impl Generator for ControlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::krea_memory_strategy_safety_check(&self.memory_contract, self.loaded_quant, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        if let gen_core::MemorySafetyDecision::Reject { reason } =
            self.memory_strategy_safety_check(context)
        {
            return Err(gen_core::Error::Unsupported(reason));
        }
        #[cfg(feature = "cuda")]
        return Ok(Some(Box::new(crate::KreaMemoryScope {
            device: self.device.clone(),
            memory: self.memory_contract.generation_memory(&context.selection),
            provider_id: self.descriptor.id.to_owned(),
            mode: context.mode.clone(),
            reference_count: context.geometry.reference_count,
            use_pid: context.use_pid,
            has_phases: context.has_phases,
            finished: false,
        })));
        #[cfg(not(feature = "cuda"))]
        Ok(None)
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if request
            .phases
            .as_ref()
            .is_some_and(|phases| !phases.is_empty())
            || request.use_pid
        {
            return Err(gen_core::Error::Unsupported(
                "Krea pose control does not compose multi-phase or PiD requests".into(),
            ));
        }
        if let Some(memory) = request.memory {
            crate::validate_memory_parameters(self.descriptor.id, memory)?;
            if memory.stream_transformer_blocks {
                return Err(gen_core::Error::Unsupported(
                    "Krea pose control has no transformer streaming path".into(),
                ));
            }
        }
        let (image, _) = control_input(request)?;
        if request.prompt.trim().is_empty()
            || request.steps == Some(0)
            || !request.width.is_multiple_of(crate::SIZE_MULTIPLE)
            || !request.height.is_multiple_of(crate::SIZE_MULTIPLE)
            || (image.width, image.height) != (request.width, request.height)
        {
            return Err(gen_core::Error::Unsupported(
                "Invalid Krea pose prompt, steps, or control geometry".into(),
            ));
        }
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (image, scale) = control_input(request)?;
        let chunk_attention = request.memory.is_some_and(|memory| memory.chunk_attention);
        let mut state = self.model.borrow_mut();
        if state
            .as_ref()
            .is_none_or(|(chunked, _)| *chunked != chunk_attention)
        {
            // Attention chunking is configured on the native load. Drop the previous resident
            // before rebuilding so switching this request policy never holds both trunks at once.
            *state = None;
            let mut paths = self.paths.clone();
            paths.chunk_attention = chunk_attention;
            *state = Some((
                chunk_attention,
                Krea2Control::load_with_spec(&paths, &self.spec).map_err(gen_core::Error::from)?,
            ));
        }
        let model = &state.as_ref().expect("control model initialized").1;
        let mut images = Vec::new();
        for index in 0..request.count {
            let control_request = Krea2ControlRequest {
                guidance: request.guidance,
                negative_prompt: request.negative_prompt.clone(),
                sampler: request.sampler.clone(),
                scheduler: request.scheduler.clone(),
                scheduler_shift: request.scheduler_shift,
                prompt: request.prompt.clone(),
                width: request.width,
                height: request.height,
                steps: request.steps.unwrap_or(crate::TURBO_STEPS as u32) as usize,
                seed: request.seed.unwrap_or(0).wrapping_add(index as u64),
                control_scale: scale
                    .or(request.control_scale)
                    .unwrap_or(crate::DEFAULT_CONTROL_SCALE),
                text_style_gain: request.text_style_gain,
                tile_vae_decode: request.memory.is_some_and(|memory| memory.tile_vae_decode),
                stage_residency: request.memory.is_some_and(|memory| memory.stage_residency),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
            };
            images.push(
                model
                    .generate(&control_request, image, progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub(crate) fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(
        spec,
        &[
            BASE_SNAPSHOT_COMPONENT,
            gen_core::KREA_CONVROT_DIT_COMPONENT,
        ],
        descriptor().id,
    )?;
    if spec.identity.is_some()
        || !spec.extra_controls.is_empty()
        || spec.pid.is_some()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "Krea pose control does not accept identity, extra control, or PiD overlays".into(),
        ));
    }
    let (root, native_dit) = match &spec.weights {
        WeightsSource::Dir(root) => (root.clone(), None),
        WeightsSource::File(file) => {
            let Some(WeightsSource::Dir(root)) = spec.components.get(BASE_SNAPSHOT_COMPONENT)
            else {
                return Err(gen_core::Error::Unsupported(
                    "Imported Krea pose control requires base_snapshot".into(),
                ));
            };
            (root.clone(), Some(file.clone()))
        }
    };
    let Some(WeightsSource::File(control)) = &spec.control else {
        return Err(gen_core::Error::Unsupported(
            "Krea pose control requires a control overlay file".into(),
        ));
    };
    let loaded_quant = crate::actual_quant_tier(spec, descriptor().id)?;
    let paths = Krea2ControlPaths {
        root,
        native_dit,
        control: control.clone(),
        adapters: spec.adapters.clone(),
        convrot_dit: crate::convrot_selector(spec, descriptor().id)?,
        branch_tier: gen_core::tier_integrity::control_branch_tier(loaded_quant),
        chunk_attention: false,
        offload_policy: spec.offload_policy,
    };
    if paths.native_dit.is_some() && paths.convrot_dit.is_some() {
        return Err(gen_core::Error::Unsupported(
            "Krea pose control accepts one replacement transformer".into(),
        ));
    }
    let memory_contract = crate::build_krea_control_memory_strategy_contract(spec)?;
    let model = Krea2Control::load_with_spec(&paths, spec).map_err(gen_core::Error::from)?;
    #[cfg(feature = "cuda")]
    let device = model.device.clone();
    Ok(Box::new(ControlGenerator {
        descriptor: descriptor(),
        model: std::cell::RefCell::new(Some((false, model))),
        paths,
        spec: spec.clone(),
        #[cfg(feature = "cuda")]
        device,
        memory_contract,
        loaded_quant,
    }))
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imported_pose_registration_uses_the_control_provider() {
        let registry = crate::provider_registry().unwrap();
        let descriptor = registry
            .imported_model_descriptor(
                "krea_2",
                gen_core::ImportedModelSource::TransformerFile,
                gen_core::ImportedModelOperation::Pose,
            )
            .unwrap();
        assert_eq!(descriptor.id, "krea_2_turbo_control");
        assert_eq!(
            descriptor.capabilities.conditioning,
            &[ConditioningKind::Control]
        );
        assert_eq!(
            descriptor.control_kinds,
            Some(gen_core::control::AcceptedControlKinds::Only(vec![
                ControlKind::Pose
            ]))
        );
        assert_eq!(descriptor.required_components, &[BASE_SNAPSHOT_COMPONENT]);
        assert!(descriptor.capabilities.supports_lora && descriptor.capabilities.supports_lokr);
        let spec = LoadSpec::new(WeightsSource::File("fine-tune.safetensors".into()));
        assert!(load(&spec)
            .err()
            .unwrap()
            .to_string()
            .contains("base_snapshot"));
    }
}
