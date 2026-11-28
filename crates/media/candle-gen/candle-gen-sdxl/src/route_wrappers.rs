//! Registered `Generator` wrappers for Candle SDXL Edit (`sdxl_edit`), Detail (`sdxl_detail`),
//! Control (`sdxl_control`), and IP-Adapter (`sdxl_ip_adapter`) (`SW-6982`, `REG-6`, `PIPE-18`).

use std::cell::RefCell;

use candle_gen::gen_core::{
    self, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, ModelDescriptor, Progress, WeightsSource,
};

use crate::control_provider::SdxlControlGenerator;
use crate::detail_provider::{SdxlDetail, SdxlDetailPaths, SdxlDetailRequest};
use crate::edit_provider::{SdxlEdit, SdxlEditPaths, SdxlEditRequest, DEFAULT_EDIT_STRENGTH};
use crate::ip_provider::{
    IpAdapterSdxl, IpAdapterSdxlPaths, IpAdapterSdxlRequest, DEFAULT_IP_ADAPTER_SCALE,
};
use crate::memory_strategy::SdxlArtifactSeal;
use crate::pipeline::{SdxlComponents, REQUIRED_COMPONENTS};
use crate::{MODEL_ID, SIZE_MULTIPLE};

fn resolve_strength(
    conditioning_strength: Option<f32>,
    request_strength: Option<f32>,
    default_strength: f32,
) -> f32 {
    conditioning_strength
        .or(request_strength)
        .unwrap_or(default_strength)
}

fn require_vae(components: &SdxlComponents, route_id: &str) -> gen_core::Result<WeightsSource> {
    components.vae_fp16_fix.clone().ok_or_else(|| {
        gen_core::Error::Msg(format!("{route_id} requires the `vae_fp16_fix` component"))
    })
}

fn require_dir_weights(spec: &LoadSpec, route_id: &str) -> gen_core::Result<std::path::PathBuf> {
    match &spec.weights {
        WeightsSource::Dir(root) => Ok(root.clone()),
        WeightsSource::File(_) => Err(gen_core::Error::Msg(format!(
            "{route_id} requires a diffusers SDXL snapshot directory"
        ))),
    }
}

// -------------------------------------------------------------------------------------------------
// sdxl_edit
// -------------------------------------------------------------------------------------------------

pub(crate) fn route_sdxl_edit() -> gen_core::RouteRecord {
    let mut route = crate::route_sdxl();
    route.route_id = "sdxl_edit";
    route.structure.conditioning.kinds = vec![ConditioningKind::Reference, ConditioningKind::Mask];
    route.structure.conditioning.control_kinds = None;
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresEdit;
    route.structure.sampling = gen_core::SamplingSurface {
        samplers: vec!["euler_ancestral"],
        schedulers: vec!["discrete"],
        guidance_methods: vec!["cfg"],
        supports_guidance: true,
        ..Default::default()
    };
    route.structure.denoiser =
        gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new("sdxl_unet"));
    route.defaults.steps = Some(30);
    route.defaults.guidance = Some(5.0);
    route.defaults.sampler = Some("euler_ancestral");
    route.defaults.schedule = Some("discrete");
    route.defaults.strength = Some(DEFAULT_EDIT_STRENGTH);
    route.bindings = gen_core::RouteBindings::generator(edit_descriptor, load_edit)
        .with_feature_route(Some(MODEL_ID), edit_descriptor);
    route
}

pub fn edit_descriptor() -> ModelDescriptor {
    route_sdxl_edit().compute_descriptor()
}

fn extract_edit_inputs(
    request: &GenerationRequest,
) -> gen_core::Result<(&Image, Option<f32>, Option<&Image>)> {
    let mut source = None;
    let mut strength = None;
    let mut mask = None;
    for cond in &request.conditioning {
        match cond {
            Conditioning::Reference { image, strength: s } => {
                if source.replace(image).is_some() {
                    return Err(gen_core::Error::Unsupported(
                        "sdxl_edit accepts at most one Reference conditioning image".into(),
                    ));
                }
                strength = *s;
            }
            Conditioning::Mask { image } => {
                if mask.replace(image).is_some() {
                    return Err(gen_core::Error::Unsupported(
                        "sdxl_edit accepts at most one Mask conditioning image".into(),
                    ));
                }
            }
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "sdxl_edit only accepts Reference and optional Mask conditioning".into(),
                ));
            }
        }
    }
    let source = source.ok_or_else(|| {
        gen_core::Error::Msg("sdxl_edit requires a Reference conditioning image".into())
    })?;
    Ok((source, strength, mask))
}

struct SdxlEditGenerator {
    descriptor: ModelDescriptor,
    model: SdxlEdit,
}

impl Generator for SdxlEditGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("sdxl/route", self.descriptor.id.into());
        })
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if request.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "sdxl_edit: prompt must not be empty".into(),
            ));
        }
        if !request.width.is_multiple_of(SIZE_MULTIPLE)
            || !request.height.is_multiple_of(SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "sdxl_edit: width/height must be multiples of {SIZE_MULTIPLE}"
            )));
        }
        let _ = extract_edit_inputs(request)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (source, strength, mask) = extract_edit_inputs(request)?;
        let defaults = SdxlEditRequest::default();
        let steps = request.steps.map(|s| s as usize).unwrap_or(defaults.steps);
        let guidance = request.guidance.unwrap_or(defaults.guidance);
        let strength = resolve_strength(strength, request.strength, defaults.strength);
        let base_seed = request.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let edit_req = SdxlEditRequest {
                prompt: request.prompt.clone(),
                negative: request.negative_prompt.clone().unwrap_or_default(),
                width: request.width,
                height: request.height,
                steps,
                guidance,
                strength,
                seed: base_seed.wrapping_add(idx as u64),
                use_pid: request.use_pid(),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
            };
            let image = match mask {
                Some(mask_img) => {
                    self.model
                        .generate_masked(&edit_req, source, mask_img, on_progress)
                }
                None => self.model.generate(&edit_req, source, on_progress),
            }
            .map_err(gen_core::Error::from)?;
            images.push(image);
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub(crate) fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl_edit does not accept control, identity, or IP-Adapter overlays".into(),
        ));
    }
    let sdxl_base = require_dir_weights(spec, "sdxl_edit")?;
    let components = SdxlComponents::from_spec(spec, "sdxl_edit")?;
    let paths = SdxlEditPaths {
        sdxl_base,
        tokenizer_clip_l: components.tokenizer_clip_l.clone(),
        tokenizer_clip_bigg: components.tokenizer_clip_bigg.clone(),
        vae_fp16_fix: require_vae(&components, "sdxl_edit")?,
        adapters: spec.adapters.clone(),
    };
    let mut model = SdxlEdit::load(&paths).map_err(gen_core::Error::from)?;
    if let Some(pid) = &spec.pid {
        model = model.with_pid(pid).map_err(gen_core::Error::from)?;
    }
    Ok(Box::new(SdxlEditGenerator {
        descriptor: edit_descriptor(),
        model,
    }))
}

// -------------------------------------------------------------------------------------------------
// sdxl_detail
// -------------------------------------------------------------------------------------------------

pub(crate) fn route_sdxl_detail() -> gen_core::RouteRecord {
    let mut route = crate::route_sdxl();
    route.route_id = "sdxl_detail";
    route.structure.conditioning.kinds =
        vec![ConditioningKind::Reference, ConditioningKind::Control];
    route.structure.conditioning.control_kinds = Some(gen_core::AcceptedControlKinds::Any);
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresReference;
    route.structure.sampling = gen_core::SamplingSurface {
        samplers: vec!["euler_ancestral"],
        schedulers: vec!["discrete"],
        guidance_methods: vec!["cfg"],
        supports_guidance: true,
        ..Default::default()
    };
    route.structure.weight_plan.requires_control_overlay = true;
    route.structure.denoiser =
        gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new("sdxl_unet"));
    route.defaults.steps = Some(24);
    route.defaults.guidance = Some(5.0);
    route.defaults.sampler = Some("euler_ancestral");
    route.defaults.schedule = Some("discrete");
    route.defaults.strength = Some(0.55);
    route.bindings = gen_core::RouteBindings::generator(detail_descriptor, load_detail)
        .with_feature_route(Some(MODEL_ID), detail_descriptor);
    route
}

pub fn detail_descriptor() -> ModelDescriptor {
    route_sdxl_detail().compute_descriptor()
}

fn extract_detail_inputs(
    request: &GenerationRequest,
) -> gen_core::Result<(&Image, &Image, Option<f32>, Option<f32>)> {
    let mut source = None;
    let mut strength = None;
    let mut control = None;
    let mut control_scale = None;
    for cond in &request.conditioning {
        match cond {
            Conditioning::Reference { image, strength: s } => {
                if source.replace(image).is_some() {
                    return Err(gen_core::Error::Unsupported(
                        "sdxl_detail accepts at most one Reference conditioning image".into(),
                    ));
                }
                strength = *s;
            }
            Conditioning::Control { image, scale, .. } => {
                if control.replace(image).is_some() {
                    return Err(gen_core::Error::Unsupported(
                        "sdxl_detail accepts at most one Control conditioning image".into(),
                    ));
                }
                control_scale = *scale;
            }
            _ => {
                return Err(gen_core::Error::Unsupported(
                    "sdxl_detail only accepts Reference and optional Control conditioning".into(),
                ));
            }
        }
    }
    let source = source.ok_or_else(|| {
        gen_core::Error::Msg("sdxl_detail requires a Reference conditioning image".into())
    })?;
    let control = control.unwrap_or(source);
    Ok((source, control, strength, control_scale))
}

struct SdxlDetailGenerator {
    descriptor: ModelDescriptor,
    model: SdxlDetail,
}

impl Generator for SdxlDetailGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("sdxl/route", self.descriptor.id.into());
        })
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if !request.width.is_multiple_of(SIZE_MULTIPLE)
            || !request.height.is_multiple_of(SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "sdxl_detail: width/height must be multiples of {SIZE_MULTIPLE}"
            )));
        }
        let _ = extract_detail_inputs(request)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (source, control, strength, cond_scale) = extract_detail_inputs(request)?;
        let defaults = SdxlDetailRequest::default();
        let prompt = if request.prompt.trim().is_empty() {
            defaults.prompt.clone()
        } else {
            request.prompt.clone()
        };
        let negative = request
            .negative_prompt
            .clone()
            .unwrap_or_else(|| defaults.negative.clone());
        let steps = request.steps.map(|s| s as usize).unwrap_or(defaults.steps);
        let guidance = request.guidance.unwrap_or(defaults.guidance);
        let strength = resolve_strength(strength, request.strength, defaults.strength);
        let control_scale = cond_scale
            .or(request.control_scale)
            .unwrap_or(defaults.control_scale);
        let base_seed = request.seed.unwrap_or(defaults.seed);
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let detail_req = SdxlDetailRequest {
                prompt: prompt.clone(),
                negative: negative.clone(),
                width: request.width,
                height: request.height,
                steps,
                guidance,
                strength,
                control_scale,
                seed: base_seed.wrapping_add(idx as u64),
                cancel: request.cancel.clone(),
                preview: request.preview.clone(),
            };
            images.push(
                self.model
                    .generate(&detail_req, source, control, on_progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub(crate) fn load_detail(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    if !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "sdxl_detail does not accept extra control, identity, IP-Adapter, or PiD overlays"
                .into(),
        ));
    }
    let sdxl_base = require_dir_weights(spec, "sdxl_detail")?;
    let tile_controlnet = spec.control.clone().ok_or_else(|| {
        gen_core::Error::Msg("sdxl_detail requires spec.control (tile ControlNet)".into())
    })?;
    let components = SdxlComponents::from_spec(spec, "sdxl_detail")?;
    let paths = SdxlDetailPaths {
        sdxl_base,
        tokenizer_clip_l: components.tokenizer_clip_l.clone(),
        tokenizer_clip_bigg: components.tokenizer_clip_bigg.clone(),
        vae_fp16_fix: require_vae(&components, "sdxl_detail")?,
        tile_controlnet,
        adapters: spec.adapters.clone(),
    };
    let model = SdxlDetail::load(&paths).map_err(gen_core::Error::from)?;
    Ok(Box::new(SdxlDetailGenerator {
        descriptor: detail_descriptor(),
        model,
    }))
}

// -------------------------------------------------------------------------------------------------
// sdxl_control
// -------------------------------------------------------------------------------------------------

pub(crate) fn route_sdxl_control() -> gen_core::RouteRecord {
    let mut route = crate::route_sdxl();
    route.route_id = "sdxl_control";
    route.structure.conditioning.kinds = vec![ConditioningKind::Control];
    route.structure.conditioning.control_kinds = Some(gen_core::AcceptedControlKinds::Any);
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresControl;
    route.structure.weight_plan.requires_control_overlay = true;
    route.structure.denoiser =
        gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new("sdxl_unet"));
    route.bindings = gen_core::RouteBindings::generator(control_route_descriptor, load_control)
        .with_feature_route(Some(MODEL_ID), control_route_descriptor);
    route
}

pub fn control_route_descriptor() -> ModelDescriptor {
    route_sdxl_control().compute_descriptor()
}

pub(crate) fn load_control(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    if spec.control.is_none() {
        return Err(gen_core::Error::Msg(
            "sdxl_control requires spec.control".into(),
        ));
    }
    let component_paths = SdxlComponents::from_spec(spec, "sdxl_control")?;
    let authoritative_seal = if spec.resolved_route.is_some() {
        Some(SdxlArtifactSeal::capture(spec)?)
    } else {
        None
    };
    let generator = SdxlControlGenerator::new(
        spec,
        component_paths,
        candle_gen::default_device()?,
        authoritative_seal,
    )?
    .with_descriptor(control_route_descriptor());
    Ok(Box::new(generator))
}

// -------------------------------------------------------------------------------------------------
// sdxl_ip_adapter
// -------------------------------------------------------------------------------------------------

pub(crate) fn route_sdxl_ip_adapter() -> gen_core::RouteRecord {
    let mut route = crate::route_sdxl();
    route.route_id = "sdxl_ip_adapter";
    route.structure.conditioning.kinds = vec![ConditioningKind::Reference];
    route.structure.conditioning.control_kinds = None;
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresReference;
    route.structure.conditioning.modules = vec!["sdxl_ip_adapter_plus"];
    route.structure.sampling.guidance_methods = vec!["cfg"];
    route.structure.weight_plan.requires_ip_adapter_overlay = true;
    route.structure.denoiser =
        gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new("sdxl_unet"));
    route.defaults.guidance = Some(5.0);
    route.defaults.sampler = Some("euler_ancestral");
    route.defaults.schedule = Some("discrete");
    route.defaults.strength = Some(DEFAULT_IP_ADAPTER_SCALE);
    route.bindings = gen_core::RouteBindings::generator(ip_adapter_descriptor, load_ip_adapter)
        .with_feature_route(Some(MODEL_ID), ip_adapter_descriptor);
    route
}

pub fn ip_adapter_descriptor() -> ModelDescriptor {
    route_sdxl_ip_adapter().compute_descriptor()
}

struct SdxlIpAdapterGenerator {
    descriptor: ModelDescriptor,
    model: RefCell<IpAdapterSdxl>,
}

impl Generator for SdxlIpAdapterGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_with(self, req, on_progress, || {
            candle_gen::execution::policy("sdxl/route", self.descriptor.id.into());
        })
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if request.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "sdxl_ip_adapter: prompt must not be empty".into(),
            ));
        }
        if !request.width.is_multiple_of(SIZE_MULTIPLE)
            || !request.height.is_multiple_of(SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "sdxl_ip_adapter: width/height must be multiples of {SIZE_MULTIPLE}"
            )));
        }
        match request.conditioning.as_slice() {
            [Conditioning::Reference { .. }] => Ok(()),
            _ => Err(gen_core::Error::Unsupported(
                "sdxl_ip_adapter requires exactly one Reference conditioning image".into(),
            )),
        }
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        let (reference, strength) = match request.conditioning.as_slice() {
            [Conditioning::Reference { image, strength }] => (image, *strength),
            _ => unreachable!("validated above"),
        };
        let defaults = IpAdapterSdxlRequest::default();
        let steps = request.steps.map(|s| s as usize).unwrap_or(defaults.steps);
        let guidance = request.guidance.unwrap_or(defaults.guidance);
        let ip_adapter_scale =
            resolve_strength(strength, request.strength, defaults.ip_adapter_scale);
        let base_seed = request.seed.unwrap_or(0);
        let mut model = self.model.borrow_mut();
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let ip_req = IpAdapterSdxlRequest {
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
                use_pid: request.use_pid(),
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
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.identity.is_some() {
        return Err(gen_core::Error::Unsupported(
            "sdxl_ip_adapter does not accept control or identity overlays".into(),
        ));
    }
    let sdxl_base = require_dir_weights(spec, "sdxl_ip_adapter")?;
    let mut allowed: Vec<&str> = REQUIRED_COMPONENTS.to_vec();
    allowed.push("image_encoder");
    gen_core::reject_unknown_components(spec, &allowed, "sdxl_ip_adapter")?;
    let tokenizer_clip_l = gen_core::require_component(
        spec,
        "sdxl_ip_adapter",
        "tokenizer_clip_l",
        "CLIP-L tokenizer",
    )?
    .clone();
    let tokenizer_clip_bigg = gen_core::require_component(
        spec,
        "sdxl_ip_adapter",
        "tokenizer_clip_bigg",
        "CLIP-bigG tokenizer",
    )?
    .clone();
    let vae_fp16_fix =
        gen_core::require_component(spec, "sdxl_ip_adapter", "vae_fp16_fix", "fp16-fix SDXL VAE")?
            .clone();
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::File(path) | WeightsSource::Dir(path)) => path.clone(),
        None => {
            return Err(gen_core::Error::Msg(
                "sdxl_ip_adapter requires spec.ip_adapter".into(),
            ));
        }
    };
    let image_encoder = match spec.components.get("image_encoder") {
        Some(WeightsSource::Dir(path) | WeightsSource::File(path)) => path.clone(),
        None => sdxl_base.join("image_encoder"),
    };
    let paths = IpAdapterSdxlPaths {
        sdxl_base,
        ip_adapter,
        image_encoder,
        tokenizer_clip_l,
        tokenizer_clip_bigg,
        vae_fp16_fix,
        adapters: spec.adapters.clone(),
    };
    let mut model = IpAdapterSdxl::load(&paths).map_err(gen_core::Error::from)?;
    if let Some(pid) = &spec.pid {
        model = model.with_pid(pid).map_err(gen_core::Error::from)?;
    }
    Ok(Box::new(SdxlIpAdapterGenerator {
        descriptor: ip_adapter_descriptor(),
        model: RefCell::new(model),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_image() -> Image {
        Image {
            width: 64,
            height: 64,
            pixels: vec![128; 64 * 64 * 3],
        }
    }

    #[test]
    fn sdxl_edit_and_detail_restrict_sampling_capabilities_to_implemented_controls() {
        for (id, desc) in [
            ("sdxl_edit", edit_descriptor()),
            ("sdxl_detail", detail_descriptor()),
        ] {
            assert_eq!(desc.capabilities.samplers, vec!["euler_ancestral"], "{id}");
            assert_eq!(desc.capabilities.schedulers, vec!["discrete"], "{id}");
            assert_eq!(
                desc.capabilities.supported_guidance_methods,
                vec!["cfg"],
                "{id}"
            );

            let valid = GenerationRequest {
                prompt: "studio portrait".into(),
                width: 1024,
                height: 1024,
                sampler: Some("euler_ancestral".into()),
                scheduler: Some("discrete".into()),
                guidance_method: Some("cfg".into()),
                conditioning: vec![Conditioning::Reference {
                    image: dummy_image(),
                    strength: None,
                }],
                ..Default::default()
            };
            assert!(
                desc.capabilities.validate_request(id, &valid).is_ok(),
                "{id} must accept its implemented euler_ancestral/discrete/cfg controls"
            );

            let mut bad_sampler = valid.clone();
            bad_sampler.sampler = Some("heun".into());
            assert!(
                desc.capabilities
                    .validate_request(id, &bad_sampler)
                    .is_err(),
                "{id} must refuse unimplemented sampler heun"
            );

            let mut bad_scheduler = valid.clone();
            bad_scheduler.scheduler = Some("karras".into());
            assert!(
                desc.capabilities
                    .validate_request(id, &bad_scheduler)
                    .is_err(),
                "{id} must refuse unimplemented scheduler karras"
            );

            let mut bad_guidance = valid;
            bad_guidance.guidance_method = Some("cfg_pp".into());
            assert!(
                desc.capabilities
                    .validate_request(id, &bad_guidance)
                    .is_err(),
                "{id} must refuse unimplemented guidance method cfg_pp"
            );
        }
    }

    #[test]
    fn resolve_strength_prefers_conditioning_then_request_then_default() {
        assert_eq!(resolve_strength(Some(0.35), Some(0.65), 0.8), 0.35);
        assert_eq!(resolve_strength(None, Some(0.65), 0.8), 0.65);
        assert_eq!(resolve_strength(None, None, 0.8), 0.8);
    }
}
