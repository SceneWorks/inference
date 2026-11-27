//! Registered `RouteRecord` and `Generator` wrapper for MLX InstantID (`instantid`)
//! (`SW-6982`, `REG-6`, `PIPE-18`).

use std::cell::RefCell;

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, gen_core, Conditioning, ConditioningKind,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Progress,
    WeightsSource,
};

use crate::model::{
    InstantId, InstantIdPaths, InstantIdRequest, DEFAULT_CONTROLNET_SCALE, DEFAULT_IP_SCALE,
    SIZE_MULTIPLE,
};

pub(crate) fn route_instantid() -> gen_core::RouteRecord {
    let mut samplers = curated_sampler_names();
    if !samplers.contains(&"euler_ancestral") {
        samplers.push("euler_ancestral");
    }
    let schedulers = curated_scheduler_names();

    gen_core::RouteRecord {
        route_id: "instantid",
        family: "instantid",
        backend: "mlx",
        modality: Modality::Image,
        mac_only: true,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![
                    gen_core::TextEncoderDeclaration::new("clip_l", Some(77)),
                    gen_core::TextEncoderDeclaration::new("clip_bigg", Some(77)),
                ],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::RequiresReference,
                supports_negative_prompt: true,
                modules: vec!["instantid_resampler", "instantid_identitynet", "arcface"],
                ..Default::default()
            },
            sampling: gen_core::SamplingSurface {
                samplers,
                schedulers,
                guidance_methods: Vec::new(),
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(gen_core::FamilyModelSpec::new(
                "sdxl_unet_instantid",
            )),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("sdxl_vae"),
                output_latent_space: Some(&gen_core::SDXL_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                ..Default::default()
            },
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::BaseDenoiserOnly,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::ConditioningEncode,
                        ["clip_l", "clip_bigg", "arcface", "instantid_resampler"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["sdxl_unet", "instantid_identitynet"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["sdxl_vae"],
                        false,
                    ),
                ],
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                requires_identity_overlay: true,
                requires_control_overlay: true,
                ..Default::default()
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 256,
            max_size: 2048,
            size_floor: gen_core::SizeFloor::RangeCheckedOnGridArea {
                multiple: SIZE_MULTIPLE,
                max_cells: u32::MAX,
            },
            ..Default::default()
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 4,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(30),
            guidance: Some(5.0),
            strength: Some(DEFAULT_IP_SCALE),
            ..Default::default()
        },
        bindings: gen_core::RouteBindings::generator(descriptor, load_instantid),
    }
}

pub fn descriptor() -> ModelDescriptor {
    route_instantid().compute_descriptor()
}

struct InstantIdGenerator {
    descriptor: ModelDescriptor,
    adapter_count: usize,
    model: RefCell<InstantId>,
}

impl Generator for InstantIdGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        let mut progress = |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
        mlx_gen::image_execution::complete_image_attempt(&self.descriptor, &req.cancel, || {
            mlx_gen::image_execution::choice(
                "instantid.route",
                gen_core::generator::ExecutionFact::resolved(self.descriptor.id),
            );
            if self.adapter_count != 0 {
                mlx_gen::image_execution::unavailable(
                    "instantid_adapter_application_unobserved",
                    "loaded SDXL adapters have no per-target application report on this route",
                    "loaded_adapter_application",
                );
            }
            self.generate(req, &mut progress)
        })
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, request)?;
        if request.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "instantid: prompt must not be empty".into(),
            ));
        }
        if !request.width.is_multiple_of(SIZE_MULTIPLE)
            || !request.height.is_multiple_of(SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "instantid: width/height must be multiples of {SIZE_MULTIPLE}"
            )));
        }
        match request.conditioning.as_slice() {
            [Conditioning::Reference { .. }] => Ok(()),
            _ => Err(gen_core::Error::Unsupported(
                "instantid requires exactly one Reference face conditioning image".into(),
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
        let defaults = InstantIdRequest::default();
        let steps = request.steps.map(|s| s as usize).unwrap_or(defaults.steps);
        let guidance = request.guidance.unwrap_or(defaults.guidance);
        let ip_adapter_scale = strength.unwrap_or(DEFAULT_IP_SCALE);
        let controlnet_scale = request.control_scale.unwrap_or(DEFAULT_CONTROLNET_SCALE);
        let base_seed = request.seed.unwrap_or(0);
        let mut model = self.model.borrow_mut();
        let mut images = Vec::with_capacity(request.count as usize);
        for idx in 0..request.count {
            let id_req = InstantIdRequest {
                prompt: request.prompt.clone(),
                negative: request.negative_prompt.clone().unwrap_or_default(),
                width: request.width,
                height: request.height,
                steps,
                guidance,
                ip_adapter_scale,
                controlnet_scale,
                openpose_scale: defaults.openpose_scale,
                sampler: request.sampler.as_deref().map(str::to_owned),
                scheduler: request.scheduler.as_deref().map(str::to_owned),
                seed: base_seed.wrapping_add(idx as u64),
                use_pid: request.use_pid(),
                cancel: request.cancel.clone(),
            };
            images.push(
                model
                    .generate(&id_req, reference, on_progress)
                    .map_err(gen_core::Error::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn resolve_instantid_paths(spec: &LoadSpec) -> gen_core::Result<(InstantIdPaths, std::path::PathBuf)> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], "instantid")?;
    if !spec.extra_controls.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "instantid generator route does not support extra_controls (OpenPose pose conditioning is not wired on this route)"
                .into(),
        ));
    }
    let sdxl_base = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "instantid requires an SDXL snapshot directory".into(),
            ));
        }
    };
    let identitynet = spec.control.clone().ok_or_else(|| {
        gen_core::Error::Unsupported(
            "instantid requires spec.control (IdentityNet ControlNet)".into(),
        )
    })?;
    let ip_adapter = match spec
        .ip_adapter
        .as_ref()
        .or_else(|| spec.identity.as_ref().and_then(|id| id.encoder.as_ref()))
    {
        Some(WeightsSource::File(p) | WeightsSource::Dir(p)) => p.clone(),
        None => {
            return Err(gen_core::Error::Unsupported(
                "instantid requires spec.ip_adapter or spec.identity.encoder".into(),
            ));
        }
    };
    let face_dir = match spec.identity.as_ref().and_then(|id| id.face_dir.as_ref()) {
        Some(WeightsSource::Dir(p) | WeightsSource::File(p)) => p.clone(),
        None => {
            return Err(gen_core::Error::Msg(
                "instantid requires spec.identity.face_dir (face analysis weights directory)".into(),
            ));
        }
    };
    let paths = InstantIdPaths {
        sdxl_base,
        identitynet,
        ip_adapter,
        adapters: spec.adapters.clone(),
    };
    Ok((paths, face_dir))
}

pub fn load_instantid(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let (paths, face_dir) = resolve_instantid_paths(spec)?;
    let adapter_count = paths.adapters.len();
    let mut model = InstantId::load(&paths).map_err(gen_core::Error::from)?;
    model = model
        .with_face_paths(
            &face_dir.join("scrfd_10g.safetensors"),
            &face_dir.join("arcface_iresnet100.safetensors"),
        )
        .map_err(gen_core::Error::from)?;
    if let Some(pid) = &spec.pid {
        model = model.with_pid(pid).map_err(gen_core::Error::from)?;
    }
    Ok(Box::new(InstantIdGenerator {
        descriptor: descriptor(),
        adapter_count,
        model: RefCell::new(model),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::IdentityWeights;

    fn base_spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/tmp/sdxl".into()))
            .with_control(WeightsSource::Dir("/tmp/identitynet".into()))
            .with_ip_adapter(WeightsSource::File("/tmp/ip-adapter.safetensors".into()));
        spec.identity = Some(IdentityWeights {
            encoder: None,
            eva: None,
            face_dir: Some(WeightsSource::Dir("/tmp/face".into())),
        });
        spec
    }

    #[test]
    fn load_instantid_rejects_missing_face_dir_before_loading_weights() {
        let mut no_identity = base_spec();
        no_identity.identity = None;
        let err = load_instantid(&no_identity).err().expect("must fail closed");
        assert!(
            err.to_string().contains("face_dir"),
            "expected face_dir diagnostic, got: {err}"
        );

        let mut no_face_dir = base_spec();
        no_face_dir.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File("/tmp/ip-adapter.safetensors".into())),
            eva: None,
            face_dir: None,
        });
        let err = load_instantid(&no_face_dir).err().expect("must fail closed");
        assert!(
            err.to_string().contains("face_dir"),
            "expected face_dir diagnostic, got: {err}"
        );
    }

    #[test]
    fn load_instantid_rejects_non_empty_extra_controls() {
        let mut spec = base_spec();
        spec.extra_controls
            .push(WeightsSource::Dir("/tmp/openpose".into()));
        let err = load_instantid(&spec).err().expect("must reject extra_controls");
        assert!(
            matches!(err, gen_core::Error::Unsupported(_))
                && err.to_string().contains("extra_controls"),
            "expected Unsupported extra_controls error, got: {err:?}"
        );
    }
}
