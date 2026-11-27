//! Registered `RouteRecord` and `Generator` wrapper for Candle PuLID-FLUX (`pulid_flux`)
//! (`SW-6982`, `REG-6`, `PIPE-18`).

use candle_gen::gen_core::{
    self, Conditioning, ConditioningKind, GenerationMemory, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, RenderPlan, RouteRecord,
    WeightsSource,
};
use candle_gen_flux::Variant;

use crate::pulid_flux::{
    compile_pulid_plan, PulidFlux, PulidFluxPaths, PulidFluxRequest, DEFAULT_GUIDANCE,
    DEFAULT_ID_WEIGHT, SIZE_MULTIPLE,
};

pub(crate) fn route_pulid_flux() -> RouteRecord {
    let samplers = candle_gen::menu_with_aliases(
        candle_gen::curated_sampler_names(),
        &["flow_match", "flow_match_euler"],
    );
    let schedulers = candle_gen::menu_with_aliases(
        candle_gen::curated_scheduler_names(),
        &["native", "flow_match", "flow_match_euler"],
    );
    RouteRecord {
        route_id: "pulid_flux",
        family: "pulid",
        backend: "candle",
        modality: Modality::Image,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![
                    gen_core::TextEncoderDeclaration::new("clip_l", Some(77)),
                    gen_core::TextEncoderDeclaration::new("t5_xxl", Some(512)),
                ],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::RequiresReference,
                supports_negative_prompt: false,
                modules: vec!["pulid_idformer", "eva_clip", "arcface"],
                ..Default::default()
            },
            sampling: gen_core::SamplingSurface {
                samplers,
                schedulers,
                guidance_methods: Vec::new(),
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: true,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new("flux1_mmdit_pulid")
                    .with_pag(Variant::Dev.pag_capability()),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&gen_core::FLUX1_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                token_width: Some(64),
                ..Default::default()
            },
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::ForwardResidual,
            ),
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::LatentProjection,
                ..Default::default()
            },
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::ConditioningEncode,
                        ["clip_l", "t5_xxl", "eva_clip", "arcface", "pulid_idformer"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["flux1_mmdit", "pulid_ca"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["flux1_ae"],
                        false,
                    ),
                ],
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[],
                requires_identity_overlay: true,
                ..Default::default()
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: SIZE_MULTIPLE,
            max_size: u32::MAX,
            size_floor: gen_core::SizeFloor::RangeCheckedOnGridArea {
                multiple: SIZE_MULTIPLE,
                max_cells: u32::MAX,
            },
            ..Default::default()
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(25),
            guidance: Some(DEFAULT_GUIDANCE),
            strength: Some(DEFAULT_ID_WEIGHT),
            ..Default::default()
        },
        bindings: {
            #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
            let mut bindings = gen_core::RouteBindings::generator(descriptor, load_pulid_flux)
                .with_render_plan(compile_pulid_render_plan);
            #[cfg(feature = "cuda")]
            {
                bindings = bindings
                    .with_memory_registration(crate::memory_strategy::PULID_MEMORY_REGISTRATION)
                    .with_memory_fixture(
                        crate::memory_strategy::memory_contract_surface_specs,
                        crate::memory_strategy::registered_provider_contract,
                    )
                    .with_memory_behavior_registration(
                        crate::memory_strategy::PULID_MEMORY_BEHAVIOR,
                    );
            }
            bindings
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    route_pulid_flux().compute_descriptor()
}

pub(crate) fn paths_from_spec(spec: &LoadSpec) -> gen_core::Result<PulidFluxPaths> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], "pulid_flux")?;
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "pulid_flux does not accept control, IP-Adapter, or text-encoder overlays".into(),
        ));
    }
    let flux_base = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "pulid_flux requires a FLUX.1-dev snapshot directory".into(),
            ));
        }
    };
    let identity = spec.identity.as_ref().ok_or_else(|| {
        gen_core::Error::Unsupported(
            "pulid_flux requires LoadSpec::identity with encoder, eva, and face_dir".into(),
        )
    })?;
    let pulid_weights = match &identity.encoder {
        Some(WeightsSource::File(p) | WeightsSource::Dir(p)) => p.clone(),
        None => {
            return Err(gen_core::Error::Unsupported(
                "pulid_flux requires LoadSpec::identity.encoder".into(),
            ));
        }
    };
    let eva_weights = match &identity.eva {
        Some(WeightsSource::File(p) | WeightsSource::Dir(p)) => p.clone(),
        None => {
            return Err(gen_core::Error::Unsupported(
                "pulid_flux requires LoadSpec::identity.eva".into(),
            ));
        }
    };
    let face_dir = match &identity.face_dir {
        Some(WeightsSource::Dir(p) | WeightsSource::File(p)) => p.clone(),
        None => {
            return Err(gen_core::Error::Unsupported(
                "pulid_flux requires LoadSpec::identity.face_dir".into(),
            ));
        }
    };
    Ok(PulidFluxPaths {
        flux_base,
        pulid_weights,
        eva_weights,
        face_dir,
        adapters: spec.adapters.clone(),
    })
}

fn pulid_req_and_ref(
    req: &GenerationRequest,
) -> gen_core::Result<(PulidFluxRequest, &gen_core::Image)> {
    let (reference, strength) = match req.conditioning.as_slice() {
        [Conditioning::Reference { image, strength }] => (image, *strength),
        _ => {
            return Err(gen_core::Error::Unsupported(
                "pulid_flux requires exactly one Reference face conditioning image".into(),
            ));
        }
    };
    let pulid_req = PulidFluxRequest {
        prompt: req.prompt.clone(),
        width: req.width,
        height: req.height,
        steps: req.steps.map(|s| s as usize).unwrap_or(25),
        guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE),
        id_weight: strength.unwrap_or(DEFAULT_ID_WEIGHT),
        sampler: req.sampler.as_deref().map(str::to_owned),
        scheduler: req.scheduler.as_deref().map(str::to_owned),
        scheduler_shift: req.scheduler_shift,
        seed: req.seed.unwrap_or(0),
        use_pid: req.use_pid(),
        preview: req.preview.clone(),
        pag: req.features.pag.clone(),
        cancel: req.cancel.clone(),
    };
    Ok((pulid_req, reference))
}

pub fn compile_pulid_render_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<RenderPlan>> {
    let desc = descriptor();
    desc.capabilities.validate_request(desc.id, req)?;
    let paths = paths_from_spec(spec)?;
    let (pulid_req, reference) = pulid_req_and_ref(req)?;
    compile_pulid_plan(&paths, spec, &pulid_req, reference).map(Some)
}

struct PulidFluxGenerator {
    descriptor: ModelDescriptor,
    paths: PulidFluxPaths,
    spec: LoadSpec,
    model: PulidFlux,
}

impl Generator for PulidFluxGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let (pulid_req, reference) = pulid_req_and_ref(req)?;
        let plan = compile_pulid_plan(&self.paths, &self.spec, &pulid_req, reference)?;
        self.model
            .generate_with_execution_report(&pulid_req, reference, &plan, on_progress)
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        let _ = compile_pulid_render_plan(&self.spec, request)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_execution_report(request, on_progress)?
            .generation
            .output
            .ok_or_else(|| gen_core::Error::Msg("pulid_flux: generation produced no output".into()))
    }
}

pub fn load_pulid_flux(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let paths = paths_from_spec(spec)?;
    let mut model = PulidFlux::load_with_memory_spec(&paths, GenerationMemory::default(), spec)
        .map_err(gen_core::Error::from)?;
    if let Some(pid) = &spec.pid {
        model = model.with_pid(pid).map_err(gen_core::Error::from)?;
    }
    Ok(Box::new(PulidFluxGenerator {
        descriptor: descriptor(),
        paths,
        spec: spec.clone(),
        model,
    }))
}
