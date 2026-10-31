//! Registered SDXL ControlNet path.
//!
//! The generic candle SDXL generator used to reject every `Conditioning::Control` request even
//! though this crate already carried the packed-aware vendored UNet, stock SDXL `ControlNet` loader,
//! and conditioned curated denoiser for InstantID and tile detail.  This module joins those proven
//! primitives at the registered `sdxl` boundary.  A control overlay is therefore never accepted and
//! ignored: the load spec must contain one base control checkpoint plus zero or more ordered extra
//! checkpoints, and a request must contain exactly the same number of ordered control images.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::sampling::{
    schedule_sigmas, DiscreteModelSampling, SamplerPolicy, Scheduler,
};
use candle_gen::gen_core::{
    self, AcceptedControlKinds, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, ModelDescriptor, Progress, WeightsSource,
};
use candle_gen::{CandleError, Result};

use crate::conditioning::SdxlConditioner;
use crate::denoise::{
    decode_image_with_tiling, denoise_curated, seeded_sigma_prior, text_time_ids, ControlContext,
};
use crate::loaders::{load_instantid_unet_with_adapters, load_sdxl_controlnet, load_sdxl_vae};
use crate::pipeline::{lightning_policy, sdxl_alpha_schedule, SdxlComponents};
use crate::unet::{ControlNet, ControlResiduals, UNet2DConditionModel};
use crate::{descriptor, SdxlArtifactSeal, SdxlVaeDecoder, MODEL_ID, SIZE_MULTIPLE};

const DTYPE: DType = DType::F16;
const DEFAULT_STEPS: usize = 30;
const DEFAULT_GUIDANCE: f64 = 7.0;
const LIGHTNING_DEFAULT_STEPS: usize = 4;
const DEFAULT_CONTROL_SCALE: f32 = 1.0;

/// Resolve the guidance actually used by the selected denoise loop. Lightning is trained and run
/// CFG-free, so an omitted value means `1.0`; ordinary SDXL retains its production default `7.0`.
fn effective_guidance(req: &GenerationRequest) -> f64 {
    if req.sampler.as_deref() == Some("lightning") {
        req.guidance.unwrap_or(1.0) as f64
    } else {
        req.guidance.unwrap_or(DEFAULT_GUIDANCE as f32) as f64
    }
}

/// The loaded, prompt-independent SDXL ControlNet graph.  The CLIP conditioner deliberately stays
/// outside this cache: it is loaded, used, and dropped before the UNet/control/VAE group, preserving
/// the existing staged peak-memory ordering.
#[derive(Clone)]
struct ControlComponents {
    denoise: ControlDenoiseComponents,
    vae: Arc<SdxlVaeDecoder>,
}

#[derive(Clone)]
struct ControlDenoiseComponents {
    unet: Arc<UNet2DConditionModel>,
    controls: Arc<Vec<ControlNet>>,
}

/// The registered generic SDXL ControlNet generator.  It serves ordinary SDXL-family snapshots
/// (SDXL, RealVisXL, RealVisXL Lightning routing when it selects a supported sampler, and
/// Illustrious XL) with a stock diffusers-shaped ControlNet overlay.
pub(super) struct SdxlControlGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    component_paths: SdxlComponents,
    controls: Vec<WeightsSource>,
    adapters: Vec<gen_core::AdapterSpec>,
    file_pin_spec: LoadSpec,
    components: Mutex<Option<ControlComponents>>,
    conditioner: Mutex<Option<Arc<SdxlConditioner>>>,
    request: Mutex<()>,
    memory_contract: Option<gen_core::MemoryProviderContract>,
    artifact_seal: Option<SdxlArtifactSeal>,
}

impl SdxlControlGenerator {
    pub(super) fn new(
        spec: &LoadSpec,
        component_paths: SdxlComponents,
        device: Device,
        artifact_seal: Option<SdxlArtifactSeal>,
    ) -> gen_core::Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(_) => {
                return Err(gen_core::Error::Unsupported(
                    "sdxl: ControlNet requires a diffusers snapshot directory; imported fused checkpoints do not carry the vendored add-embedding graph".into(),
                ));
            }
        };
        if spec.ip_adapter.is_some() {
            return Err(gen_core::Error::Unsupported(
                "sdxl: generic ControlNet does not combine an IP-Adapter overlay; use the dedicated provider".into(),
            ));
        }
        if spec.pid.is_some() {
            return Err(gen_core::Error::Unsupported(
                "sdxl: generic ControlNet does not support a PiD overlay".into(),
            ));
        }

        let mut controls = Vec::with_capacity(1 + spec.extra_controls.len());
        controls.push(
            spec.control
                .clone()
                .expect("control mode is constructed only when LoadSpec::control is present"),
        );
        controls.extend(spec.extra_controls.iter().cloned());

        let memory_contract = artifact_seal.as_ref().map(|seal| seal.contract().clone());
        Ok(Self {
            descriptor: control_descriptor(),
            root,
            device,
            component_paths,
            controls,
            adapters: spec.adapters.clone(),
            file_pin_spec: spec.clone(),
            components: Mutex::new(None),
            conditioner: Mutex::new(None),
            request: Mutex::new(()),
            memory_contract,
            artifact_seal,
        })
    }

    fn components(&self) -> gen_core::Result<ControlComponents> {
        let mut guard = candle_gen::lock_recover(&self.components);
        if let Some(components) = guard.as_ref() {
            return Ok(components.clone());
        }
        let built = ControlComponents {
            denoise: self.load_denoise()?,
            vae: self.load_vae()?,
        };
        *guard = Some(built.clone());
        Ok(built)
    }

    fn load_denoise(&self) -> Result<ControlDenoiseComponents> {
        crate::verify_artifacts(&self.artifact_seal)?;
        self.file_pin_spec.read_prepared_files_unchanged(|| {
            let unet =
                load_instantid_unet_with_adapters(&self.root, &self.device, DTYPE, &self.adapters)?;
            let controls = self
                .controls
                .iter()
                .map(|source| load_sdxl_controlnet(source, &self.device, DTYPE))
                .collect::<Result<Vec<_>>>()?;
            Ok(ControlDenoiseComponents {
                unet: Arc::new(unet),
                controls: Arc::new(controls),
            })
        })
    }

    fn load_vae(&self) -> Result<Arc<SdxlVaeDecoder>> {
        crate::verify_artifacts(&self.artifact_seal)?;
        self.file_pin_spec.read_prepared_files_unchanged(|| {
            let source = self.component_paths.vae_fp16_fix.as_ref().ok_or_else(|| {
                CandleError::Msg(
                    "sdxl: ControlNet snapshot requires the staged fp16-fix VAE".into(),
                )
            })?;
            Ok(Arc::new(load_sdxl_vae(source, &self.device, DTYPE)?))
        })
    }

    fn load_conditioner(&self) -> Result<Arc<SdxlConditioner>> {
        crate::verify_artifacts(&self.artifact_seal)?;
        self.file_pin_spec.read_prepared_files_unchanged(|| {
            Ok(Arc::new(SdxlConditioner::load(
                &self.root,
                &self.device,
                DTYPE,
                &self.component_paths.tokenizer_clip_l,
                &self.component_paths.tokenizer_clip_bigg,
            )?))
        })
    }

    fn denoise_images(
        &self,
        req: &GenerationRequest,
        conditioning: &Tensor,
        pooled: &Tensor,
        components: &ControlDenoiseComponents,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<Tensor>> {
        let lightning = req.sampler.as_deref() == Some("lightning");
        let guidance = effective_guidance(req);
        let cfg_on = !lightning && guidance > 1.0;
        let steps = req.steps.unwrap_or(if lightning {
            LIGHTNING_DEFAULT_STEPS as u32
        } else {
            DEFAULT_STEPS as u32
        }) as usize;
        let controls = self.control_requests(req).map_err(CandleError::from)?;
        let time_ids = text_time_ids(if cfg_on { 2 } else { 1 }, &self.device, DTYPE)?;
        let contexts = controls
            .iter()
            .zip(components.controls.iter())
            .map(|((image, scale), controlnet)| {
                let image =
                    preprocess_generic_control_image(image, req.width, req.height, &self.device)?
                        .to_dtype(DTYPE)?;
                let image = if cfg_on {
                    Tensor::cat(&[&image, &image], 0)?
                } else {
                    image
                };
                Ok(ControlContext {
                    controlnet,
                    cond_embed: controlnet.embed_cond(&image)?,
                    scale: *scale as f64,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let seed = req.seed.unwrap_or_else(gen_core::default_seed);

        let mut images = Vec::with_capacity(req.count as usize);
        crate::with_request_attention_budget(req, || -> Result<()> {
            for index in 0..req.count {
                if req.cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                let image_seed = seed.wrapping_add(index as u64);
                let latents = if lightning {
                    denoise_lightning_control(
                        &components.unet,
                        conditioning,
                        pooled,
                        &time_ids,
                        &contexts,
                        image_seed,
                        req.width,
                        req.height,
                        &self.device,
                        steps,
                        &req.cancel,
                        on_progress,
                        &req.preview,
                    )?
                } else {
                    let schedule = sdxl_alpha_schedule()?;
                    let sampling = DiscreteModelSampling::sdxl(&schedule);
                    let native = schedule_sigmas(Scheduler::Normal, &sampling, steps);
                    let sigmas = candle_gen::resolve_schedule(
                        req.scheduler.as_deref(),
                        &sampling,
                        steps,
                        &native,
                    );
                    let preview = crate::preview::ve_hook(&req.preview);
                    let latents = seeded_sigma_prior(
                        image_seed,
                        req.width,
                        req.height,
                        sigmas[0],
                        &self.device,
                    )?;
                    denoise_curated(
                        &components.unet,
                        req.sampler.as_deref().or(Some("ddim")),
                        &sampling,
                        &sigmas,
                        latents,
                        conditioning,
                        pooled,
                        &time_ids,
                        guidance,
                        DTYPE,
                        image_seed,
                        &req.cancel,
                        on_progress,
                        Some(&preview),
                        &contexts,
                        conditioning,
                    )?
                };
                images.push(latents);
            }
            Ok(())
        })?;
        Ok(images)
    }

    fn decode_images(
        &self,
        req: &GenerationRequest,
        latents: &[Tensor],
        vae: &SdxlVaeDecoder,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<gen_core::Image>> {
        latents
            .iter()
            .map(|latent| {
                on_progress(Progress::Decoding);
                decode_image_with_tiling(
                    vae,
                    latent,
                    None,
                    Some(&req.cancel),
                    crate::denoise::decode_tiling(req.memory),
                )
            })
            .collect()
    }

    fn control_requests<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> gen_core::Result<Vec<(&'a gen_core::Image, f32)>> {
        let controls: Vec<_> = req
            .conditioning
            .iter()
            .filter_map(|conditioning| match conditioning {
                Conditioning::Control { image, scale, .. } => {
                    Some((image, scale.unwrap_or(DEFAULT_CONTROL_SCALE)))
                }
                _ => None,
            })
            .collect();
        if controls.len() != self.controls.len() {
            return Err(gen_core::Error::Msg(format!(
                "sdxl: {} Control conditioning(s) passed but the model was loaded with {} ControlNet checkpoint(s) (set LoadSpec::control + extra_controls, one per Control, in order)",
                controls.len(),
                self.controls.len()
            )));
        }
        Ok(controls)
    }
}

impl Generator for SdxlControlGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_contract.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_contract.as_ref() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        crate::memory_strategy::safety_check(contract, context, self.artifact_seal.as_ref())
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_contract.as_ref() else {
            return Ok(None);
        };
        let seal = self.artifact_seal.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported("sdxl control: authoritative contract has no seal".into())
        })?;
        crate::memory_strategy::validate_context(contract, context, seal)?;
        Ok(Some(Box::new(crate::memory_strategy::request_scope(
            self.device.clone(),
            contract,
            context,
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "sdxl: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "sdxl: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "sdxl: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if req.use_pid {
            return Err(gen_core::Error::Unsupported(
                "sdxl: generic ControlNet does not support the PiD decoder".into(),
            ));
        }
        if req.control_scale.is_some() {
            return Err(gen_core::Error::Unsupported(
                "sdxl: set Control conditioning `scale` per image; request-level control_scale is not part of this ControlNet contract".into(),
            ));
        }
        if req.sampler.as_deref() == Some("lightning") {
            if req.guidance.is_some_and(|guidance| guidance > 1.0) {
                return Err(gen_core::Error::Unsupported(
                    "sdxl: ControlNet with the CFG-free `lightning` sampler requires guidance <= 1.0"
                        .into(),
                ));
            }
            if let Some(scheduler) = req.scheduler.as_deref() {
                let recognized = gen_core::sampling::Scheduler::from_name(scheduler);
                if recognized.is_some_and(|scheduler| scheduler != Scheduler::Normal) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "sdxl: the `lightning` sampler uses its own fixed trailing schedule and ignores the `scheduler` axis (got `{scheduler}`)"
                    )));
                }
            }
        }
        self.control_requests(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if let Some(seal) = &self.artifact_seal {
            seal.ensure_unchanged()?;
        }
        let _request = candle_gen::lock_recover(&self.request);
        let staged = crate::staged_execution(self.file_pin_spec.offload_policy, req.memory);
        let mut effective_req = req.clone();
        effective_req.seed = Some(req.seed.unwrap_or_else(gen_core::default_seed));
        if staged {
            effective_req
                .memory
                .get_or_insert_with(|| gen_core::GenerationMemory {
                    tile_vae_decode: crate::vae_tiling_enabled(),
                    ..Default::default()
                })
                .stage_residency = true;
        }
        let req = &effective_req;
        candle_gen::residency::check_cancel(&req.cancel)?;
        let cfg_on = req.sampler.as_deref() != Some("lightning") && effective_guidance(req) > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        if staged {
            crate::clear_resident_caches(&self.components, &self.conditioner, || {
                Ok(self.device.synchronize()?)
            })?;
        }
        let images = if staged {
            let mut state = (None, Vec::new(), on_progress);
            candle_gen::residency::run_three_stage_sequential(
                &mut state,
                |_| self.load_conditioner(),
                |conditioner, state| {
                    candle_gen::residency::check_cancel(&req.cancel)?;
                    state.0 = Some(candle_gen::gen_core::with_prompt_encoding(state.2, || {
                        conditioner.encode(&req.prompt, negative, cfg_on)
                    })?);
                    Ok(())
                },
                |_| {
                    candle_gen::residency::check_cancel(&req.cancel)?;
                    self.load_denoise()
                },
                |components, state| {
                    let (conditioning, pooled) = state.0.as_ref().ok_or_else(|| {
                        CandleError::Msg("sdxl: missing control text phase".into())
                    })?;
                    state.1 =
                        self.denoise_images(req, conditioning, pooled, components, state.2)?;
                    state.0.take();
                    Ok(())
                },
                |_| {
                    candle_gen::residency::check_cancel(&req.cancel)?;
                    self.load_vae()
                },
                |vae, state| self.decode_images(req, &state.1, vae, state.2),
                || Ok(self.device.synchronize()?),
            )?
        } else {
            let conditioner = candle_gen::cached(&self.conditioner, || self.load_conditioner())?;
            let components = self.components()?;
            let result = (|| -> Result<_> {
                let (conditioning, pooled) =
                    candle_gen::gen_core::with_prompt_encoding(on_progress, || {
                        conditioner.encode(&req.prompt, negative, cfg_on)
                    })?;
                let latents = self.denoise_images(
                    req,
                    &conditioning,
                    &pooled,
                    &components.denoise,
                    on_progress,
                )?;
                self.decode_images(req, &latents, &components.vae, on_progress)
            })();
            candle_gen::residency::synchronize_result(&self.device, result)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Prepare an arbitrary caller-supplied SDXL ControlNet map. Generic ControlNet follows the MLX and
/// diffusers API contract: resize the RGB map to the requested render geometry with Lanczos, then
/// normalize `[0, 255]` to `[0, 1]` NCHW. This is intentionally separate from
/// [`crate::denoise::preprocess_control_image`], whose exact-size rejection belongs to the
/// InstantID kps/OpenPose renderers that always draw at target size.
fn preprocess_generic_control_image(
    image: &gen_core::Image,
    target_width: u32,
    target_height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (source_width, source_height) = (image.width as usize, image.height as usize);
    let expected = gen_core::imageops::checked_image_buffer_len(source_width, source_height, 3)
        .unwrap_or(usize::MAX);
    if image.pixels.len() != expected {
        return Err(CandleError::Msg(format!(
            "sdxl control image pixel buffer {} != {source_width}x{source_height}x3",
            image.pixels.len()
        )));
    }

    let (target_width, target_height) = (target_width as usize, target_height as usize);
    let resized = if (source_width, source_height) == (target_width, target_height) {
        image.pixels.iter().map(|&pixel| pixel as f32).collect()
    } else {
        gen_core::imageops::resize_lanczos_u8(
            &image.pixels,
            source_height,
            source_width,
            target_height,
            target_width,
        )?
    };
    let normalized: Vec<f32> = resized.into_iter().map(|pixel| pixel / 255.0).collect();
    let hwc = Tensor::from_vec(normalized, (target_height, target_width, 3), device)?;
    Ok(hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?)
}

fn control_descriptor() -> ModelDescriptor {
    let mut descriptor = descriptor();
    descriptor.control_kinds = Some(AcceptedControlKinds::Any);
    descriptor.capabilities.conditioning = vec![ConditioningKind::Control];
    descriptor
}

/// The CFG-free RealVisXL-Lightning control path.  The base Lightning loop already establishes the
/// correct trailing-Euler coefficients; this is its ControlNet-shaped twin, retaining the same
/// `[1, ...]` conditioned batch and injecting the summed residuals before every UNet prediction.
#[allow(clippy::too_many_arguments)]
fn denoise_lightning_control(
    unet: &UNet2DConditionModel,
    conditioning: &Tensor,
    pooled: &Tensor,
    time_ids: &Tensor,
    controls: &[ControlContext],
    seed: u64,
    width: u32,
    height: u32,
    device: &Device,
    steps: usize,
    cancel: &gen_core::CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &gen_core::PreviewSink,
) -> Result<Tensor> {
    let policy = lightning_policy(steps)?;
    let mut latents = seeded_sigma_prior(seed, width, height, policy.init_noise_scale(), device)?;
    let preview_counter = candle_gen::preview::PreviewCounter::with_steps(policy.num_steps());
    for index in 0..policy.num_steps() {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let coefficients = policy.coeffs(index);
        candle_gen::preview::emit_preview_at(preview, &preview_counter, index, || {
            crate::preview::project_spatial_latents(&latents.affine(coefficients.c_in as f64, 0.0)?)
        });
        let input = latents
            .affine(coefficients.c_in as f64, 0.0)?
            .to_dtype(DTYPE)?;
        let mut combined: Option<ControlResiduals> = None;
        for control in controls {
            let residuals = control.controlnet.forward(
                &input,
                &control.cond_embed,
                coefficients.timestep as f64,
                conditioning,
                pooled,
                time_ids,
                control.scale,
            )?;
            combined = Some(match combined {
                None => residuals,
                Some(previous) => previous.add(&residuals)?,
            });
        }
        let residuals =
            combined.expect("admission requires one Control conditioning per checkpoint");
        let epsilon = unet.forward_instantid(
            &input,
            coefficients.timestep as f64,
            conditioning,
            pooled,
            time_ids,
            Some(&residuals.down),
            Some(&residuals.mid),
        )?;
        latents = (latents
            + epsilon
                .to_dtype(DType::F32)?
                .affine(coefficients.a_out as f64, 0.0)?)?;
        on_progress(Progress::Step {
            current: index as u32 + 1,
            total: policy.num_steps() as u32,
        });
    }
    Ok(latents.to_dtype(DTYPE)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{ControlKind, Image, Quant};

    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sdxl".into()))
            .with_control(WeightsSource::Dir("/nonexistent/control".into()))
            .with_component(
                "tokenizer_clip_l",
                WeightsSource::File("/nonexistent/clip-l-tokenizer.json".into()),
            )
            .with_component(
                "tokenizer_clip_bigg",
                WeightsSource::File("/nonexistent/clip-g-tokenizer.json".into()),
            )
            .with_component(
                "vae_fp16_fix",
                WeightsSource::File("/nonexistent/vae.safetensors".into()),
            )
    }

    fn control() -> Conditioning {
        Conditioning::Control {
            image: Image {
                width: 512,
                height: 512,
                pixels: vec![0; 512 * 512 * 3],
            },
            kind: ControlKind::Pose,
            scale: None,
        }
    }

    #[test]
    fn control_descriptor_is_truthful_and_includes_lightning() {
        let descriptor = control_descriptor();
        assert_eq!(
            descriptor.capabilities.conditioning,
            vec![ConditioningKind::Control]
        );
        assert_eq!(descriptor.control_kinds, Some(AcceptedControlKinds::Any));
        assert!(descriptor.capabilities.samplers.contains(&"ddim"));
        assert!(descriptor.capabilities.samplers.contains(&"lightning"));
    }

    #[test]
    fn control_admission_requires_exact_ordered_branch_count() {
        let generator = SdxlControlGenerator::new(
            &spec(),
            SdxlComponents::from_spec(&spec(), MODEL_ID).unwrap(),
            Device::Cpu,
            None,
        )
        .unwrap();
        let request = GenerationRequest {
            prompt: "a studio portrait".into(),
            conditioning: vec![control()],
            ..Default::default()
        };
        assert!(generator.validate(&request).is_ok());

        let missing = GenerationRequest {
            prompt: "a studio portrait".into(),
            ..Default::default()
        };
        assert!(generator
            .validate(&missing)
            .unwrap_err()
            .to_string()
            .contains("0 Control conditioning"));

        let mut extra = request.clone();
        extra.conditioning.push(control());
        assert!(generator
            .validate(&extra)
            .unwrap_err()
            .to_string()
            .contains("2 Control conditioning"));
    }

    #[test]
    fn control_admission_rejects_unwired_axes_and_preserves_packed_tiers() {
        let generator = SdxlControlGenerator::new(
            &spec(),
            SdxlComponents::from_spec(&spec(), MODEL_ID).unwrap(),
            Device::Cpu,
            None,
        )
        .unwrap();
        let base = GenerationRequest {
            prompt: "a studio portrait".into(),
            conditioning: vec![control()],
            ..Default::default()
        };

        let mut lightning = base.clone();
        lightning.sampler = Some("lightning".into());
        // The registered Lightning contract is CFG-free by default: omitted guidance resolves to
        // 1.0, not the ordinary SDXL default 7.0, and therefore must validate.
        lightning.guidance = None;
        assert!(generator.validate(&lightning).is_ok());
        assert!((effective_guidance(&lightning) - 1.0).abs() < f64::EPSILON);
        lightning.guidance = Some(1.0);
        assert!(generator.validate(&lightning).is_ok());
        lightning.guidance = Some(7.0);
        assert!(generator.validate(&lightning).is_err());

        let mut scalar = base.clone();
        scalar.control_scale = Some(0.5);
        assert!(generator.validate(&scalar).is_err());

        assert_eq!(
            generator.descriptor().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8],
            "control uses the existing packed-aware vendored UNet loader"
        );
    }

    #[test]
    fn generic_control_resizes_and_normalizes_but_instantid_stays_exact_size() {
        let source = Image {
            width: 2,
            height: 1,
            pixels: vec![255; 2 * 3],
        };
        let resized = preprocess_generic_control_image(&source, 4, 2, &Device::Cpu).unwrap();
        assert_eq!(resized.dims4().unwrap(), (1, 3, 2, 4));
        let values = resized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values.len(), 4 * 2 * 3);
        assert!(
            values
                .iter()
                .all(|value| (*value - 1.0).abs() < f32::EPSILON),
            "an all-white map must remain normalized white after Lanczos resize"
        );

        // The generic helper must not weaken the InstantID-specific kps/OpenPose contract: those
        // callers draw at target size and a mismatch remains an error there.
        assert!(crate::denoise::preprocess_control_image(&source, 4, 2, &Device::Cpu).is_err());
    }
}
