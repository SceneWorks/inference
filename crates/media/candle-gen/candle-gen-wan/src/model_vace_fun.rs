//! Candle `wan2_2_vace_fun_14b`: two complete VACE experts over the shared z16 VAE/UMT5 control
//! path. `transformer/` is the high-noise expert, `transformer_2/` the low-noise expert; the scheduler
//! crosses once at `0.875 * 1000`, preserving its multistep history across the swap.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadPhase, LoadSpec, Modality, ModelDescriptor, MoeExpert,
    OffloadPolicy, PerComponentBytes, Progress, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use crate::config::{
    TextEncoderConfig, Vae16Config, WanVaceConfig, DEFAULT_FPS_VACE, DEFAULT_STEPS_14B,
    MAX_AREA_14B, NEGATIVE_FALLBACK, SIZE_MULTIPLE_14B, T2V_14B_FLOW_SHIFT, T2V_14B_GUIDANCE_HIGH,
    T2V_14B_GUIDANCE_LOW, VAE16_STRIDE_TEMPORAL,
};
use crate::pipeline::{create_noise, frames_to_images, preprocess_ti2v_image};
use crate::rope::WanRope;
use crate::scheduler::{FlowScheduler, Sampler};
use crate::text_encoder::Umt5Encoder;
use crate::vace::{
    build_vace_control, crossing_index, denoise_vace_range, prepare_masks, prepare_video_latents,
    vace_control_scales, WanVaceTransformer,
};
use crate::vace_fun_tier::VaceFunTierPaths;
use crate::vae16::WanVae16;
use crate::wan14b::staged_expert_swap;

pub const MODEL_ID_VACE_FUN: &str = "wan2_2_vace_fun_14b";
pub type ProviderVae = WanVae16;

/// Resolve the VACE-Fun route's load-bearing z16 VAE geometry.
pub fn vae_tiling(provider_id: &str) -> Option<candle_gen::gen_core::tiling::VaeTiling> {
    (provider_id == MODEL_ID_VACE_FUN).then_some(ProviderVae::VAE_TILING)
}

const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 16;
const VAE_T: usize = VAE16_STRIDE_TEMPORAL as usize;
const BOUNDARY: f64 = 0.875;
const NUM_TRAIN_TIMESTEPS: f64 = 1000.0;

fn expert_schedule_crossing(steps: usize, sampler: Sampler, shift: f64) -> gen_core::Result<usize> {
    if !shift.is_finite() || shift <= 0.0 {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID_VACE_FUN}: scheduler_shift must be finite and positive (got {shift})"
        )));
    }
    let scheduler = FlowScheduler::new(sampler, steps, shift);
    let timesteps = (0..steps)
        .map(|step| scheduler.timestep(step))
        .collect::<Vec<_>>();
    if timesteps.iter().any(|timestep| !timestep.is_finite())
        || timesteps.windows(2).any(|pair| pair[0] < pair[1])
    {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID_VACE_FUN}: schedule must contain finite, monotonically decreasing timesteps"
        )));
    }
    let crossing = crossing_index(&timesteps, BOUNDARY * NUM_TRAIN_TIMESTEPS);
    if crossing == 0 || crossing == steps {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID_VACE_FUN}: schedule did not cross boundary {BOUNDARY} exactly once \
             (crossing {crossing}/{steps})"
        )));
    }
    Ok(crossing)
}

#[derive(Clone)]
struct SharedComponents {
    te: Arc<Umt5Encoder>,
    vae: Arc<ProviderVae>,
    tok: Arc<candle_gen::gen_core::tokenizer::TextTokenizer>,
}

#[derive(Clone)]
struct ExpertComponents {
    high: Arc<WanVaceTransformer>,
    low: Arc<WanVaceTransformer>,
}

#[allow(clippy::unnecessary_map_or)] // `Option::is_none_or` is newer than the repository MSRV.
fn selected_adapters(adapters: &[AdapterSpec], expert: MoeExpert) -> Vec<AdapterSpec> {
    adapters
        .iter()
        .filter(|adapter| adapter.moe_expert.map_or(true, |target| target == expert))
        .cloned()
        .collect()
}

struct Pipeline {
    root: PathBuf,
    tier: Option<VaceFunTierPaths>,
    device: Device,
    text_cfg: TextEncoderConfig,
    vae_cfg: Vae16Config,
    vace_cfg: WanVaceConfig,
    adapters: Vec<AdapterSpec>,
}

impl Pipeline {
    fn new(
        root: &Path,
        device: &Device,
        adapters: Vec<AdapterSpec>,
        tier: Option<VaceFunTierPaths>,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            tier,
            device: device.clone(),
            text_cfg: TextEncoderConfig::umt5_xxl(),
            vae_cfg: Vae16Config::wan21(),
            vace_cfg: WanVaceConfig::vace_14b(),
            adapters,
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        if let Some(tier) = &self.tier {
            if matches!(sub, "transformer" | "transformer_2") {
                return tier.component_vb(sub, dtype, &self.device);
            }
            return crate::text_encode::component_vb(
                self.shared_root(),
                sub,
                dtype,
                &self.device,
                MODEL_ID_VACE_FUN,
                "Wan2.2 VACE-Fun A14B split packed tier",
            );
        }
        crate::text_encode::component_vb(
            &self.root,
            sub,
            dtype,
            &self.device,
            MODEL_ID_VACE_FUN,
            "Wan2.2 VACE-Fun A14B dual-expert diffusers",
        )
    }

    /// The snapshot root that owns the dense shared components. A hosted split tier carries only
    /// the two packed experts; its text encoder, VAE, and tokenizer are siblings at this root.
    fn shared_root(&self) -> &Path {
        self.tier
            .as_ref()
            .map(VaceFunTierPaths::shared_root)
            .unwrap_or(&self.root)
    }

    fn build_tokenizer(&self) -> CResult<candle_gen::gen_core::tokenizer::TextTokenizer> {
        crate::text_encode::build_umt5_tokenizer(
            self.shared_root(),
            &self.text_cfg,
            MODEL_ID_VACE_FUN,
        )
    }

    fn load_shared(&self) -> CResult<SharedComponents> {
        let te = Umt5Encoder::new(
            &self.text_cfg,
            self.component_vb("text_encoder", ENC_DTYPE)?,
        )?;
        let vae = WanVae16::new_with_encoder(&self.vae_cfg, self.component_vb("vae", VAE_DTYPE)?)?;
        let tok = self.build_tokenizer()?;
        Ok(SharedComponents {
            te: Arc::new(te),
            vae: Arc::new(vae),
            tok: Arc::new(tok),
        })
    }

    fn load_decode_vae(&self) -> CResult<WanVae16> {
        Ok(WanVae16::new(
            &self.vae_cfg,
            self.component_vb("vae", VAE_DTYPE)?,
        )?)
    }

    fn build_expert(&self, expert: MoeExpert) -> CResult<WanVaceTransformer> {
        let sub = match expert {
            MoeExpert::High => "transformer",
            MoeExpert::Low => "transformer_2",
        };
        let adapters = selected_adapters(&self.adapters, expert);
        let vb = self.component_vb(sub, DIT_DTYPE)?;
        if vb.contains_tensor("blocks.0.attn1.to_q.scales") && !adapters.is_empty() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID_VACE_FUN}: adapters on a packed VACE-Fun tier are not implemented; \
                 refusing to materialize dense weights for the {expert:?} expert"
            )));
        }
        if adapters.is_empty() {
            return Ok(WanVaceTransformer::new(&self.vace_cfg, vb)?);
        }
        drop(vb);
        let mut map = crate::text_encode::load_component_map(&self.root, sub, MODEL_ID_VACE_FUN)?;
        // `merge_adapters` fails closed per selected file if it matches no compatible projection.
        crate::adapters::merge_adapters(&mut map, &adapters)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok(WanVaceTransformer::new(&self.vace_cfg, vb)?)
    }

    fn load_experts(&self) -> CResult<ExpertComponents> {
        Ok(ExpertComponents {
            high: Arc::new(self.build_expert(MoeExpert::High)?),
            low: Arc::new(self.build_expert(MoeExpert::Low)?),
        })
    }

    fn encode_text(&self, shared: &SharedComponents, prompt: &str) -> CResult<Tensor> {
        crate::text_encode::umt5_encode_padded(
            &shared.tok,
            &self.text_cfg,
            &shared.te,
            prompt,
            &self.device,
            ENC_DTYPE,
            MODEL_ID_VACE_FUN,
        )
    }

    fn preprocess_clip(&self, frames: &[Image], width: u32, height: u32) -> CResult<Tensor> {
        if frames.is_empty() {
            return Err(CandleError::Msg(
                "wan VACE-Fun control clip has no frames".into(),
            ));
        }
        let frames = frames
            .iter()
            .map(|image| preprocess_ti2v_image(image, width, height, &self.device))
            .collect::<CResult<Vec<_>>>()?;
        Ok(Tensor::cat(&frames.iter().collect::<Vec<_>>(), 2)?)
    }

    fn prepare(&self, req: &GenerationRequest, shared: &SharedComponents) -> CResult<Prepared> {
        let clip = req
            .control_clip()
            .ok_or_else(|| CandleError::Msg("wan VACE-Fun requires a ControlClip".into()))?;
        let width = req.width;
        let height = req.height;
        let video = self.preprocess_clip(clip.frames, width, height)?;
        let mask = ((self.preprocess_clip(clip.mask, width, height)? + 1.0)? * 0.5)?;
        let references = req
            .conditioning
            .iter()
            .filter_map(|conditioning| match conditioning {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .map(|image| preprocess_ti2v_image(image, width, height, &self.device))
            .collect::<CResult<Vec<_>>>()?;
        let num_ref = references.len();
        let video_latents = prepare_video_latents(&shared.vae, &video, &mask, &references)?;
        let mask_latents = prepare_masks(&mask, self.vace_cfg.base.patch.1, num_ref)?;
        let control = build_vace_control(&video_latents, &mask_latents)?;
        let (_, _, t, h, w) = control.dims5()?;
        let (pt, ph, pw) = self.vace_cfg.base.patch;
        let (cos, sin) =
            WanRope::new(&self.vace_cfg.base).cos_sin(t / pt, h / ph, w / pw, &self.device)?;
        let pos = self.encode_text(shared, &req.prompt)?;
        // VACE-Fun inherits the Wan2.2 T2V-A14B MoE defaults: low/high CFG 3/4. A scalar request
        // override applies to both experts, matching mlx-gen-wan.
        let (guidance_low, guidance_high) = match req.guidance {
            Some(guidance) => (guidance as f64, guidance as f64),
            None => (T2V_14B_GUIDANCE_LOW as f64, T2V_14B_GUIDANCE_HIGH as f64),
        };
        let neg = if guidance_low > 1.0 || guidance_high > 1.0 {
            Some(self.encode_text(
                shared,
                req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK),
            )?)
        } else {
            None
        };
        let steps = req.steps.unwrap_or(DEFAULT_STEPS_14B) as usize;
        let shift = req
            .scheduler_shift
            .map(f64::from)
            .unwrap_or(T2V_14B_FLOW_SHIFT);
        let sampler = Sampler::parse(req.sampler.as_deref());
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let latents = create_noise(seed, Z_DIM, t, h, w, &self.device)?;
        Ok(Prepared {
            control,
            // VACE exposes one conditioning scale for the whole hint stack. Multiplying it by the
            // requested masking strength weights the mask/video control without thresholding a soft
            // mask away in `prepare_video_latents`. Resolved by the shared `vace.rs` seam so this
            // route and the single-expert one cannot drift; pinned by
            // `render_binds_scales_to_the_shared_resolver`.
            scales: vace_control_scales(req, self.vace_cfg.vace_layers.len()),
            pos,
            neg,
            cos,
            sin,
            latents,
            guidance_low,
            guidance_high,
            steps,
            shift,
            sampler,
            num_ref,
            fps: req.fps.unwrap_or(DEFAULT_FPS_VACE),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn use_expert(
        &self,
        expert: &WanVaceTransformer,
        prepared: &mut Prepared,
        scheduler: &mut FlowScheduler,
        timesteps: &[f64],
        range: std::ops::Range<usize>,
        guidance: f64,
        cancel: &candle_gen::gen_core::runtime::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<()> {
        let pos = expert.embed_text(&prepared.pos)?;
        let neg = prepared
            .neg
            .as_ref()
            .map(|negative| expert.embed_text(negative))
            .transpose()?;
        let total = prepared.steps as u32;
        let mut on_step = |step: usize| {
            on_progress(Progress::Step {
                current: step as u32,
                total,
            })
        };
        denoise_vace_range(
            expert,
            &prepared.control,
            &prepared.scales,
            scheduler,
            guidance,
            &pos,
            neg.as_ref(),
            &mut prepared.latents,
            timesteps,
            range,
            &prepared.cos,
            &prepared.sin,
            cancel,
            &mut on_step,
        )
    }

    fn finish(
        &self,
        prepared: Prepared,
        vae: &WanVae16,
        cancel: &candle_gen::gen_core::runtime::CancelFlag,
    ) -> CResult<(Vec<Image>, u32)> {
        let t = prepared.latents.dim(2)?;
        let latents = if prepared.num_ref > 0 {
            prepared
                .latents
                .narrow(2, prepared.num_ref, t - prepared.num_ref)?
        } else {
            prepared.latents
        };
        let decoded = vae.decode_budgeted_with_cancel(&latents, cancel)?;
        Ok((frames_to_images(&decoded)?, prepared.fps))
    }
}

struct Prepared {
    control: Tensor,
    scales: Vec<f32>,
    pos: Tensor,
    neg: Option<Tensor>,
    cos: Tensor,
    sin: Tensor,
    latents: Tensor,
    guidance_low: f64,
    guidance_high: f64,
    steps: usize,
    shift: f64,
    sampler: Sampler,
    num_ref: usize,
    fps: u32,
}

pub struct WanVaceFunGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    adapters: Vec<AdapterSpec>,
    offload: OffloadPolicy,
    tier: Option<VaceFunTierPaths>,
    shared: Mutex<Option<SharedComponents>>,
    experts: Mutex<Option<ExpertComponents>>,
    i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
}

impl WanVaceFunGenerator {
    fn shared(&self, pipeline: &Pipeline) -> gen_core::Result<SharedComponents> {
        Ok(candle_gen::cached(&self.shared, || pipeline.load_shared())?)
    }

    fn experts(&self, pipeline: &Pipeline) -> gen_core::Result<ExpertComponents> {
        Ok(candle_gen::cached(&self.experts, || {
            pipeline.load_experts()
        })?)
    }

    fn validate_request(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID_VACE_FUN, req)?;
        if req.prompt.trim().is_empty() || req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "wan VACE-Fun requires a prompt and at least one step".into(),
            ));
        }
        let steps = req.steps.unwrap_or(DEFAULT_STEPS_14B) as usize;
        let shift = req
            .scheduler_shift
            .map(f64::from)
            .unwrap_or(T2V_14B_FLOW_SHIFT);
        expert_schedule_crossing(steps, Sampler::parse(req.sampler.as_deref()), shift)?;
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "wan VACE-Fun width/height must be multiples of {SIZE_MULTIPLE_14B}"
            )));
        }
        if req.width as usize * req.height as usize > MAX_AREA_14B {
            return Err(gen_core::Error::Msg(format!(
                "wan VACE-Fun geometry exceeds the {MAX_AREA_14B} px area cap"
            )));
        }
        let control_clip_count = req
            .conditioning
            .iter()
            .filter(|conditioning| matches!(conditioning, Conditioning::ControlClip { .. }))
            .count();
        if control_clip_count != 1 {
            return Err(gen_core::Error::Msg(
                "wan VACE-Fun requires exactly one ControlClip; additional clips cannot be applied"
                    .into(),
            ));
        }
        let clip = req
            .control_clip()
            .ok_or_else(|| gen_core::Error::Msg("wan VACE-Fun requires a ControlClip".into()))?;
        if !clip.masking_strength.is_finite() || !(0.0..=1.0).contains(&clip.masking_strength) {
            return Err(gen_core::Error::Msg(format!(
                "wan VACE-Fun masking_strength must be finite and in [0,1] (got {})",
                clip.masking_strength
            )));
        }
        if clip.start_frame != 0 {
            return Err(gen_core::Error::Unsupported(
                "wan VACE-Fun currently applies ControlClip only at start_frame=0".into(),
            ));
        }
        // `mode` is already realized in the worker-rasterized control mask; all replacement modes
        // therefore use the same VACE mask path here.
        if clip.frames.len() != clip.mask.len()
            || clip.frames.is_empty()
            || clip.frames.len() % VAE_T != 1
            || clip.frames.len() > crate::MAX_WAN_FRAMES
        {
            return Err(gen_core::Error::Msg(
                "wan VACE-Fun control and mask must have equal 1+4·k frame counts within the Wan limit"
                    .into(),
            ));
        }
        if req
            .frames
            .is_some_and(|frames| frames as usize != clip.frames.len())
        {
            return Err(gen_core::Error::Msg(
                "wan VACE-Fun req.frames must match the ControlClip length".into(),
            ));
        }
        if req.strength.is_some() {
            return Err(gen_core::Error::Unsupported(
                "wan VACE-Fun does not support top-level image strength".into(),
            ));
        }
        let mut references = 0usize;
        for conditioning in &req.conditioning {
            if let Conditioning::Reference { strength, .. } = conditioning {
                if strength.is_some() {
                    return Err(gen_core::Error::Unsupported(
                        "wan VACE-Fun does not support explicit Reference strength".into(),
                    ));
                }
                references += 1;
            }
        }
        let combined = crate::combined_conditioning_latents(clip.frames.len(), references)
            .ok_or_else(|| {
                gen_core::Error::Msg("wan VACE-Fun conditioning size overflow".into())
            })?;
        if combined > crate::MAX_WAN_CONDITIONING_LATENTS {
            return Err(gen_core::Error::Msg(
                "wan VACE-Fun combined control/reference latent budget exceeded".into(),
            ));
        }
        Ok(())
    }
}

impl Generator for WanVaceFunGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_request(req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate_request(req)?;
        if let Some(prepared) = &self.i2v_memory {
            crate::i2v_memory_strategy::validate_active_request(prepared, req)?;
        }
        let pipeline = Pipeline::new(
            &self.root,
            &self.device,
            self.adapters.clone(),
            self.tier.clone(),
        );
        // Sequential follows Wan14B's staged residency: the heavy UMT5 + encoder VAE are local to
        // control preparation and drop before either expert loads. Resident keeps the shared cache.
        let (mut prepared, resident_shared) = match self.offload {
            OffloadPolicy::Resident => {
                let shared = self.shared(&pipeline)?;
                let prepared = pipeline.prepare(req, &shared)?;
                (prepared, Some(shared))
            }
            OffloadPolicy::Sequential => {
                let shared = pipeline.load_shared()?;
                let prepared = pipeline.prepare(req, &shared)?;
                self.device.synchronize().map_err(CandleError::from)?;
                drop(shared);
                (prepared, None)
            }
        };
        let mut scheduler = FlowScheduler::new(prepared.sampler, prepared.steps, prepared.shift);
        let crossing = expert_schedule_crossing(prepared.steps, prepared.sampler, prepared.shift)?;
        let timesteps = (0..prepared.steps)
            .map(|step| scheduler.timestep(step))
            .collect::<Vec<_>>();
        match self.offload {
            OffloadPolicy::Resident => {
                let experts = self.experts(&pipeline)?;
                let steps = prepared.steps;
                let guidance_high = prepared.guidance_high;
                let guidance_low = prepared.guidance_low;
                pipeline.use_expert(
                    &experts.high,
                    &mut prepared,
                    &mut scheduler,
                    &timesteps,
                    0..crossing,
                    guidance_high,
                    &req.cancel,
                    on_progress,
                )?;
                pipeline.use_expert(
                    &experts.low,
                    &mut prepared,
                    &mut scheduler,
                    &timesteps,
                    crossing..steps,
                    guidance_low,
                    &req.cancel,
                    on_progress,
                )?;
            }
            OffloadPolicy::Sequential => {
                struct SwapState<'a> {
                    prepared: &'a mut Prepared,
                    scheduler: &'a mut FlowScheduler,
                    on_progress: &'a mut dyn FnMut(Progress),
                }
                let steps = prepared.steps;
                let mut state = SwapState {
                    prepared: &mut prepared,
                    scheduler: &mut scheduler,
                    on_progress: &mut *on_progress,
                };
                staged_expert_swap(
                    crossing,
                    steps,
                    &mut state,
                    |state| {
                        (state.on_progress)(Progress::Loading(LoadPhase::Renderer));
                        pipeline.build_expert(MoeExpert::High)
                    },
                    |high, state| {
                        pipeline.use_expert(
                            high,
                            state.prepared,
                            state.scheduler,
                            &timesteps,
                            0..crossing,
                            state.prepared.guidance_high,
                            &req.cancel,
                            state.on_progress,
                        )
                    },
                    |state| {
                        (state.on_progress)(Progress::Loading(LoadPhase::Renderer));
                        pipeline.build_expert(MoeExpert::Low)
                    },
                    |low, state| {
                        pipeline.use_expert(
                            low,
                            state.prepared,
                            state.scheduler,
                            &timesteps,
                            crossing..steps,
                            state.prepared.guidance_low,
                            &req.cancel,
                            state.on_progress,
                        )
                    },
                    || Ok(self.device.synchronize()?),
                )?;
            }
        }
        let vae = match resident_shared {
            Some(shared) => shared.vae,
            None => {
                on_progress(Progress::Loading(LoadPhase::Renderer));
                Arc::new(pipeline.load_decode_vae()?)
            }
        };
        on_progress(Progress::Decoding);
        let (frames, fps) = pipeline.finish(prepared, &vae, &req.cancel)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.i2v_memory.as_ref().map(|p| &p.contract)
    }
    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        self.i2v_memory.as_ref().map_or_else(
            || gen_core::MemorySafetyDecision::Reject {
                reason: "wan2_2_vace_fun_14b loaded route has no sealed memory contract".to_owned(),
            },
            |p| crate::i2v_memory_strategy::safety_check(p, context),
        )
    }
    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        self.i2v_memory.as_ref().map_or(Ok(None), |p| {
            crate::i2v_memory_strategy::begin_request(p, self.device.clone(), context)
        })
    }
}

pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // VACE-Fun rides the same z16 Wan VAE as its VACE sibling (`ProviderVae = WanVae16`), so it
        // declares the same denoiser latent space.
        denoiser_output_latent_space: Some(&candle_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID_VACE_FUN,
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            conditioning: vec![ConditioningKind::ControlClip, ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["uni_pc", "euler", "unipc"],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            // The public Candle route is the immutable dense bf16 VACE-Fun snapshot.
            // Do not advertise dormant packed tiers: they have no provider receipt.
            supported_quants: &[],
            supports_sequential_offload: true,
            ..Default::default()
        },
    }
}

/// Provider-owned inventory of the exact VACE-Fun component roots loaded by [`Pipeline`]. Both
/// complete transformer experts contribute to `dit`; missing component paths report no signal rather
/// than turning preflight accounting into an early load failure.
fn component_footprint(spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer", "transformer_2"],
        &["vae"],
    )
}

pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "wan2_2_vace_fun_14b expects a dual-expert snapshot directory".into(),
            ));
        }
    };
    let tier = VaceFunTierPaths::detect(&root)?;
    if let Some(tier) = &tier {
        // Preserve structural validation so malformed packed layouts fail deterministically;
        // a valid packed layout is still not an admitted public Candle artifact.
        tier.validate_requested_quant(spec.quantize)?;
        return Err(gen_core::Error::Unsupported(
            "wan2_2_vace_fun_14b Candle accepts only the sealed public raw dense bf16 snapshot; packed tiers are not a supported provider artifact".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "wan2_2_vace_fun_14b Candle accepts only the sealed public raw dense bf16 snapshot; quantized tiers are not supported".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "wan2_2_vace_fun_14b does not support ControlNet/IP-Adapter load overlays; pass VACE control as a per-request ControlClip".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_VACE_FUN)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(
            spec,
            MODEL_ID_VACE_FUN,
        )?)
    } else {
        None
    };
    Ok(Box::new(WanVaceFunGenerator {
        descriptor: descriptor(),
        root,
        device,
        adapters: spec.adapters.clone(),
        offload: spec.offload_policy,
        tier,
        shared: Mutex::new(None),
        experts: Mutex::new(None),
        i2v_memory,
    }))
}

candle_gen::register_generators! {
    pub(crate) const VACE_FUN_REGISTRATION = descriptor => load;
    footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vace::weighted_control_scale;
    use candle_gen::gen_core::Quant;
    use candle_gen::gen_core::{AdapterKind, ReplacementMode};
    use std::cell::{Cell, RefCell};

    fn request() -> GenerationRequest {
        let image = Image {
            width: 64,
            height: 64,
            pixels: vec![0; 64 * 64 * 3],
        };
        GenerationRequest {
            prompt: "controlled motion".into(),
            width: 64,
            height: 64,
            conditioning: vec![Conditioning::ControlClip {
                frames: vec![image.clone(); 5],
                mask: vec![image; 5],
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_and_registry_pin_dual_expert_surface() {
        let descriptor = descriptor();
        assert_eq!(descriptor.id, MODEL_ID_VACE_FUN);
        assert!(descriptor.capabilities.supports_lora);
        assert!(descriptor.capabilities.supports_lokr);
        assert!(descriptor.capabilities.supports_sequential_offload);
        assert!(descriptor
            .capabilities
            .accepts(ConditioningKind::ControlClip));
        let generator = crate::provider_registry()
            .unwrap()
            .load(
                MODEL_ID_VACE_FUN,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .expect("VACE-Fun registration is lazy");
        assert!(generator.validate(&request()).is_ok());
    }

    #[test]
    fn split_tier_loads_tokenizer_from_the_shared_parent_root() {
        let snapshot = tempfile::tempdir().unwrap();
        let tier_root = snapshot.path().join("q8");
        std::fs::create_dir(&tier_root).unwrap();
        for component in ["text_encoder", "vae", "tokenizer"] {
            std::fs::create_dir(snapshot.path().join(component)).unwrap();
        }
        // The tier directory deliberately has no tokenizer/: hosted split tiers keep this file at
        // the parent snapshot alongside the dense TE/VAE components.
        std::fs::write(
            snapshot.path().join("tokenizer/tokenizer.json"),
            r#"{
  "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
  "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null, "decoder": null,
  "model": { "type": "WordLevel", "vocab": { "<unk>": 0 }, "unk_token": "<unk>" }
}"#,
        )
        .unwrap();
        let tier = VaceFunTierPaths::test_paths(
            tier_root.clone(),
            snapshot.path().to_path_buf(),
            Quant::Q8,
        );
        let pipeline = Pipeline::new(&tier_root, &Device::Cpu, Vec::new(), Some(tier));
        assert_eq!(pipeline.shared_root(), snapshot.path());
        assert!(pipeline.build_tokenizer().is_ok());
    }

    #[test]
    fn component_footprint_counts_shared_and_both_expert_roots() {
        use std::fs::{self, File};

        let root = tempfile::tempdir().unwrap();
        for (component, bytes) in [
            ("text_encoder", 1_000),
            ("transformer", 2_000),
            ("transformer_2", 3_000),
            ("vae", 4_000),
        ] {
            let dir = root.path().join(component);
            fs::create_dir(&dir).unwrap();
            File::create(dir.join("model.safetensors"))
                .unwrap()
                .set_len(bytes)
                .unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));
        assert_eq!(
            component_footprint(&spec).unwrap(),
            PerComponentBytes {
                text_encoder: 1_000,
                dit: 5_000,
                vae: 4_000,
            }
        );

        let missing = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(missing.path().to_path_buf()));
        assert_eq!(
            component_footprint(&spec).unwrap(),
            PerComponentBytes::default()
        );
    }

    #[test]
    fn validation_applies_or_rejects_every_vace_fun_conditioning_field_preflight() {
        let generator = crate::provider_registry()
            .unwrap()
            .load(
                MODEL_ID_VACE_FUN,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap();

        let mut weighted = request();
        if let Conditioning::ControlClip {
            masking_strength, ..
        } = &mut weighted.conditioning[0]
        {
            *masking_strength = 0.4;
        }
        assert!(generator.validate(&weighted).is_ok());
        assert!((weighted_control_scale(Some(0.5), 0.4) - 0.2).abs() < f32::EPSILON);

        for invalid_strength in [-0.01, 1.01, f32::NAN] {
            let mut invalid = request();
            if let Conditioning::ControlClip {
                masking_strength, ..
            } = &mut invalid.conditioning[0]
            {
                *masking_strength = invalid_strength;
            }
            assert!(generator.validate(&invalid).is_err());
        }

        let mut duplicate = request();
        duplicate
            .conditioning
            .push(duplicate.conditioning[0].clone());
        assert!(generator.validate(&duplicate).is_err());

        let mut offset = request();
        if let Conditioning::ControlClip { start_frame, .. } = &mut offset.conditioning[0] {
            *start_frame = 1;
        }
        assert!(generator.validate(&offset).is_err());

        let mut full_person_mode = request();
        if let Conditioning::ControlClip { mode, .. } = &mut full_person_mode.conditioning[0] {
            *mode = gen_core::ReplacementMode::FullPersonKeepOutfit;
        }
        assert!(generator.validate(&full_person_mode).is_ok());

        let mut weighted_reference = request();
        weighted_reference
            .conditioning
            .push(Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![0; 64 * 64 * 3],
                },
                strength: Some(0.5),
            });
        assert!(generator.validate(&weighted_reference).is_err());
        assert!(generator
            .validate(&GenerationRequest {
                strength: Some(0.5),
                ..request()
            })
            .is_err());

        for shift in [0.0, f32::NAN, f32::INFINITY] {
            let invalid = GenerationRequest {
                scheduler_shift: Some(shift),
                ..request()
            };
            assert!(generator.validate(&invalid).is_err());
        }
        assert!(generator
            .validate(&GenerationRequest {
                steps: Some(1),
                ..request()
            })
            .is_err());
    }

    #[test]
    fn adapters_route_shared_and_expert_specific_without_silent_drop() {
        let shared = AdapterSpec::new("shared.safetensors".into(), 1.0, AdapterKind::Lora);
        let high = AdapterSpec::new("high.safetensors".into(), 1.0, AdapterKind::Lokr)
            .with_moe_expert(MoeExpert::High);
        let low = AdapterSpec::new("low.safetensors".into(), 1.0, AdapterKind::Lora)
            .with_moe_expert(MoeExpert::Low);
        let adapters = vec![shared, high, low];
        let high = selected_adapters(&adapters, MoeExpert::High);
        let low = selected_adapters(&adapters, MoeExpert::Low);
        assert_eq!(high.len(), 2);
        assert_eq!(low.len(), 2);
        assert!(high
            .iter()
            .any(|adapter| adapter.moe_expert == Some(MoeExpert::High)));
        assert!(low
            .iter()
            .any(|adapter| adapter.moe_expert == Some(MoeExpert::Low)));
    }

    struct LiveExperts {
        live: Cell<usize>,
        peak: Cell<usize>,
        events: RefCell<Vec<&'static str>>,
    }

    impl LiveExperts {
        fn new() -> Self {
            Self {
                live: Cell::new(0),
                peak: Cell::new(0),
                events: RefCell::new(Vec::new()),
            }
        }
    }

    struct ExpertWitness<'a> {
        live: &'a LiveExperts,
        drop_event: &'static str,
    }

    impl<'a> ExpertWitness<'a> {
        fn new(live: &'a LiveExperts, load: &'static str, drop_event: &'static str) -> Self {
            live.live.set(live.live.get() + 1);
            live.peak.set(live.peak.get().max(live.live.get()));
            live.events.borrow_mut().push(load);
            Self { live, drop_event }
        }
    }

    impl Drop for ExpertWitness<'_> {
        fn drop(&mut self) {
            self.live.live.set(self.live.live.get() - 1);
            self.live.events.borrow_mut().push(self.drop_event);
        }
    }

    #[test]
    fn sequential_vace_fun_never_co_resides_experts() {
        let live = LiveExperts::new();
        staged_expert_swap(
            2,
            5,
            &mut (),
            |_| Ok(ExpertWitness::new(&live, "load-high", "drop-high")),
            |_, _| Ok(()),
            |_| Ok(ExpertWitness::new(&live, "load-low", "drop-low")),
            |_, _| Ok(()),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(live.peak.get(), 1);
        assert_eq!(
            *live.events.borrow(),
            ["load-high", "drop-high", "load-low", "drop-low"]
        );
    }

    #[test]
    fn default_moe_contract_matches_wan22_vace_fun() {
        assert_eq!(BOUNDARY, 0.875);
        assert_eq!(DEFAULT_STEPS_14B, 40);
        assert_eq!(T2V_14B_FLOW_SHIFT, 12.0);
        assert_eq!((T2V_14B_GUIDANCE_LOW, T2V_14B_GUIDANCE_HIGH), (3.0, 4.0));
        let scheduler = FlowScheduler::new(
            Sampler::parse(None),
            DEFAULT_STEPS_14B as usize,
            T2V_14B_FLOW_SHIFT,
        );
        let timesteps = (0..DEFAULT_STEPS_14B as usize)
            .map(|step| scheduler.timestep(step))
            .collect::<Vec<_>>();
        let crossing = crossing_index(&timesteps, BOUNDARY * NUM_TRAIN_TIMESTEPS);
        assert!(crossing > 0 && crossing < timesteps.len());
        assert!(timesteps[..crossing].iter().all(|&t| t >= 875.0));
        assert!(timesteps[crossing..].iter().all(|&t| t < 875.0));
    }

    #[test]
    fn quantized_load_fails_closed_before_any_weight_or_adapter_load() {
        let spec = LoadSpec::new(WeightsSource::Dir("/snapshot".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&spec).err().expect("quantized load must fail"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn load_time_control_overlays_fail_closed_before_any_weight_or_adapter_load() {
        let spec = LoadSpec::new(WeightsSource::Dir("/snapshot".into()))
            .with_control(WeightsSource::Dir("/controlnet".into()));
        assert!(matches!(
            load(&spec)
                .err()
                .expect("load-time control overlay must fail"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
