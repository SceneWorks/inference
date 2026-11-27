//! The native Bernini **renderer** provider (sc-10994, the candle/CUDA sibling of `mlx-gen-bernini`'s
//! renderer, sc-4706): loads the dual-expert Wan2.2-T2V-A14B snapshot (converted from
//! `ByteDance/Bernini-Diffusers`, or a stock Wan2.2-T2V-A14B tier) + the UMT5/VAE/tokenizer, and runs
//! the boundary-switched, APG-guided denoise in **spatial latent space**, decoding to an image (1
//! frame) or video.
//!
//! Mirrors [`candle_gen_wan::wan14b`]'s dual-expert staging (UMT5 → high/low experts → z16 VAE) with the
//! plain-CFG combine replaced by the Bernini renderer's guided velocity: for the caption-only render the
//! default mode is `t2v_apg` (APG in x-space; the reference's `resolve_mode(None,false,false)`), with
//! plain CFG (`t2v`) selectable via `video_mode="t2v"`. Dual-expert switch: **high-noise** expert while
//! the integer timestep `≥ switch_dit_boundary·1000`, **low-noise** below — and on the first low-noise
//! step all omegas are scaled once by `OMEGA_SCALE` (the reference's `omega_scale`).
//!
//! **Scope (sc-11004):** every renderer guidance mode runs through [`crate::forward`]'s token-axis packed
//! conditioning — the text-only `t2v`/`t2v_apg` **and** the packed source-id conditioning modes
//! (`v2v`/`v2v_chain`/`v2v_apg`/`r2v_apg`/`rv2v`). A `Reference` (i2i), `MultiReference` (r2v refs), or
//! `VideoClip` (v2v/rv2v) is VAE-encoded to a z16 source latent, patch-embedded with its own source-id
//! RoPE, and packed on the DiT token axis with the noisy target; the target tokens are sliced back out to
//! the velocity. The Qwen2.5-VL planner / MAR / ViT-guidance path (the `*_wapg` modes) remains a follow-up
//! (the planner seams are sc-10995).
//!
//! **Dtypes:** UMT5 + z16 VAE run **f32**; the two experts run **bf16** (norms/modulation upcast to
//! f32); APG runs f32. `backend = "candle"`, `mac_only = false`. Q4/Q8 is a **packed tier** (the two
//! experts load through the sc-10025 packed-detect seam), streamed one expert at a time.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::driver::{
    run_candle_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverSupport, FamilyDriver, SliceInput, SliceSpec, SliceTransition, StepHooks,
};
use candle_gen::gen_core::{
    self, BoundaryState, CancelFlag, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};

use candle_gen_wan::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use candle_gen_wan::pipeline::{create_noise, frames_to_images};
use candle_gen_wan::scheduler::{flow_sigmas, Sampler, WanNativeDriver};
use gen_core_family::bernini_plan::{
    build_bernini_render_plan, BerniniBackendKind, BerniniEntrypoint, BerniniOutputContract,
    BerniniPlanKnobs, BerniniRenderPlan, BerniniRendererGuidanceMode,
};

use crate::components::RendererComponents;
use crate::config::{BerniniKnobs, Defaults, Mode};
use crate::forward::{
    guided_velocity, num_momentum_buffers, GuidanceParams, PackedForward, VitGuidanceParams,
};
use crate::guidance::MomentumBuffer;
use crate::latent_dims;
use crate::preprocess::{encode_image, encode_videoclip};

/// The experts run bf16 (the diffusers weights load as bf16, matching the 5B/14B regime); UMT5 + VAE f32.
const ENC_DTYPE: DType = DType::F32;
/// The A14B DiT emits 16-channel latents (z16 VAE).
const Z_DIM: usize = 16;

/// SceneWorks/engine model id (matches `mlx-gen-bernini`'s renderer so a consumer resolves the same
/// engine across backends). A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "bernini_renderer";

/// Primary typed route declaration for the Bernini renderer (`SW-6979`, `REG-1`).
pub(crate) fn route_renderer() -> gen_core::RouteRecord {
    #[cfg_attr(not(any(feature = "cuda", test)), allow(unused_mut))]
    let mut bindings = gen_core::RouteBindings::generator(descriptor, load);
    #[cfg(any(feature = "cuda", test))]
    {
        bindings = bindings
            .with_memory_registration(crate::memory_strategy::RENDERER_MEMORY_REGISTRATION)
            .with_memory_fixture_registration(crate::memory_strategy::RENDERER_MEMORY_FIXTURE)
            .with_memory_behavior_registration(crate::memory_strategy::RENDERER_MEMORY_BEHAVIOR);
    }

    gen_core::RouteRecord {
        route_id: MODEL_ID,
        family: "bernini",
        backend: "candle",
        modality: Modality::Video,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![gen_core::TextEncoderDeclaration::new(
                    "umt5_xxl",
                    Some(512),
                )],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                    ConditioningKind::VideoClip,
                ],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: vec![],
            },
            sampling: gen_core::SamplingSurface {
                // Curated `uni_pc` (sc-7296) → Wan's native UniPC; `euler` flow Euler. Legacy
                // `unipc` alias.
                samplers: vec!["uni_pc", "euler", "unipc"],
                schedulers: vec![],
                guidance_methods: vec![],
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_driver(
                gen_core::FamilyDriverSpec::new(stage2_driver_support("candle_bernini_renderer"))
                    .with_expert_switching(),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("wan_z16_vae"),
                output_latent_space: Some(&candle_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
                pixel_stride: Some(16),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            upsamplers: gen_core::UpsamplerSurface::default(),
            // User LoRA/LoKr stacks apply to both dense bf16 and packed q4/q8 renderer tiers.
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::ForwardResidual,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["umt5_xxl"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["wan_a14b_high_expert", "wan_a14b_low_expert"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["wan_z16_vae"],
                        false,
                    ),
                ],
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 16,
            max_size: 1280,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(Defaults::STEPS as u32),
            guidance: Some(Defaults::OMEGA_TXT),
            sampler: Some("uni_pc"),
            schedule: None,
            shift: Some(BerniniKnobs::default().shift),
            strength: None,
            dimensions: Some((832, 480)),
            ..Default::default()
        },
        bindings,
    }
}

/// Stable identity + advertised capabilities for the Bernini renderer (Wan2.2-A14B dual-expert with
/// APG guidance; caption→video/image). `backend = "candle"`, `mac_only = false`.
///
/// Computed from [`route_renderer`].
pub fn descriptor() -> ModelDescriptor {
    route_renderer().compute_descriptor()
}

/// A loaded Bernini renderer: resolved Bernini knobs + the snapshot dir, with the heavy components
/// (UMT5, the two experts, the z16 VAE) loaded lazily on the first `generate` and cached.
pub struct BerniniRenderer {
    descriptor: ModelDescriptor,
    knobs: BerniniKnobs,
    root: PathBuf,
    device: Device,
    adapters: Vec<candle_gen::gen_core::AdapterSpec>,
    components: Mutex<Option<Arc<RendererComponents>>>,
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    memory_tier: Option<gen_core::MemoryNumericTier>,
    /// Why this load has no sealed memory receipt. Present exactly when `memory_strategy` is
    /// `None`; it is the refusal the memory seams return instead of admitting an unsealed artifact.
    memory_refusal: String,
    loaded_spec: LoadSpec,
}

impl BerniniKnobs {
    pub(crate) fn to_plan_knobs(&self) -> BerniniPlanKnobs {
        BerniniPlanKnobs {
            switch_dit_boundary: self.switch_dit_boundary,
            shift: self.shift,
            max_trained_src_id: self.max_trained_src_id,
            interpolate_src_id: self.interpolate_src_id,
            max_sequence_length: self.max_sequence_length,
        }
    }
}

fn renderer_mode_from_plan(mode: BerniniRendererGuidanceMode) -> Mode {
    match mode {
        BerniniRendererGuidanceMode::T2v => Mode::T2v,
        BerniniRendererGuidanceMode::T2vApg => Mode::T2vApg,
        BerniniRendererGuidanceMode::V2v => Mode::V2v,
        BerniniRendererGuidanceMode::V2vChain => Mode::V2vChain,
        BerniniRendererGuidanceMode::V2vApg => Mode::V2vApg,
        BerniniRendererGuidanceMode::R2vApg => Mode::R2vApg,
        BerniniRendererGuidanceMode::Rv2v => Mode::Rv2v,
    }
}

/// Active dual-expert branch for Bernini Stage 2 DiT denoising (`High` for `t >= boundary_ts`,
/// `Low` below it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BerniniExpert {
    High,
    Low,
}

/// Unified guidance weights carried across the Stage 2 `high -> low` boundary switch where all
/// active omegas are multiplied once by `omega_scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BerniniStage2Omegas {
    pub omega_vid: f32,
    pub omega_img: f32,
    pub omega_txt: f32,
    pub omega_tgt: Option<f32>,
}

impl BerniniStage2Omegas {
    pub fn from_vit(g: &VitGuidanceParams) -> Self {
        Self {
            omega_vid: g.omega_vid,
            omega_img: g.omega_img,
            omega_txt: g.omega_txt,
            omega_tgt: Some(g.omega_tgt),
        }
    }

    pub fn to_guidance(&self, base: &GuidanceParams) -> GuidanceParams {
        GuidanceParams {
            omega_vid: self.omega_vid,
            omega_img: self.omega_img,
            omega_txt: self.omega_txt,
            eta: base.eta,
            norm_threshold: base.norm_threshold,
        }
    }

    pub fn to_vit(&self, base: &VitGuidanceParams) -> VitGuidanceParams {
        VitGuidanceParams {
            omega_txt: self.omega_txt,
            omega_img: self.omega_img,
            omega_vid: self.omega_vid,
            omega_tgt: self.omega_tgt.unwrap_or(base.omega_tgt),
            eta: base.eta,
            norm_threshold: base.norm_threshold,
        }
    }

    fn scale_once(&mut self, factor: f32) {
        self.omega_vid *= factor;
        self.omega_img *= factor;
        self.omega_txt *= factor;
        if let Some(tgt) = &mut self.omega_tgt {
            *tgt *= factor;
        }
    }

    fn bits(self) -> [u32; 4] {
        [
            self.omega_vid.to_bits(),
            self.omega_img.to_bits(),
            self.omega_txt.to_bits(),
            self.omega_tgt.map_or(0, f32::to_bits),
        ]
    }
}

/// Forward-evaluation seam for [`BerniniDualExpertDriver`] (`PIPE-7`, `DEC-13`).
pub trait BerniniStage2Forward {
    /// Compute the guided Stage 2 velocity at `step` for `expert`, using the boundary-scaled
    /// `omegas` and persistent APG `mbufs`.
    #[allow(clippy::too_many_arguments)]
    fn predict_step_velocity(
        &self,
        expert: BerniniExpert,
        latent: &Tensor,
        step: usize,
        timestep: f64,
        sigma: f32,
        omegas: BerniniStage2Omegas,
        switched_to_low: bool,
        mbufs: &mut [MomentumBuffer],
    ) -> CResult<Tensor>;
}

struct BerniniStage2History {
    sampler: Sampler,
    total_steps: usize,
    boundary_ts_bits: u64,
    omega_scale_bits: u32,
    shift_bits: u64,
    base_omegas_bits: [u32; 4],
    num_momentum_buffers: usize,
    momentum_bits: u32,
    switched: bool,
    active_omegas: BerniniStage2Omegas,
    mbufs: Vec<MomentumBuffer>,
    expected_next_step: usize,
    wan_state: DriverRunState,
}

struct ActiveStage2State {
    switched: bool,
    active_omegas: BerniniStage2Omegas,
    mbufs: Vec<MomentumBuffer>,
}

/// Typed [`FamilyDriver`] for Candle Bernini Stage 2 dual-expert flow denoising across both
/// `bernini_renderer` (`Mode`) and full `bernini` (`VitMode`) (`PIPE-7`, `DEC-13`).
///
/// Delegates solver integration (`UniPC` / `Dpmpp2m` / `Euler`) to [`WanNativeDriver`] while
/// persisting the one-time `omega_scale` boundary latch and APG momentum buffers inside
/// [`DriverRunState`].
pub struct BerniniDualExpertDriver<F> {
    sampler: Sampler,
    boundary_ts: f64,
    steps: usize,
    shift: f64,
    sigmas: Vec<f32>,
    base_omegas: BerniniStage2Omegas,
    omega_scale: f32,
    num_momentum_buffers: usize,
    momentum: f32,
    forward: F,
    support: DriverSupport,
}

pub(crate) fn stage2_driver_support(driver_id: &'static str) -> DriverSupport {
    DriverSupport {
        driver_id,
        arbitrary_slice: true,
        carry_history: true,
        reset_history: true,
        renoise_from_clean: false,
        geometry_change: false,
        read_only_boundary_estimate: false,
        previous_estimate_capture: false,
        uses_global_rng: false,
        refusal_reasons: DriverRefusalReasons {
            renoise_from_clean:
                "native Candle Bernini Stage 2 driver does not re-noise from clean estimates",
            geometry_change:
                "native Candle Bernini Stage 2 driver does not support geometry-changing phase boundaries",
            read_only_boundary_estimate:
                "native Candle Bernini Stage 2 driver does not expose read-only boundary clean estimates",
            ..DriverRefusalReasons::default()
        },
    }
}

impl<F: BerniniStage2Forward> BerniniDualExpertDriver<F> {
    fn build_support(driver_id: &'static str) -> DriverSupport {
        stage2_driver_support(driver_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        driver_id: &'static str,
        sampler: Sampler,
        boundary_ts: f64,
        steps: usize,
        shift: f64,
        base_omegas: BerniniStage2Omegas,
        omega_scale: f32,
        num_momentum_buffers: usize,
        momentum: f32,
        forward: F,
    ) -> CResult<Self> {
        if steps == 0 {
            return Err(CandleError::Msg(format!(
                "{driver_id}: Stage 2 steps must be >= 1"
            )));
        }
        let sigmas = flow_sigmas(steps, shift);
        Ok(Self {
            sampler,
            boundary_ts,
            steps,
            shift,
            sigmas,
            base_omegas,
            omega_scale,
            num_momentum_buffers,
            momentum,
            forward,
            support: Self::build_support(driver_id),
        })
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
}

impl<F: BerniniStage2Forward> FamilyDriver<Tensor> for BerniniDualExpertDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Tensor>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Tensor>,
    ) -> candle_gen::gen_core::Result<BoundaryState<Tensor>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps != self.steps || input.span.end > self.steps {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Bernini Stage 2 steps {}",
                self.support.driver_id, input.span, input.total_steps, self.steps
            )));
        }

        let carrying = input.transition == SliceTransition::Continue
            && input.history == DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<BerniniStage2History>() else {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry requested without prior Bernini Stage 2 history in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.sampler != self.sampler
                || existing.total_steps != input.total_steps
                || existing.boundary_ts_bits != self.boundary_ts.to_bits()
                || existing.omega_scale_bits != self.omega_scale.to_bits()
                || existing.shift_bits != self.shift.to_bits()
                || existing.base_omegas_bits != self.base_omegas.bits()
                || existing.num_momentum_buffers != self.num_momentum_buffers
                || existing.momentum_bits != self.momentum.to_bits()
            {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible Bernini Stage 2 schedule or guidance settings",
                    self.support.driver_id
                )));
            }
            if existing.expected_next_step != input.span.start {
                return Err(candle_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused at non-contiguous step {} (expected {})",
                    self.support.driver_id, input.span.start, existing.expected_next_step
                )));
            }
        }

        state.enter_slice(&input)?;
        let mut stage2 = if carrying {
            state
                .take_history::<BerniniStage2History>()
                .expect("validated BerniniStage2History")
        } else {
            BerniniStage2History {
                sampler: self.sampler,
                total_steps: input.total_steps,
                boundary_ts_bits: self.boundary_ts.to_bits(),
                omega_scale_bits: self.omega_scale.to_bits(),
                shift_bits: self.shift.to_bits(),
                base_omegas_bits: self.base_omegas.bits(),
                num_momentum_buffers: self.num_momentum_buffers,
                momentum_bits: self.momentum.to_bits(),
                switched: false,
                active_omegas: self.base_omegas,
                mbufs: (0..self.num_momentum_buffers)
                    .map(|_| MomentumBuffer::new(self.momentum))
                    .collect(),
                expected_next_step: input.span.start,
                wan_state: DriverRunState::new(),
            }
        };

        let prev_evals = stage2.wan_state.internal_evaluations;
        let prev_end = stage2.expected_next_step;
        let active_cell = RefCell::new(ActiveStage2State {
            switched: stage2.switched,
            active_omegas: stage2.active_omegas,
            mbufs: std::mem::take(&mut stage2.mbufs),
        });

        let wan_res = {
            let wan_driver = WanNativeDriver::new(
                self.support.driver_id,
                self.sampler,
                self.steps,
                self.shift,
                |latent: &Tensor, step: usize, t: f64| {
                    let mut st = active_cell.borrow_mut();
                    let expert = if t >= self.boundary_ts {
                        BerniniExpert::High
                    } else {
                        if !st.switched {
                            st.switched = true;
                            st.active_omegas.scale_once(self.omega_scale);
                        }
                        BerniniExpert::Low
                    };
                    let switched = st.switched;
                    let omegas = st.active_omegas;
                    let sigma = self.sigmas[step];
                    self.forward.predict_step_velocity(
                        expert,
                        latent,
                        step,
                        t,
                        sigma,
                        omegas,
                        switched,
                        &mut st.mbufs,
                    )
                },
            );
            wan_driver.run_slice(
                SliceInput {
                    pass: input.pass,
                    phase: input.phase,
                    span: input.span.clone(),
                    total_steps: input.total_steps,
                    state: input.state,
                    transition: input.transition,
                    history: input.history,
                    geometry: input.geometry,
                    seed: input.seed,
                    cancel: input.cancel,
                    request_boundary_estimate: input.request_boundary_estimate,
                },
                &mut stage2.wan_state,
                hooks,
            )
        };

        let active = active_cell.into_inner();
        stage2.switched = active.switched;
        stage2.active_omegas = active.active_omegas;
        stage2.mbufs = active.mbufs;

        let delta_evals = stage2.wan_state.internal_evaluations - prev_evals;
        for _ in 0..delta_evals {
            state.record_evaluation(DriverEvaluationKind::Trajectory);
        }
        let new_end = stage2.wan_state.last_span_end.unwrap_or(prev_end);
        for completed_step in prev_end..new_end {
            state.record_outer_step(completed_step);
        }
        stage2.expected_next_step = new_end;
        state.set_history(stage2);

        let out = wan_res?;
        state.finish_slice(input.span.end);
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_bernini_candle_stage2_driver<F: BerniniStage2Forward>(
    driver_id: &'static str,
    sampler: Sampler,
    boundary_ts: f64,
    steps: usize,
    shift: f64,
    init_noise: &Tensor,
    base_omegas: BerniniStage2Omegas,
    omega_scale: f32,
    num_momentum_buffers: usize,
    momentum: f32,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    forward: F,
) -> CResult<Tensor> {
    let driver = BerniniDualExpertDriver::new(
        driver_id,
        sampler,
        boundary_ts,
        steps,
        shift,
        base_omegas,
        omega_scale,
        num_momentum_buffers,
        momentum,
        forward,
    )?;
    let mut state = DriverRunState::new();
    let mut hooks = StepHooks::new(on_progress);
    let out = run_candle_driver_slices(
        &driver,
        [SliceSpec::full(steps)],
        BoundaryState::XSigma(init_noise.clone()),
        steps,
        seed,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    let BoundaryState::XSigma(latent) = out else {
        unreachable!("BerniniDualExpertDriver always emits BoundaryState::XSigma");
    };
    Ok(latent)
}

impl BerniniRenderer {
    /// Build and validate the pre-load [`BerniniRenderPlan`] before loading weights (`DEC-3`).
    pub fn render_plan(&self, req: &GenerationRequest) -> gen_core::Result<BerniniRenderPlan> {
        build_bernini_render_plan(
            &self.descriptor,
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Candle,
            self.knobs.to_plan_knobs(),
            req,
        )
    }

    fn components(&self) -> CResult<Arc<RendererComponents>> {
        candle_gen::cached(&self.components, || {
            if let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) {
                crate::memory_strategy::validate_loaded_contract(
                    MODEL_ID,
                    &self.loaded_spec,
                    contract,
                    tier,
                )
                .map_err(|error| CandleError::Msg(error.to_string()))?;
            }
            let components =
                RendererComponents::load(&self.root, &self.device, MODEL_ID, &self.adapters)?;
            if let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) {
                crate::memory_strategy::validate_loaded_contract(
                    MODEL_ID,
                    &self.loaded_spec,
                    contract,
                    tier,
                )
                .map_err(|error| CandleError::Msg(error.to_string()))?;
            }
            Ok(Arc::new(components))
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (f32), zero-padded/truncated to `max_length`
    /// (the DiT cross-attends over the 512-padded context — the Wan training convention, sc-3697). The
    /// empty-prompt guard (sc-7078) emits one pad token so a 0-length sequence never reaches the CUDA
    /// embedding gather. Replicates `candle_gen_wan`'s crate-private `umt5_encode_padded`.
    fn encode(&self, comps: &RendererComponents, prompt: &str) -> CResult<Tensor> {
        let te_cfg = TextEncoderConfig::umt5_xxl();
        let out = comps
            .tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("bernini_renderer: tokenize: {e}")))?;
        let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        if ids.is_empty() {
            ids.push(te_cfg.pad_token_id as u32);
        }
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = comps.te.encode(&input_ids)?.to_dtype(ENC_DTYPE)?; // [1, S, 4096]
        let max_len = te_cfg.max_length;
        let dim = embeds.dim(2)?;
        let padded = match len.cmp(&max_len) {
            std::cmp::Ordering::Less => {
                let pad = Tensor::zeros((1, max_len - len, dim), ENC_DTYPE, &self.device)?;
                Tensor::cat(&[&embeds, &pad], 1)?
            }
            std::cmp::Ordering::Greater => embeds.narrow(1, 0, max_len)?,
            std::cmp::Ordering::Equal => embeds,
        };
        Ok(padded)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &RendererComponents,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let render_plan = self
            .render_plan(req)
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let frames = render_plan.frames as u32;
        let width = render_plan.width;
        let height = render_plan.height;
        let steps = render_plan.steps;
        let seed = render_plan.seed.unwrap_or_else(gen_core::default_seed);
        let sampler = Sampler::parse(Some(&render_plan.sampler));
        let shift = render_plan.shift;
        let mode = renderer_mode_from_plan(
            render_plan
                .renderer_mode
                .expect("Renderer plan always resolves a guidance mode"),
        );

        // VAE-encode the source media → z16 conditioning latents (each `[1, 16, T, H8, W8]`). VideoClips
        // become `videos`; Reference / MultiReference images become `images`. The packed forward assigns
        // each its source-id RoPE (videos first) and concatenates them on the DiT token axis.
        let mut videos: Vec<Tensor> = Vec::new();
        let mut images: Vec<Tensor> = Vec::new();
        for c in &req.conditioning {
            match c {
                Conditioning::VideoClip { frames: fr, .. } => {
                    videos.push(encode_videoclip(
                        &comps.vae,
                        fr,
                        width,
                        height,
                        &self.device,
                    )?);
                }
                Conditioning::Reference { image, .. } => {
                    images.push(encode_image(
                        &comps.vae,
                        image,
                        width,
                        height,
                        &self.device,
                    )?);
                }
                Conditioning::MultiReference { images: imgs } => {
                    for im in imgs {
                        images.push(encode_image(&comps.vae, im, width, height, &self.device)?);
                    }
                }
                other => {
                    return Err(CandleError::Msg(format!(
                        "bernini_renderer: unsupported conditioning kind {:?} (only Reference / \
                         MultiReference / VideoClip are packed as source-id conditioning)",
                        other.kind()
                    )));
                }
            }
        }

        // --- Text encode (pos + neg) once; project to each expert's context (per-expert embedder) ---
        let (pos, neg) = candle_gen::gen_core::with_prompt_encoding(
            on_progress,
            || -> std::result::Result<_, candle_gen::CandleError> {
                let pos = self.encode(comps, &req.prompt)?;
                let neg_prompt = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
                let neg = self.encode(comps, neg_prompt)?;
                Ok((pos, neg))
            },
        )?;
        let high_pos = comps.high.embed_text(&pos)?;
        let low_pos = comps.low.embed_text(&pos)?;
        let high_neg = comps.high.embed_text(&neg)?;
        let low_neg = comps.low.embed_text(&neg)?;

        // --- Latent geometry (z16 strides). RoPE is now built per token-segment inside `PackedForward`
        // (each source + the target gets its own source-id-shifted table), so no shared cos/sin here. ---
        let (t_lat, h_lat, w_lat) = latent_dims(frames, width, height);
        let dit_cfg = TransformerConfig::t2v_14b();

        let init_noise = create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;
        let boundary_ts = render_plan.boundary_timestep;

        let pf = PackedForward::new(
            dit_cfg,
            render_plan.trained_sources,
            self.knobs.interpolate_src_id,
        );
        let base_omegas = BerniniStage2Omegas {
            omega_vid: render_plan.omega_vid,
            omega_img: render_plan.omega_img,
            omega_txt: render_plan.omega_txt,
            omega_tgt: None,
        };
        let base_guidance = GuidanceParams {
            omega_vid: render_plan.omega_vid,
            omega_img: render_plan.omega_img,
            omega_txt: render_plan.omega_txt,
            eta: render_plan.eta,
            norm_threshold: [Defaults::NORM_THRESHOLD, Defaults::NORM_THRESHOLD],
        };

        struct RendererStage2Forward<'a> {
            pf: &'a PackedForward,
            mode: Mode,
            comps: &'a RendererComponents,
            high_pos: &'a Tensor,
            high_neg: &'a Tensor,
            low_pos: &'a Tensor,
            low_neg: &'a Tensor,
            videos: &'a [Tensor],
            images: &'a [Tensor],
            base_guidance: &'a GuidanceParams,
        }

        impl BerniniStage2Forward for RendererStage2Forward<'_> {
            fn predict_step_velocity(
                &self,
                expert: BerniniExpert,
                latent: &Tensor,
                _step: usize,
                timestep: f64,
                sigma: f32,
                omegas: BerniniStage2Omegas,
                switched_to_low: bool,
                mbufs: &mut [MomentumBuffer],
            ) -> CResult<Tensor> {
                let (dit, ctx_pos, ctx_neg) = match expert {
                    BerniniExpert::High => (&self.comps.high, self.high_pos, self.high_neg),
                    BerniniExpert::Low => (&self.comps.low, self.low_pos, self.low_neg),
                };
                let g = omegas.to_guidance(self.base_guidance);
                candle_gen::execution::policy(
                    "bernini_renderer_guidance",
                    serde_json::json!({
                        "mode": format!("{:?}", self.mode), "low_expert": switched_to_low,
                        "omega_txt_bits": g.omega_txt.to_bits(), "omega_img_bits": g.omega_img.to_bits(),
                        "omega_vid_bits": g.omega_vid.to_bits(), "eta_bits": g.eta.to_bits(),
                        "norm_threshold_bits": g.norm_threshold.map(f32::to_bits)
                    }),
                );
                guided_velocity(
                    self.pf,
                    self.mode,
                    dit,
                    latent,
                    self.videos,
                    self.images,
                    timestep,
                    sigma,
                    ctx_pos,
                    ctx_neg,
                    &g,
                    mbufs,
                )
            }
        }

        let latents = run_bernini_candle_stage2_driver(
            "bernini_candle_renderer_stage2_driver",
            sampler,
            boundary_ts,
            steps,
            shift,
            &init_noise,
            base_omegas,
            render_plan.omega_scale,
            num_momentum_buffers(mode),
            Defaults::MOMENTUM,
            seed,
            &req.cancel,
            on_progress,
            RendererStage2Forward {
                pf: &pf,
                mode,
                comps,
                high_pos: &high_pos,
                high_neg: &high_neg,
                low_pos: &low_pos,
                low_neg: &low_neg,
                videos: &videos,
                images: &images,
                base_guidance: &base_guidance,
            },
        )?;

        on_progress(Progress::Decoding);
        let decode_cap = crate::memory_strategy::selected_decode_cap(req)?;
        let decoded = comps.vae.decode_budgeted_with_cancel_and_tile_cap(
            &latents,
            &req.cancel,
            decode_cap,
        )?;
        let out_images = frames_to_images(&decoded)?;

        match render_plan.output_contract {
            BerniniOutputContract::Images { .. } => {
                let first = out_images.into_iter().next().ok_or_else(|| {
                    CandleError::Msg("bernini_renderer: VAE decode produced no frames".into())
                })?;
                Ok(GenerationOutput::Images(vec![first]))
            }
            BerniniOutputContract::Video { fps, .. } => Ok(GenerationOutput::Video {
                frames: out_images,
                fps,
                audio: None,
            }),
        }
    }
}

#[cfg(test)]
type Expert = BerniniExpert;

#[cfg(test)]
fn select_expert(t: f64, boundary_ts: f64, switched: &mut bool, omega: &mut f32) -> Expert {
    if t >= boundary_ts {
        Expert::High
    } else {
        if !*switched {
            *switched = true;
            *omega *= Defaults::OMEGA_SCALE;
        }
        Expert::Low
    }
}

/// Load the Bernini renderer from a converted snapshot directory (`text_encoder/`, `transformer/`,
/// `transformer_2/`, `vae/`, `tokenizer/`, + optional `bernini_renderer.json` knobs). The dual-expert
/// snapshot is a Wan2.2-T2V-A14B diffusers layout produced by [`crate::convert`] from
/// `ByteDance/Bernini-Diffusers` (or a stock Wan2.2-T2V-A14B tier for a raw-render validation).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "bernini_renderer: expected a snapshot directory (text_encoder/ transformer/ \
                 transformer_2/ vae/ tokenizer/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    // Control / VACE / IP-adapter overlays are not part of the renderer surface.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "bernini_renderer does not support control / VACE / IP-adapter overlays".into(),
        ));
    }
    // `spec.quantize` is a no-op tier-select marker (mirrors wan14b/ltx): a q4/q8 A14B tier is
    // pre-packed (the packed-detect loaders read its `.scales`), a dense tier loads dense — so it does
    // NOT reject here; both experts load through the sc-10025 packed-detect seam.
    let knobs = BerniniKnobs::from_dir(&root)?;
    let device = candle_gen::default_device()?;
    // A crossed, corrupted, or unparseable snapshot yields no contract — but it must never yield a
    // silent `Accept` at the memory seams, so carry the refusal with the load.
    #[cfg(any(feature = "cuda", test))]
    let (memory_strategy, memory_tier, memory_refusal) =
        match crate::memory_strategy::contract_for_loaded(spec, MODEL_ID) {
            Ok((contract, tier)) => (Some(contract), Some(tier), String::new()),
            Err(reason) => (None, None, reason),
        };
    #[cfg(not(any(feature = "cuda", test)))]
    let (memory_strategy, memory_tier, memory_refusal) = (
        None,
        None,
        format!("{MODEL_ID}: this build has no Candle/CUDA memory-strategy support"),
    );
    Ok(Box::new(BerniniRenderer {
        descriptor: descriptor(),
        knobs,
        root,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
        memory_strategy,
        memory_tier,
        memory_refusal,
        loaded_spec: spec.clone(),
    }))
}

impl Generator for BerniniRenderer {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.render_plan(req).map(|_| ())
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        crate::components::capture_renderer_execution(self, req, on_progress, self.adapters.len())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let comps = self.components()?;
        Ok(self.render(req, &comps, on_progress)?)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) else {
            return gen_core::MemorySafetyDecision::Reject {
                reason: self.memory_refusal.clone(),
            };
        };
        crate::memory_strategy::safety_check(contract, tier, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) else {
            return Err(gen_core::Error::Unsupported(self.memory_refusal.clone()));
        };
        crate::memory_strategy::begin_request(contract, tier, self.device.clone(), context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen_wan::config::NUM_TRAIN_TIMESTEPS;
    use candle_gen_wan::scheduler::FlowScheduler;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // load is lazy (components build on first generate), so the registry resolves + `load` succeeds
        // even for a missing dir; the descriptor identity is what we pin here.
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("bernini_renderer is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "bernini");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
        assert!(!g.descriptor().capabilities.mac_only);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert!(!d.capabilities.supports_sequential_offload);
        assert!(!d.capabilities.unconditionally_engages_staged_residency);
        assert_eq!(
            d.capabilities.staged_residency_availability(),
            candle_gen::gen_core::StagedResidencyAvailability::Absent,
        );
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        // Q4 and Q8 packed tiers both supported (streamed one expert at a time).
        assert!(d.capabilities.supported_quants.contains(&Quant::Q4));
        assert!(d.capabilities.supported_quants.contains(&Quant::Q8));
        assert!(d.capabilities.samplers.contains(&"uni_pc"));
        assert!(d.capabilities.samplers.contains(&"unipc")); // legacy alias
                                                             // Packed source-id conditioning surface (sc-11004): reference image / multi-reference / video clip.
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.accepts(ConditioningKind::VideoClip));
    }

    #[test]
    fn load_rejects_single_file_and_overlays() {
        // single-file source
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // Quant is accepted (no-op packed-tier marker): load is lazy, so it succeeds past the marker.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "q8 is a packed-tier select marker, not rejected"
        );
        let quant4 = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load(&quant4).is_ok());
    }

    #[test]
    fn validate_enforces_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            guidance: Some(4.0),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                frames: Some(16),
                ..Default::default()
            },
            // size not a multiple of 16
            GenerationRequest {
                prompt: "x".into(),
                width: 300,
                height: 256,
                ..Default::default()
            },
            // explicit zero steps
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                steps: Some(0),
                ..Default::default()
            },
            // over the max-area envelope
            GenerationRequest {
                prompt: "x".into(),
                width: 1280,
                height: 1024,
                frames: Some(17),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// F-096: `validate` rejects a conditioning-mode name with no source (would silently render
    /// text-only) and a text-only mode name with a source present (would silently drop it). The
    /// text-only default with no conditioning stays valid.
    #[test]
    fn validate_rejects_mode_conditioning_mismatch() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        // Conditioning mode name, no conditioning → reject.
        let v2v_no_src = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            video_mode: Some("v2v".into()),
            ..Default::default()
        };
        assert!(g.validate(&v2v_no_src).is_err());
        // Text-only default, no conditioning → ok.
        let t2v = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(g.validate(&t2v).is_ok());
        // Text-only mode name WITH conditioning → reject (would be silently dropped).
        let t2v_with_src = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            video_mode: Some("t2v_apg".into()),
            conditioning: vec![Conditioning::Reference {
                image: candle_gen::gen_core::Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0u8; 2 * 2 * 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(g.validate(&t2v_with_src).is_err());
    }

    /// The dual-expert switch + omega-scale-once latch (the story AC's "expert selection" test). Asserts
    /// (a) high→low transition happens exactly at the boundary, (b) omega scales exactly once on the
    /// first sub-boundary step and not again, (c) high expert above the boundary.
    #[test]
    fn select_expert_switches_at_boundary_and_scales_omega_once() {
        let boundary =
            BerniniKnobs::default().switch_dit_boundary as f64 * NUM_TRAIN_TIMESTEPS as f64; // 0.875 * 1000 = 875.0
        let base = Defaults::OMEGA_TXT;

        let mut switched = false;
        let mut omega = base;

        // (c) Above the boundary → high expert; no switch, omega untouched.
        assert_eq!(
            select_expert(900.0, boundary, &mut switched, &mut omega),
            Expert::High
        );
        assert!(!switched);
        assert_eq!(omega, base);

        // (a) Exactly AT the boundary is still high-noise (the switch is `t >= boundary_ts`).
        assert_eq!(
            select_expert(boundary, boundary, &mut switched, &mut omega),
            Expert::High
        );
        assert!(!switched);
        assert_eq!(omega, base);

        // (a) First step below the boundary → low expert, latch flips.
        assert_eq!(
            select_expert(boundary - 1.0, boundary, &mut switched, &mut omega),
            Expert::Low
        );
        assert!(switched);
        // (b) omega scaled exactly once.
        assert_eq!(omega, base * Defaults::OMEGA_SCALE);

        // (b) Subsequent low-noise steps do NOT scale omega again.
        let after_first = omega;
        assert_eq!(
            select_expert(100.0, boundary, &mut switched, &mut omega),
            Expert::Low
        );
        assert_eq!(omega, after_first);
        assert_eq!(
            select_expert(0.0, boundary, &mut switched, &mut omega),
            Expert::Low
        );
        assert_eq!(omega, after_first);
    }

    /// Scheduler wiring: the `boundary_ts` axis and the `flow_sigmas(steps, shift)` the APG x-space
    /// conversion indexes both share the `FlowScheduler`'s internal sigma indexing — `timestep(i) ==
    /// σ_i · NUM_TRAIN_TIMESTEPS`. Guards the sigma-vs-index alignment the APG conversion depends on.
    #[test]
    fn flow_sigmas_align_with_scheduler_timesteps() {
        let steps = 8;
        let shift = 3.0_f64;
        let sched = FlowScheduler::new(Sampler::UniPC, steps, shift);
        let sigmas = flow_sigmas(steps, shift);
        assert_eq!(sigmas.len(), steps + 1); // terminal 0.0
        for (i, &sigma) in sigmas.iter().enumerate().take(steps) {
            // sched.timestep uses the f64 sigmas; flow_sigmas is the f32-cast schedule the APG loop
            // indexes at the same `i`. They align within f32 rounding on the 0..1000 timestep axis.
            let expected = sigma as f64 * NUM_TRAIN_TIMESTEPS as f64;
            assert!(
                (sched.timestep(i) - expected).abs() < 1e-3,
                "step {i}: timestep {} vs σ_i·N {expected}",
                sched.timestep(i)
            );
        }
        // boundary_ts (render's expert switch) lives on that same 0..1000 timestep axis.
        let boundary_ts =
            BerniniKnobs::default().switch_dit_boundary as f64 * NUM_TRAIN_TIMESTEPS as f64;
        assert!((boundary_ts - 875.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod driver_tests {
    use super::*;
    use candle_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};
    use candle_gen::sampler::CandleLatentOps;
    use std::cell::RefCell;
    use std::rc::Rc;

    type Stage2Log =
        Rc<RefCell<Vec<(usize, BerniniExpert, f64, BerniniStage2Omegas, bool, usize)>>>;

    struct MockCandleStage2Forward {
        log: Stage2Log,
    }

    impl BerniniStage2Forward for MockCandleStage2Forward {
        fn predict_step_velocity(
            &self,
            expert: BerniniExpert,
            latent: &Tensor,
            step: usize,
            timestep: f64,
            sigma: f32,
            omegas: BerniniStage2Omegas,
            switched_to_low: bool,
            mbufs: &mut [MomentumBuffer],
        ) -> CResult<Tensor> {
            let mut non_empty_mbufs = 0usize;
            for buf in mbufs.iter_mut() {
                let diff = latent.affine(0.05, 0.0)?;
                let _ = buf.update(&CandleLatentOps, &diff)?;
                if !buf.is_empty() {
                    non_empty_mbufs += 1;
                }
            }
            self.log.borrow_mut().push((
                step,
                expert,
                timestep,
                omegas,
                switched_to_low,
                non_empty_mbufs,
            ));
            let expert_gain = match expert {
                BerniniExpert::High => -0.6,
                BerniniExpert::Low => -0.35,
            };
            let omega_sum = omegas.omega_txt
                + omegas.omega_img
                + omegas.omega_vid
                + omegas.omega_tgt.unwrap_or(0.0);
            Ok(latent.affine(
                expert_gain - 0.05 * sigma as f64,
                0.002 * omega_sum as f64 + 0.0001 * timestep,
            )?)
        }
    }

    fn tensor_f32_bit_equal(a: &Tensor, b: &Tensor) -> bool {
        if a.dims() != b.dims() {
            return false;
        }
        let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        va.len() == vb.len() && va.iter().zip(vb).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn bernini_candle_dual_expert_driver_passes_conformance_across_boundary() {
        let dev = Device::Cpu;
        let init = Tensor::from_vec(vec![0.5f32, -0.4, 0.8, -0.2], (4,), &dev).unwrap();
        let renderer_omegas = BerniniStage2Omegas {
            omega_vid: Defaults::OMEGA_VID,
            omega_img: Defaults::OMEGA_IMG,
            omega_txt: Defaults::OMEGA_TXT,
            omega_tgt: None,
        };
        let vit_omegas = BerniniStage2Omegas {
            omega_vid: Defaults::OMEGA_VID,
            omega_img: Defaults::OMEGA_IMG,
            omega_txt: Defaults::OMEGA_TXT,
            omega_tgt: Some(0.5),
        };

        for (name, sampler, omegas, n_bufs, momentum) in [
            (
                "bernini_candle_stage2_unipc_apg",
                Sampler::UniPC,
                renderer_omegas,
                2usize,
                0.5f32,
            ),
            (
                "bernini_candle_stage2_euler_vit",
                Sampler::Euler,
                vit_omegas,
                0usize,
                0.0f32,
            ),
        ] {
            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 6,
                split_at: 3,
                seed: 42,
                base_geometry: (256, 256),
                alt_geometry: (512, 512),
                make_driver: || {
                    Box::new(
                        BerniniDualExpertDriver::new(
                            name,
                            sampler,
                            875.0,
                            6,
                            5.0,
                            omegas,
                            Defaults::OMEGA_SCALE,
                            n_bufs,
                            momentum,
                            MockCandleStage2Forward {
                                log: Rc::new(RefCell::new(Vec::new())),
                            },
                        )
                        .unwrap(),
                    )
                },
                make_initial: || init.clone(),
                latents_equal: tensor_f32_bit_equal,
            })
            .unwrap_or_else(|err| panic!("{name} conformance failed: {err}"));
            assert_eq!(report.total_steps, 6);
            assert!(report.verified_inert_split);
            assert!(report.verified_cancellation);
        }
    }

    #[test]
    fn bernini_candle_stage2_switches_once_at_boundary_and_carries_apg_across_split() {
        let dev = Device::Cpu;
        let boundary_ts = 875.0f64;
        let base_omegas = BerniniStage2Omegas {
            omega_vid: 1.25,
            omega_img: 4.5,
            omega_txt: 4.0,
            omega_tgt: Some(0.5),
        };
        let init = Tensor::from_vec(vec![0.5f32, -0.4, 0.8, -0.2], (4,), &dev).unwrap();
        let cancel = CancelFlag::default();

        let log_unsplit: Stage2Log = Rc::new(RefCell::new(Vec::new()));
        let drv_unsplit = BerniniDualExpertDriver::new(
            "bernini_candle_boundary_unsplit",
            Sampler::UniPC,
            boundary_ts,
            6,
            5.0,
            base_omegas,
            Defaults::OMEGA_SCALE,
            2,
            0.5,
            MockCandleStage2Forward {
                log: log_unsplit.clone(),
            },
        )
        .unwrap();
        let mut st_u = DriverRunState::new();
        let mut noop = |_: Progress| {};
        let mut hooks_u = StepHooks::new(&mut noop);
        let unsplit = run_candle_driver_slices(
            &drv_unsplit,
            [SliceSpec::full(6)],
            BoundaryState::XSigma(init.clone()),
            6,
            0,
            &cancel,
            None,
            &mut st_u,
            &mut hooks_u,
            |_, out, _| Ok(out),
        )
        .unwrap();

        let log_split: Stage2Log = Rc::new(RefCell::new(Vec::new()));
        let drv_split = BerniniDualExpertDriver::new(
            "bernini_candle_boundary_split",
            Sampler::UniPC,
            boundary_ts,
            6,
            5.0,
            base_omegas,
            Defaults::OMEGA_SCALE,
            2,
            0.5,
            MockCandleStage2Forward {
                log: log_split.clone(),
            },
        )
        .unwrap();
        let mut st_s = DriverRunState::new();
        let mut noop_s = |_: Progress| {};
        let mut hooks_s = StepHooks::new(&mut noop_s);
        let split_out = run_candle_driver_slices(
            &drv_split,
            [
                SliceSpec {
                    pass: 0,
                    phase: 0,
                    span: 0..3,
                    transition: SliceTransition::Start,
                    history: DriverHistoryPolicy::Reset,
                    geometry: Some((256, 256)),
                    request_boundary_estimate: false,
                },
                SliceSpec {
                    pass: 0,
                    phase: 1,
                    span: 3..6,
                    transition: SliceTransition::Continue,
                    history: DriverHistoryPolicy::Carry,
                    geometry: Some((256, 256)),
                    request_boundary_estimate: false,
                },
            ],
            BoundaryState::XSigma(init),
            6,
            0,
            &cancel,
            None,
            &mut st_s,
            &mut hooks_s,
            |_, out, _| Ok(out),
        )
        .unwrap();

        let (BoundaryState::XSigma(a), BoundaryState::XSigma(b)) = (unsplit, split_out) else {
            panic!("expected XSigma");
        };
        assert!(tensor_f32_bit_equal(&a, &b));

        let entries = log_split.borrow();
        assert_eq!(entries.len(), 6);
        assert_eq!(log_unsplit.borrow().as_slice(), entries.as_slice());
        for &(step, expert, t, omegas, switched, non_empty_bufs) in entries.iter() {
            assert_eq!(non_empty_bufs, 2);
            if t >= boundary_ts {
                assert_eq!(expert, BerniniExpert::High, "step {step} >= boundary");
                assert!(!switched);
                assert_eq!(omegas, base_omegas);
            } else {
                assert_eq!(expert, BerniniExpert::Low, "step {step} < boundary");
                assert!(switched);
                assert_eq!(
                    omegas.omega_vid,
                    base_omegas.omega_vid * Defaults::OMEGA_SCALE
                );
                assert_eq!(
                    omegas.omega_img,
                    base_omegas.omega_img * Defaults::OMEGA_SCALE
                );
                assert_eq!(
                    omegas.omega_txt,
                    base_omegas.omega_txt * Defaults::OMEGA_SCALE
                );
            }
        }
    }

    #[test]
    fn bernini_candle_renderer_pre_load_plan_validates_output_contracts_and_refusals() {
        let g = crate::provider_registry()
            .unwrap()
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap();

        let still_req = GenerationRequest {
            prompt: "a red panda in dappled sunlight".into(),
            width: 512,
            height: 512,
            frames: Some(1),
            steps: Some(10),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        g.validate(&still_req).unwrap();
        let still_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Candle,
            BerniniKnobs::default().to_plan_knobs(),
            &still_req,
        )
        .unwrap();
        assert_eq!(
            still_plan.output_contract,
            BerniniOutputContract::Images { expected_count: 1 }
        );

        let video_req = GenerationRequest {
            prompt: "ripples on a pond".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            fps: Some(16),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        g.validate(&video_req).unwrap();
        let video_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Candle,
            BerniniKnobs::default().to_plan_knobs(),
            &video_req,
        )
        .unwrap();
        assert_eq!(
            video_plan.output_contract,
            BerniniOutputContract::Video {
                fps: 16,
                latent_frames: 5
            }
        );

        assert!(g
            .validate(&GenerationRequest {
                scheduler_shift: Some(0.0),
                ..still_req
            })
            .is_err());
    }
}
