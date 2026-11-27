//! The native Bernini renderer provider (sc-4706): loads the converted dual-expert snapshot
//! (sc-4705) + the stock Wan2.2 UMT5/VAE/tokenizer, and runs the boundary-switched, APG-guided
//! denoise in **spatial latent space**, decoding to an image (1 frame) or video.
//!
//! Mirrors `mlx_gen_wan::model::Wan14b`'s staging (UMT5 → experts → VAE) to bound peak memory, with
//! the dual-expert CFG `denoise_moe` replaced by `denoise_bernini` (per-step
//! [`crate::forward::guided_velocity`] over the resolved [`Mode`]).

/// Metadata from the same stored configuration and quantizer passed to expert construction.
pub(crate) fn expert_weight_policy(
    config: &WanModelConfig,
    quant: Option<Quant>,
) -> serde_json::Value {
    serde_json::json!({"dense_dtype":"Bfloat16", "stored":format!("{:?}",config.quantization),
        "applied_quant_bits":quant.map(Quant::bits), "affine_group_size":mlx_gen::quant::DEFAULT_GROUP_SIZE})
}

pub(crate) fn planner_weight_policy(quant: Option<Quant>) -> serde_json::Value {
    serde_json::json!({"llm_quant_bits":quant.map(Quant::bits),
        "dense_components":["vision", "connector", "clip_diff"],
        "affine_group_size":mlx_gen::quant::DEFAULT_GROUP_SIZE})
}

use std::cell::RefCell;
use std::path::PathBuf;

use mlx_rs::memory::clear_cache;
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

use gen_core_family::bernini_plan::{
    build_bernini_render_plan, BerniniBackendKind, BerniniEntrypoint, BerniniOutputContract,
    BerniniPlanKnobs, BerniniRenderPlan, BerniniRendererGuidanceMode,
};
use mlx_gen::block_residency::BlockPlan;
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverSupport, FamilyDriver, SliceInput, SliceSpec, SliceTransition, StepHooks,
};
use mlx_gen::gen_core::{
    AdapterSurface, BoundaryState, ConditioningSurface, FamilyDriverSpec, LatentCodecDeclaration,
    ModelBinding, OutputSurface, Pass0ConditioningRequirement, PreviewSupport, ProductLimits,
    ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface, RouteBindings, RouteDefaults,
    RouteRecord, RouteStructure, SamplingSurface, SeedAndBatchSemantics, SeedScheme, SizeFloor,
    StepSupport, TextEncoderDeclaration, TextEncoderSurface, WeightPlanDeclaration,
    WeightSourceKind, WAN_Z16_VIDEO_LATENT_SPACE,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result, WeightsSource,
};
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{align_dim, decode_to_frames, frames_to_images, latent_shape};
use mlx_gen_wan::scheduler::{compute_sigmas, SolverKind, WanNativeDriver};
use mlx_gen_wan::text_encoder::{load_tokenizer, Umt5Encoder};
use mlx_gen_wan::{WanBlockStream, WanTransformer};

use crate::config::{BerniniKnobs, Defaults};
use crate::forward::{
    guided_velocity, num_momentum_buffers, vit_one_step, GuidanceParams, Mode, PackedForward,
    Trunk, VitGuidanceParams, VitMode, VitStreams,
};
use crate::guidance::MomentumBuffer;
use crate::preprocess::{encode_image, encode_videoclip};
use crate::{decoded_output_geometry, ProviderVae, PROVIDER_VAE_STRIDE};

pub const MODEL_ID: &str = "bernini_renderer";

/// Primary [`DriverSupport`] declaration for Bernini Stage 2 dual-expert denoising.
pub(crate) fn bernini_stage2_driver_support(driver_id: &'static str) -> DriverSupport {
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
                "native MLX Bernini Stage 2 driver does not re-noise from clean estimates",
            geometry_change:
                "native MLX Bernini Stage 2 driver does not support geometry-changing phase boundaries",
            read_only_boundary_estimate:
                "native MLX Bernini Stage 2 driver does not expose read-only boundary clean estimates",
            ..DriverRefusalReasons::default()
        },
    }
}

/// Primary typed route declaration for the Bernini renderer (`REG-1`, `AP5`).
pub fn route_renderer() -> RouteRecord {
    RouteRecord {
        route_id: MODEL_ID,
        family: "bernini",
        backend: "mlx",
        modality: Modality::Video,
        mac_only: true,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![TextEncoderDeclaration::new("umt5_xxl", Some(512))],
                ..Default::default()
            },
            conditioning: ConditioningSurface {
                kinds: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                    ConditioningKind::VideoClip,
                ],
                control_kinds: None,
                pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: vec!["source_id_rotary_packing"],
            },
            sampling: SamplingSurface {
                samplers: vec!["unipc"],
                schedulers: vec![],
                guidance_methods: vec![],
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: ModelBinding::family_driver(
                FamilyDriverSpec::new(bernini_stage2_driver_support("bernini_renderer"))
                    .with_expert_switching(),
            ),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("wan_z16_vae"),
                output_latent_space: Some(&WAN_Z16_VIDEO_LATENT_SPACE),
                pixel_stride: None,
                token_width: None,
                supports_diffusion_decoder: false,
            },
            adapters: AdapterSurface::unsupported(),
            residency: ResidencySurface {
                lifetime_graph: vec![
                    ResidencyPhaseNode::new(ResidencyPhaseKind::PromptEncode, ["umt5_xxl"], true),
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::Denoise,
                        ["wan2_2_high_noise_expert", "wan2_2_low_noise_expert"],
                        false,
                    ),
                    ResidencyPhaseNode::new(ResidencyPhaseKind::Decode, ["wan_z16_vae"], false),
                ],
                supports_sequential_offload: false,
                unconditionally_engages_staged_residency: true,
                supports_kv_cache: true,
                ..Default::default()
            },
            weight_plan: WeightPlanDeclaration {
                source_kind: WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
            },
            output: OutputSurface {
                preview: PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: ProductLimits {
            min_size: 16,
            max_size: 1280,
            size_floor: SizeFloor::RangeChecked,
            supported_steps: StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::BackendGlobalSeeded,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            steps: Some(Defaults::STEPS as u32),
            guidance: Some(Defaults::OMEGA_TXT),
            sampler: Some("unipc"),
            schedule: None,
            shift: Some(BerniniKnobs::default().shift),
            strength: None,
            dimensions: Some((832, 480)),
            ..Default::default()
        },
        bindings: RouteBindings::generator(descriptor, |spec| load(spec).map_err(Into::into))
            .with_footprint(component_footprint)
            .with_memory_registration(crate::memory_strategy::RENDERER_MEMORY_REGISTRATION)
            .with_memory_fixture(
                mlx_gen::gen_core::mlx_memory_contract_surface_specs,
                |spec| {
                    crate::memory_strategy::weights_free_memory_strategy_contract(
                        crate::memory_strategy::RENDERER_ID,
                        spec,
                    )
                },
            )
            .with_memory_behavior_registration(crate::memory_strategy::RENDERER_MEMORY_BEHAVIOR),
    }
}

/// Stable identity + advertised capabilities for the Bernini renderer, computed from
/// [`route_renderer`].
pub fn descriptor() -> ModelDescriptor {
    route_renderer().compute_descriptor()
}

/// The loaded Bernini renderer: resolved Wan2.2 config + Bernini knobs + the snapshot dir. The heavy
/// components are staged inside `generate_impl`.
pub struct BerniniRenderer {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    knobs: BerniniKnobs,
    root: PathBuf,
    quant: Option<Quant>,
    /// The contract this LOAD publishes, and the spec it was built from (sc-18609). Both are needed
    /// by the `Generator` memory hooks below: without them the loaded generator inherits the trait
    /// defaults and the whole declared ladder is unreachable through the production route.
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    loaded_spec: LoadSpec,
}

/// Load the Bernini renderer from a converted MLX snapshot directory
/// ([`mlx_gen_wan::convert::assemble_bernini_renderer_snapshot`] output: `low/high_noise_model` +
/// `t5_encoder` + `vae` + `tokenizer.json` + `config.json` + `bernini_renderer.json`).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "bernini_renderer: expected a model directory (converted MLX snapshot), not a single file".into(),
            ))
        }
    };
    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "bernini_renderer: config.json is not a dual-expert model (model_type={}); expected the \
             assembled Bernini renderer snapshot",
            config.model_type
        )));
    }
    let knobs = BerniniKnobs::from_dir(&root)?;
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec)?;
    Ok(Box::new(BerniniRenderer {
        descriptor: descriptor(),
        config,
        knobs,
        root,
        quant: spec.quantize,
        memory_strategy,
        loaded_spec: spec.clone(),
    }))
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split. Bernini is
/// a FLAT snapshot — the renderer components are individual root-level FILES, not diffusers subdirs: the
/// UMT5 text encoder `t5_encoder.safetensors`, the WAN dual experts `low_noise_model.safetensors` +
/// `high_noise_model.safetensors`, and the z16 VAE `vae.safetensors`. A name-guessing consumer would
/// read the encoder as ZERO (no `text_encoder/` subdir at all); this seam reports the real file bytes.
/// The full `bernini` id additionally loads the Qwen2.5-VL planner files; those are not part of the
/// TE/DiT/VAE render split (they are still in the worker's whole-model total). Shared by `bernini` +
/// `bernini_renderer`.
///
/// Both ids declare unconditional staged residency — each is structurally always-staged
/// (the encoders are dropped + `clear_cache()`d before the two co-resident experts load), and this split
/// is the staged peak the fit-gate bounds (`max(encoders, DiT+VAE)`, dominated by the experts). The
/// worker can consume that physical fact independently of the false selectable-control bit and this
/// split generically through the registered footprint seam (this crate reports the bytes; the worker
/// decides how to use them).
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["t5_encoder.safetensors"],
        &[
            "low_noise_model.safetensors",
            "high_noise_model.safetensors",
        ],
        &["vae.safetensors"],
    )
}



/// One expert (high or low) with its prepared per-expert cross-attention K/V for the cond / empty-neg
/// text contexts (text embedding is per-expert, so K/V is built per expert).
struct BExpert<'a> {
    trunk: Trunk<'a>,
    cross_kv_cond: Vec<(Array, Array)>,
    cross_kv_uncond: Vec<(Array, Array)>,
}

impl<'a> BExpert<'a> {
    /// Build this expert's cross-attention K/V.
    ///
    /// On a rung-4 (windowed) trunk the DiT holds no blocks, so `prepare_cross_kv` cannot serve the
    /// cache from it; the stream walks the same plan once here instead. The caches are small
    /// (`[B, n, text_len, d]` per block) and are reused across every step and every guidance pass, so
    /// they stay resident deliberately — re-deriving them per window per forward would multiply a
    /// once-per-generate cost by the pass count for no residency saving.
    fn build(trunk: Trunk<'a>, context: &Array, context_null: &Array) -> Result<Self> {
        let dit = trunk.dit;
        let cc = dit.embed_text(context)?;
        let cu = dit.embed_text(context_null)?;
        let prepare = |embedded: &Array| -> Result<Vec<(Array, Array)>> {
            match trunk.window {
                None => dit.prepare_cross_kv(embedded),
                Some((stream, plan, cancel)) => {
                    stream.prepare_cross_kv_windowed(embedded, plan, cancel)
                }
            }
        };
        Ok(Self {
            cross_kv_cond: prepare(&cc)?,
            cross_kv_uncond: prepare(&cu)?,
            trunk,
        })
    }
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

/// Active dual-expert branch for Bernini Stage 2 DiT denoising (`High` for `t >= boundary`,
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
    pub fn from_guidance(g: &GuidanceParams) -> Self {
        Self {
            omega_vid: g.omega_vid,
            omega_img: g.omega_img,
            omega_txt: g.omega_txt,
            omega_tgt: None,
        }
    }

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
        latent: &Array,
        step: usize,
        timestep: f32,
        sigma: f32,
        omegas: BerniniStage2Omegas,
        mbufs: &mut [MomentumBuffer],
    ) -> Result<Array>;
}

struct BerniniStage2History {
    total_steps: usize,
    boundary_bits: u32,
    omega_scale_bits: u32,
    shift_bits: u32,
    num_train_timesteps: usize,
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

/// Typed [`FamilyDriver`] for Bernini Stage 2 dual-expert UniPC denoising across both
/// `bernini_renderer` (`Mode` + APG momentum buffers) and full `bernini` (`VitMode`, 4-stream ViT
/// guidance) (`PIPE-7`, `DEC-13`).
///
/// Delegates UniPC multistep integration and MLX per-step evaluation to
/// [`WanNativeDriver`] while persisting the one-time `omega_scale` boundary latch and APG momentum
/// buffers inside [`DriverRunState`].
pub struct BerniniDualExpertDriver<F> {
    boundary: f32,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    sigmas: Vec<f32>,
    base_omegas: BerniniStage2Omegas,
    omega_scale: f32,
    num_momentum_buffers: usize,
    momentum: f32,
    forward: F,
    support: DriverSupport,
}

impl<F: BerniniStage2Forward> BerniniDualExpertDriver<F> {
    fn build_support(driver_id: &'static str) -> DriverSupport {
        bernini_stage2_driver_support(driver_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        driver_id: &'static str,
        boundary: f32,
        num_train_timesteps: usize,
        steps: usize,
        shift: f32,
        base_omegas: BerniniStage2Omegas,
        omega_scale: f32,
        num_momentum_buffers: usize,
        momentum: f32,
        forward: F,
    ) -> Result<Self> {
        if steps == 0 {
            return Err(Error::Msg(format!(
                "{driver_id}: Stage 2 steps must be >= 1"
            )));
        }
        let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
        Ok(Self {
            boundary,
            num_train_timesteps,
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

impl<F: BerniniStage2Forward> FamilyDriver<Array> for BerniniDualExpertDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps != self.steps || input.span.end > self.steps {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Bernini Stage 2 steps {}",
                self.support.driver_id, input.span, input.total_steps, self.steps
            )));
        }

        let carrying = input.transition == SliceTransition::Continue
            && input.history == DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<BerniniStage2History>() else {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry requested without prior Bernini Stage 2 history in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.total_steps != input.total_steps
                || existing.boundary_bits != self.boundary.to_bits()
                || existing.omega_scale_bits != self.omega_scale.to_bits()
                || existing.shift_bits != self.shift.to_bits()
                || existing.num_train_timesteps != self.num_train_timesteps
                || existing.base_omegas_bits != self.base_omegas.bits()
                || existing.num_momentum_buffers != self.num_momentum_buffers
                || existing.momentum_bits != self.momentum.to_bits()
            {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible Bernini Stage 2 schedule or guidance settings",
                    self.support.driver_id
                )));
            }
            if existing.expected_next_step != input.span.start {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
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
                total_steps: input.total_steps,
                boundary_bits: self.boundary.to_bits(),
                omega_scale_bits: self.omega_scale.to_bits(),
                shift_bits: self.shift.to_bits(),
                num_train_timesteps: self.num_train_timesteps,
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
                SolverKind::UniPC,
                self.num_train_timesteps,
                self.steps,
                self.shift,
                |latent: &Array, step: usize, t: f32| {
                    let mut st = active_cell.borrow_mut();
                    let expert = if t >= self.boundary {
                        BerniniExpert::High
                    } else {
                        if !st.switched {
                            st.switched = true;
                            st.active_omegas.scale_once(self.omega_scale);
                        }
                        BerniniExpert::Low
                    };
                    let omegas = st.active_omegas;
                    let sigma = self.sigmas[step];
                    self.forward.predict_step_velocity(
                        expert,
                        latent,
                        step,
                        t,
                        sigma,
                        omegas,
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
fn run_bernini_mlx_stage2_driver<F: BerniniStage2Forward>(
    driver_id: &'static str,
    boundary: f32,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    base_omegas: BerniniStage2Omegas,
    omega_scale: f32,
    num_momentum_buffers: usize,
    momentum: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
    forward: F,
) -> Result<Array> {
    let driver = BerniniDualExpertDriver::new(
        driver_id,
        boundary,
        num_train_timesteps,
        steps,
        shift,
        base_omegas,
        omega_scale,
        num_momentum_buffers,
        momentum,
        forward,
    )?;
    let mut state = DriverRunState::new();
    let mut progress_adapter = |progress: Progress| {
        if let Progress::Step { current, .. } = progress {
            on_step(current as usize);
        }
    };
    let mut hooks = StepHooks::new(&mut progress_adapter);
    let out = run_mlx_driver_slices(
        &driver,
        [SliceSpec::full(steps)],
        BoundaryState::XSigma(init_noise.clone()),
        steps,
        0,
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

/// The boundary-switched, APG-guided denoise loop (the Bernini analog of
/// `mlx_gen_wan::pipeline::denoise_moe`). Runs in **spatial latent space** `[16, T, H8, W8]`: each
/// step picks the high-noise expert while `t ≥ boundary` and the low-noise expert below it, multiplies
/// all omegas by `omega_scale` once on the first low-noise step, computes the per-mode guided velocity
/// (sigma = this step's flow sigma, for the APG x-conversion), and applies the UniPC flow step.
#[allow(clippy::too_many_arguments)]
fn denoise_bernini(
    pf: &PackedForward,
    mode: Mode,
    low: &BExpert,
    high: &BExpert,
    boundary: f32,
    num_train: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    videos: &[Array],
    images: &[Array],
    base_g: &GuidanceParams,
    omega_scale: f32,
    momentum: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let sigmas = compute_sigmas(steps, shift, num_train);
    mlx_gen::video_execution::sampling(
        "bernini_renderer",
        "unipc",
        "flow_velocity_apg",
        serde_json::json!({"omega_scale_bits":omega_scale.to_bits(),"guidance":format!("{base_g:?}"),"mode":format!("{mode:?}"),"momentum_bits":momentum.to_bits(),"boundary_bits":boundary.to_bits()}),
    );
    mlx_gen::video_execution::schedule("bernini_renderer", &sigmas);

    struct RendererForward<'a> {
        pf: &'a PackedForward,
        mode: Mode,
        low: &'a BExpert<'a>,
        high: &'a BExpert<'a>,
        videos: &'a [Array],
        images: &'a [Array],
        base_g: &'a GuidanceParams,
    }

    impl BerniniStage2Forward for RendererForward<'_> {
        fn predict_step_velocity(
            &self,
            expert: BerniniExpert,
            latent: &Array,
            _step: usize,
            timestep: f32,
            sigma: f32,
            omegas: BerniniStage2Omegas,
            mbufs: &mut [MomentumBuffer],
        ) -> Result<Array> {
            let ex = match expert {
                BerniniExpert::High => self.high,
                BerniniExpert::Low => self.low,
            };
            let g = omegas.to_guidance(self.base_g);
            guided_velocity(
                self.pf,
                self.mode,
                ex.trunk,
                latent,
                self.videos,
                self.images,
                timestep,
                sigma,
                &ex.cross_kv_cond,
                &ex.cross_kv_uncond,
                &g,
                mbufs,
            )
        }
    }

    run_bernini_mlx_stage2_driver(
        "bernini_mlx_renderer_stage2_driver",
        boundary,
        num_train,
        steps,
        shift,
        init_noise,
        BerniniStage2Omegas::from_guidance(base_g),
        omega_scale,
        num_momentum_buffers(mode),
        momentum,
        cancel,
        on_step,
        RendererForward {
            pf,
            mode,
            low,
            high,
            videos,
            images,
            base_g,
        },
    )
}

/// One expert (high or low) with its prepared cross-attention K/V for the planner's **4** prompt-embed
/// streams (the full-Bernini ViT-conditioned path). Each stream is `concat_with_zero_init(UMT5(prompt),
/// planner ViT-context)` (sc-5140), in renderer `text_dim` space, so it goes through the same
/// `embed_text` → `prepare_cross_kv` as the renderer's text context.
pub struct BVitExpert<'a> {
    trunk: Trunk<'a>,
    wtxt_wvit: Vec<(Array, Array)>,
    wtxt_wovit: Vec<(Array, Array)>,
    wotxt_wvit: Vec<(Array, Array)>,
    wotxt_wovit: Vec<(Array, Array)>,
}

impl<'a> BVitExpert<'a> {
    /// `streams` = `[wtxt_wvit, wtxt_wovit, wotxt_wvit, wotxt_wovit]` prompt-embed contexts.
    pub fn build(trunk: Trunk<'a>, streams: [&Array; 4]) -> Result<Self> {
        let dit = trunk.dit;
        // See `BExpert::build` for why a windowed trunk builds its cross-K/V through the plan.
        let prep = |s: &Array| -> Result<Vec<(Array, Array)>> {
            let embedded = dit.embed_text(s)?;
            match trunk.window {
                None => dit.prepare_cross_kv(&embedded),
                Some((stream, plan, cancel)) => {
                    stream.prepare_cross_kv_windowed(&embedded, plan, cancel)
                }
            }
        };
        Ok(Self {
            wtxt_wvit: prep(streams[0])?,
            wtxt_wovit: prep(streams[1])?,
            wotxt_wvit: prep(streams[2])?,
            wotxt_wovit: prep(streams[3])?,
            trunk,
        })
    }

    fn streams(&self) -> VitStreams<'_> {
        VitStreams {
            wtxt_wvit: &self.wtxt_wvit,
            wtxt_wovit: &self.wtxt_wovit,
            wotxt_wvit: &self.wotxt_wvit,
            wotxt_wovit: &self.wotxt_wovit,
        }
    }
}

/// The full-Bernini ViT-conditioned denoise loop (`sample_bernini_wvitcfg`, `wan_diffusion.py`
/// 571-793) — the renderer-side compute that consumes the planner's 4 prompt streams. The boundary-
/// switched, [`vit_one_step`]-guided analog of `denoise_bernini`: each step picks the expert by the
/// `switch_dit_boundary`, multiplies **all four** omegas (incl. `omega_tgt`) by `omega_scale` once on
/// the first low-noise step, and applies the UniPC flow step. Runs in spatial latent space
/// `[16, T, H8, W8]`. The full end-to-end string-up (planner MAR loop → these 4 streams → here →
/// decode) is the registered pipeline in sc-5145.
#[allow(clippy::too_many_arguments)]
pub fn denoise_bernini_wvitcfg(
    pf: &PackedForward,
    mode: VitMode,
    low: &BVitExpert,
    high: &BVitExpert,
    boundary: f32,
    num_train: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    images: &[(Array, f64)],
    videos: &[(Array, f64)],
    base_g: &VitGuidanceParams,
    omega_scale: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let sigmas = compute_sigmas(steps, shift, num_train);
    mlx_gen::video_execution::sampling(
        "bernini_renderer",
        "unipc",
        "flow_velocity_apg",
        serde_json::json!({"omega_scale_bits":omega_scale.to_bits(),"guidance":format!("{base_g:?}"),"mode":format!("{mode:?}"),"boundary_bits":boundary.to_bits()}),
    );
    mlx_gen::video_execution::schedule("bernini_renderer", &sigmas);

    struct VitRendererForward<'a> {
        pf: &'a PackedForward,
        mode: VitMode,
        low: &'a BVitExpert<'a>,
        high: &'a BVitExpert<'a>,
        images: &'a [(Array, f64)],
        videos: &'a [(Array, f64)],
        base_g: &'a VitGuidanceParams,
    }

    impl BerniniStage2Forward for VitRendererForward<'_> {
        fn predict_step_velocity(
            &self,
            expert: BerniniExpert,
            latent: &Array,
            _step: usize,
            timestep: f32,
            sigma: f32,
            omegas: BerniniStage2Omegas,
            _mbufs: &mut [MomentumBuffer],
        ) -> Result<Array> {
            let ex = match expert {
                BerniniExpert::High => self.high,
                BerniniExpert::Low => self.low,
            };
            let g = omegas.to_vit(self.base_g);
            vit_one_step(
                self.pf,
                ex.trunk,
                self.mode,
                latent,
                self.images,
                self.videos,
                timestep,
                sigma,
                &ex.streams(),
                &g,
            )
        }
    }

    run_bernini_mlx_stage2_driver(
        "bernini_mlx_vit_stage2_driver",
        boundary,
        num_train,
        steps,
        shift,
        init_noise,
        BerniniStage2Omegas::from_vit(base_g),
        omega_scale,
        0,
        0.0,
        cancel,
        on_step,
        VitRendererForward {
            pf,
            mode,
            low,
            high,
            images,
            videos,
            base_g,
        },
    )
}

// Hand-written rather than `mlx_gen::impl_generator!` (sc-18609). That macro emits only
// `descriptor`/`validate`/`generate`, so a provider using it inherits the `Generator` memory-strategy
// defaults: contract `None`, a safety check that rejects EVERY optimized strategy, and a request
// scope that returns `Ok(None)`. `memory_strategy.rs` has declared the full five-rung ladder since
// sc-15528, but until this impl existed the production route reached none of it — the registry
// declaration was real and the loaded generator was silent about it. Same reason chroma spells its
// `Generator` out by hand (SC-15520).
impl BerniniRenderer {
    fn generate_execution_attempt(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl Generator for BerniniRenderer {
    fn generate_with_execution_report(
        &self,
        req: &mlx_gen::GenerationRequest,
        on_progress: &mut dyn FnMut(mlx_gen::Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::video_execution::complete(self.descriptor(), &req.cancel, || {
            self.generate_execution_attempt(req, &mut |event| {
                mlx_gen::video_execution::without_capture(|| on_progress(event));
            })
            .map(mlx_gen::gen_core::GenerationReport::from_output)
        })
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        mlx_gen::video_execution::without_capture(|| {
            self.generate_execution_attempt(req, on_progress)
        })
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::loaded_safety_check(
            MODEL_ID,
            &self.loaded_spec,
            &self.memory_strategy,
            context,
            self.config.num_layers,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy::loaded_begin_request(
            MODEL_ID,
            &self.loaded_spec,
            &self.memory_strategy,
            context,
            self.config.num_layers,
        )
    }
}

impl BerniniRenderer {
    /// Build and validate the pre-load [`BerniniRenderPlan`] before loading weights (`DEC-3`).
    pub fn render_plan(&self, req: &GenerationRequest) -> Result<BerniniRenderPlan> {
        build_bernini_render_plan(
            &self.descriptor,
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Mlx,
            self.knobs.to_plan_knobs(),
            req,
        )
        .map_err(Into::into)
    }

    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.render_plan(req).map(|_| ())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let render_plan = self.render_plan(req)?;
        let cfg = &self.config;
        let k = &self.knobs;

        // --- Resolve geometry + knobs ---
        let frames = render_plan.frames;
        // sc-12500 (F-040): `validate_impl` rejects any off-grid width/height, so the reference's
        // align-down is a no-op here — assert that instead of silently refitting the request
        // (1000×1000 used to render 992×992 with no diagnostic while candle errored).
        let width = align_dim(render_plan.width, cfg.patch_size.2, PROVIDER_VAE_STRIDE.2);
        let height = align_dim(render_plan.height, cfg.patch_size.1, PROVIDER_VAE_STRIDE.1);
        debug_assert_eq!(
            (width, height),
            (render_plan.width, render_plan.height),
            "bernini_renderer: off-grid request survived validate"
        );
        let steps = render_plan.steps;
        let seed = render_plan.seed.unwrap_or(42);
        let neg = req.negative_prompt.clone().unwrap_or_default();

        // Conditioning split: VideoClip → videos, Reference/MultiReference → images.
        let has_video = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::VideoClip { .. }));
        let has_image = req.conditioning.iter().any(|c| {
            matches!(
                c,
                Conditioning::Reference { .. } | Conditioning::MultiReference { .. }
            )
        });
        let mode = renderer_mode_from_plan(
            render_plan
                .renderer_mode
                .expect("Renderer plan always resolves a guidance mode"),
        );

        let base_g = GuidanceParams {
            omega_vid: render_plan.omega_vid,
            omega_img: render_plan.omega_img,
            omega_txt: render_plan.omega_txt,
            eta: render_plan.eta,
            norm_threshold: [Defaults::NORM_THRESHOLD, Defaults::NORM_THRESHOLD],
        };

        let lat = latent_shape(frames, height, width, cfg.vae_z_dim, PROVIDER_VAE_STRIDE)?;

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            let (context, context_null) = mlx_gen::gen_core::with_prompt_encoding(
                on_progress,
                || -> std::result::Result<_, mlx_gen::Error> {
                    let context = enc.encode(&tokenizer, &req.prompt)?;
                    let context_null = enc.encode(&tokenizer, &neg)?;
                    eval([&context, &context_null])?;
                    Ok((context, context_null))
                },
            )?;
            (context, context_null)
        };
        // Residency discipline (epic 10834, sc-10840): the UMT5-XXL text encoder + its `Weights` are
        // dropped at the block close above and `context`/`context_null` are materialized — flush the
        // freed encoder pages back to the OS NOW, before the source-VAE encode and the two ~28 GB expert
        // loads, so they do not linger in MLX's buffer cache and inflate the resident set. The live
        // conditioning arrays are untouched, so this is byte-identical (a pure memory hint). Mirrors the
        // `bernini` id's two encoder-drop flushes (bernini.rs).
        clear_cache();
        // F-135: T5 encode materialized above; honor a cancel before the source VAE encodes and the
        // two ~28 GB expert loads (all pre-denoise stages sc-9093's loop-only cancel didn't reach).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- Stage 1b: VAE-encode source media → conditioning latents (→ encoder freed) ---
        let (videos, images) = if has_video || has_image {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = ProviderVae::from_weights(&w)?;
            let mut videos = Vec::new();
            let mut images = Vec::new();
            for c in &req.conditioning {
                match c {
                    Conditioning::VideoClip { frames, .. } => {
                        videos.push(encode_videoclip(&vae, frames, width, height)?)
                    }
                    Conditioning::Reference { image, .. } => {
                        images.push(encode_image(&vae, image, width, height)?)
                    }
                    Conditioning::MultiReference { images: imgs } => {
                        for im in imgs {
                            images.push(encode_image(&vae, im, width, height)?);
                        }
                    }
                    _ => {}
                }
            }
            let all: Vec<&Array> = videos.iter().chain(images.iter()).collect();
            if !all.is_empty() {
                eval(all)?;
            }
            (videos, images)
        } else {
            (Vec::new(), Vec::new())
        };
        // Residency discipline (sc-10840): the Stage-1b source-VAE encoder + its `Weights` are dropped at
        // the block close above and the `videos`/`images` conditioning latents are materialized — flush
        // the freed VAE-encoder pages before the two ~28 GB expert loads (a fresh VAE is reloaded for the
        // Stage-3 decode). Live latents untouched, byte-identical — the second of the two encoder-drop
        // boundaries, matching the `bernini` id.
        clear_cache();
        // F-135: source encodes materialized (if any); honor a cancel before the expert loads.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // Seeded init noise (spatial, f32). Bit-parity vs torch needs the reference's CPU-MT19937
        // draw injected; the coherence bar uses the MLX RNG.
        let key = random::key(seed)?;
        mlx_gen::video_execution::draw(0, "bernini_renderer_initial_normal", seed);
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 2: load both experts, APG denoise (→ experts freed) ---
        // Load+quantize each expert before loading the next so only one bf16 transient is resident at
        // a time (sc-5360 — `WanTransformer::quantize` eval-frees its bf16 dequant). Without quant this
        // just loads both bf16.
        // ── Ladder rungs 3 + 4 (sc-15528) ────────────────────────────────────────────────────
        // Rung 4 is per-request: when a window is selected the expert's block stack is NEVER
        // materialized (`from_weights_deferred` holds zero blocks) and the stream rebuilds
        // `plan.window()` blocks at a time. On a DUAL-expert config that is the whole point --
        // sc-16354's finding is that a naive per-expert window leaves the idle expert's 40 blocks
        // resident, and a stack that was never materialized has no idle half to pay for. Both
        // experts are deferred here, so the bound is over the full 80-block trunk, not one expert.
        let window = crate::memory_strategy::transformer_window_size(req)?;
        let attn_budget = crate::memory_strategy::attention_budget(req);
        mlx_gen::video_execution::policy(
            "bernini_experts",
            serde_json::json!({"window":window,"attention_budget":format!("{attn_budget:?}")}),
        );
        let plan = window
            .map(|size| BlockPlan::new(cfg.num_layers, size))
            .transpose()?;
        let load_expert = |name: &str| -> Result<(WanTransformer, Option<WanBlockStream>)> {
            let path = self.root.join(name);
            let w = Weights::from_file(&path)?;
            mlx_gen::video_execution::policy(
                "bernini_expert_weights",
                expert_weight_policy(cfg, self.quant),
            );
            if window.is_some() {
                let dit = WanTransformer::from_weights_deferred(&w, cfg)?;
                // No adapters can reach here (`supports_lora: false`), and the stream refuses an
                // adapted load anyway -- Wan MERGES deltas at load, so a streamed block re-read from
                // the snapshot would silently carry none of them.
                let mut stream = WanBlockStream::new(WeightsSource::File(path), cfg.clone(), &[])?;
                if let Some(q) = self.quant {
                    // Replayed per materialized block so the streamed weights are byte-identical to
                    // the resident ones. A pre-packed tier takes this through `cfg.quantization`
                    // inside the block constructor instead, exactly as the resident path does.
                    stream.set_quant_bits(q.bits());
                }
                stream.set_attention_budget(attn_budget);
                Ok((dit, Some(stream)))
            } else {
                let mut dit = WanTransformer::from_weights(&w, cfg)?;
                if let Some(q) = self.quant {
                    dit.quantize(q.bits(), None)?;
                }
                dit.set_attention_budget(attn_budget);
                Ok((dit, None))
            }
        };
        let latents = {
            let (low_dit, low_stream) = load_expert("low_noise_model.safetensors")?;
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let (high_dit, high_stream) = load_expert("high_noise_model.safetensors")?;
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let low = BExpert::build(
                Trunk::for_load(&low_dit, low_stream.as_ref(), plan.as_ref(), &req.cancel),
                &context,
                &context_null,
            )?;
            let high = BExpert::build(
                Trunk::for_load(&high_dit, high_stream.as_ref(), plan.as_ref(), &req.cancel),
                &context,
                &context_null,
            )?;
            let pf = PackedForward::new(
                cfg.dim / cfg.num_heads,
                cfg.out_dim,
                cfg.patch_size,
                k.max_trained_src_id,
                k.interpolate_src_id,
            );
            let boundary = render_plan.switch_dit_boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            // `denoise_bernini` reports 1-based steps, so `current` runs exactly 1..=total (F-038).
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_bernini(
                &pf,
                mode,
                &low,
                &high,
                boundary,
                cfg.num_train_timesteps,
                steps,
                render_plan.shift as f32,
                &init_noise,
                &videos,
                &images,
                &base_g,
                render_plan.omega_scale,
                Defaults::MOMENTUM,
                &req.cancel,
                &mut on_step,
            )?
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        let (out_frames, out_height, out_width) = decoded_output_geometry(lat[1], lat[2], lat[3])?;
        // Ladder rung 2 (sc-15528) — see the note on the full pipeline's decode.
        let tiling = match crate::memory_strategy::decode_tiling(req)? {
            Some(explicit) => Some(explicit),
            None => TilingConfig::auto(out_height, out_width, out_frames),
        };
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = ProviderVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref(), Some(&req.cancel))?
        };
        let images_out = frames_to_images(&frames_u8)?;

        // num_frames == 1 ⇒ a still image (t2i/i2i). A single latent frame (T_lat = 1) still decodes
        // to one VAE temporal chunk (the causal decode emits `vae_stride_t` near-identical frames);
        // the still image is the first of them, matching the reference's single-frame PNG.
        match render_plan.output_contract {
            BerniniOutputContract::Images { .. } => {
                let first = images_out.into_iter().next().ok_or_else(|| {
                    Error::Msg("bernini_renderer: VAE decode produced no frames".into())
                })?;
                Ok(GenerationOutput::Images(vec![first]))
            }
            BerniniOutputContract::Video { fps, .. } => Ok(GenerationOutput::Video {
                frames: images_out,
                fps,
                audio: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn execution_report_weight_tiers_distinguish_dense_and_affine_defaults() {
        let cfg = tiny_cfg();
        let dense = super::expert_weight_policy(&cfg, None);
        let q4 = super::expert_weight_policy(&cfg, Some(mlx_gen::Quant::Q4));
        let q8 = super::expert_weight_policy(&cfg, Some(mlx_gen::Quant::Q8));
        assert!(dense["applied_quant_bits"].is_null());
        assert_eq!(q4["applied_quant_bits"], 4);
        assert_eq!(q8["applied_quant_bits"], 8);
        assert_ne!(dense, q4);
        assert_ne!(q4, q8);
        let mut packed = cfg.clone();
        packed.quantization = Some(mlx_gen_wan::config::WanQuant {
            bits: 4,
            group_size: 32,
        });
        let stored_q4 = super::expert_weight_policy(&packed, None);
        assert_ne!(stored_q4, dense);
        assert_ne!(stored_q4, q4);
        assert!(stored_q4["applied_quant_bits"].is_null());
        assert_eq!(
            super::planner_weight_policy(None)["llm_quant_bits"],
            serde_json::Value::Null
        );
        assert_ne!(
            super::planner_weight_policy(Some(mlx_gen::Quant::Q4)),
            super::planner_weight_policy(Some(mlx_gen::Quant::Q8))
        );
        assert_eq!(
            super::planner_weight_policy(Some(mlx_gen::Quant::Q4))["dense_components"],
            serde_json::json!(["vision", "connector", "clip_diff"])
        );
    }

    use super::*;
    use mlx_rs::ops::multiply;

    /// Component residency (epic 10834, sc-10840): the renderer is structurally always-staged —
    /// `generate_impl` drops the UMT5 text encoder and the
    /// source-VAE encoder (each + `clear_cache()`) before loading the two co-resident MoE experts, so
    /// peak unified memory is already bounded to the dominant expert phase (the footprint's DiT split).
    #[test]
    fn declares_unconditional_staged_residency_without_a_selectable_control() {
        let capabilities = descriptor().capabilities;
        assert!(!capabilities.supports_sequential_offload);
        assert!(capabilities.unconditionally_engages_staged_residency);
        assert_eq!(
            capabilities.staged_residency_availability(),
            mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged,
        );
    }

    fn tiny_cfg() -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_1_3b();
        c.dim = 128;
        c.num_heads = 1;
        c.num_layers = 2;
        c.ffn_dim = 256;
        c.freq_dim = 256;
        c.text_dim = 32;
        c.text_len = 8;
        c.in_dim = 16;
        c.out_dim = 16;
        c.vae_z_dim = 16;
        c.boundary = 0.875;
        c.num_train_timesteps = 1000;
        c
    }

    fn load(name: &str) -> Weights {
        let path = format!(
            "{}/../mlx-gen-wan/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        Weights::from_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// The ViT-conditioned denoise loop runs end-to-end over a tiny dual-expert (crossing the
    /// boundary so the `omega_scale` switch fires) and preserves the spatial latent shape. Pins the
    /// loop plumbing (scheduler / expert switch / per-step `vit_one_step` / flow step); the per-step
    /// math is validated in slices A/B and the e2e coherence in sc-5145.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn denoise_wvitcfg_runs_and_keeps_shape() {
        let w = load("s5_low.safetensors");
        let cfg = tiny_cfg();
        let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");
        let pf = PackedForward::new(
            cfg.dim / cfg.num_heads,
            cfg.out_dim,
            cfg.patch_size,
            5.0,
            true,
        );
        let noisy = w.require("init_noise").unwrap(); // [16, 2, 2, 2]
        let cc = w.require("ctx_cond").unwrap();
        let cu = w.require("ctx_uncond").unwrap();
        let scale = |a: &Array, s: f32| multiply(a, Array::from_f32(s)).unwrap();
        // 4 distinct prompt streams.
        let (s0, s1, s2, s3) = (
            scale(cc, 1.0),
            scale(cu, 1.0),
            scale(cc, 0.5),
            scale(cu, 0.5),
        );
        let streams = [&s0, &s1, &s2, &s3];
        let low = BVitExpert::build(Trunk::resident(&dit), streams).expect("low expert");
        let high = BVitExpert::build(Trunk::resident(&dit), streams).expect("high expert");
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 3.0,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let mut on_step = |_i: usize| {};
        let out = denoise_bernini_wvitcfg(
            &pf,
            VitMode::VaeTxtVitWapg,
            &low,
            &high,
            0.875 * cfg.num_train_timesteps as f32, // boundary 875 → crosses with 3 steps
            cfg.num_train_timesteps,
            3,
            5.0,
            noisy,
            &[],
            &[],
            &g,
            0.8,
            &CancelFlag::default(),
            &mut on_step,
        )
        .expect("denoise");
        assert_eq!(
            out.shape(),
            noisy.shape(),
            "loop preserves spatial latent shape"
        );
        crate::model_inference_device::record_model_inference_device();
        println!(
            "MODEL_INFERENCE_BODY_COMPLETED pipeline::tests::denoise_wvitcfg_runs_and_keeps_shape"
        );
    }
}

#[cfg(test)]
mod driver_tests {
    use super::*;
    use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};
    use std::cell::RefCell;
    use std::rc::Rc;

    type Stage2Log = Rc<RefCell<Vec<(usize, BerniniExpert, f32, BerniniStage2Omegas, usize)>>>;

    struct MockStage2Forward {
        log: Stage2Log,
    }

    impl BerniniStage2Forward for MockStage2Forward {
        fn predict_step_velocity(
            &self,
            expert: BerniniExpert,
            latent: &Array,
            step: usize,
            timestep: f32,
            sigma: f32,
            omegas: BerniniStage2Omegas,
            mbufs: &mut [MomentumBuffer],
        ) -> Result<Array> {
            let mut non_empty_mbufs = 0usize;
            for buf in mbufs.iter_mut() {
                let diff = mlx_rs::ops::multiply(latent, Array::from_f32(0.05))?;
                let _ = buf.update(&mlx_gen::MlxLatentOps, &diff)?;
                if !buf.is_empty() {
                    non_empty_mbufs += 1;
                }
            }
            self.log
                .borrow_mut()
                .push((step, expert, timestep, omegas, non_empty_mbufs));
            let expert_gain = match expert {
                BerniniExpert::High => -0.6,
                BerniniExpert::Low => -0.35,
            };
            let omega_sum = omegas.omega_txt
                + omegas.omega_img
                + omegas.omega_vid
                + omegas.omega_tgt.unwrap_or(0.0);
            mlx_rs::ops::add(
                &mlx_rs::ops::multiply(latent, Array::from_f32(expert_gain - 0.05 * sigma))?,
                Array::from_f32(0.002 * omega_sum + 0.0001 * timestep),
            )
            .map_err(Into::into)
        }
    }

    fn mlx_f32_bit_equal(a: &Array, b: &Array) -> bool {
        let af = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let bf = b.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&af, &bf]).unwrap();
        af.shape() == bf.shape()
            && af
                .as_slice::<f32>()
                .iter()
                .zip(bf.as_slice::<f32>().iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn bernini_mlx_dual_expert_driver_passes_conformance_across_boundary() {
        let init = Array::from_slice(&[0.5f32, -0.4, 0.8, -0.2], &[4]);
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

        for (name, omegas, n_bufs, momentum) in [
            ("bernini_mlx_stage2_apg", renderer_omegas, 2usize, 0.5f32),
            ("bernini_mlx_stage2_vit", vit_omegas, 0usize, 0.0f32),
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
                            875.0,
                            1000,
                            6,
                            5.0,
                            omegas,
                            Defaults::OMEGA_SCALE,
                            n_bufs,
                            momentum,
                            MockStage2Forward {
                                log: Rc::new(RefCell::new(Vec::new())),
                            },
                        )
                        .unwrap(),
                    )
                },
                make_initial: || init.clone(),
                latents_equal: mlx_f32_bit_equal,
            })
            .unwrap_or_else(|err| panic!("{name} conformance failed: {err}"));
            assert_eq!(report.total_steps, 6);
            assert!(report.verified_inert_split);
            assert!(report.verified_cancellation);
        }
    }

    #[test]
    fn bernini_mlx_stage2_switches_once_at_boundary_and_carries_apg_across_split() {
        let boundary = 875.0f32;
        let base_omegas = BerniniStage2Omegas {
            omega_vid: 1.25,
            omega_img: 4.5,
            omega_txt: 4.0,
            omega_tgt: Some(0.5),
        };
        let init = Array::from_slice(&[0.5f32, -0.4, 0.8, -0.2], &[4]);
        let cancel = CancelFlag::default();

        // Find step crossing with 6 steps, shift 5.0 so the split lands right across boundary.
        let sigmas = compute_sigmas(6, 5.0, 1000);
        assert!(sigmas[2] * 1000.0 >= boundary);
        assert!(sigmas[3] * 1000.0 < boundary);

        let log_unsplit: Stage2Log = Rc::new(RefCell::new(Vec::new()));
        let drv_unsplit = BerniniDualExpertDriver::new(
            "bernini_mlx_boundary_unsplit",
            boundary,
            1000,
            6,
            5.0,
            base_omegas,
            Defaults::OMEGA_SCALE,
            2,
            0.5,
            MockStage2Forward {
                log: log_unsplit.clone(),
            },
        )
        .unwrap();
        let mut st_u = DriverRunState::new();
        let mut noop = |_: Progress| {};
        let mut hooks_u = StepHooks::new(&mut noop);
        let unsplit = run_mlx_driver_slices(
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
            "bernini_mlx_boundary_split",
            boundary,
            1000,
            6,
            5.0,
            base_omegas,
            Defaults::OMEGA_SCALE,
            2,
            0.5,
            MockStage2Forward {
                log: log_split.clone(),
            },
        )
        .unwrap();
        let mut st_s = DriverRunState::new();
        let mut noop_s = |_: Progress| {};
        let mut hooks_s = StepHooks::new(&mut noop_s);
        let split_out = run_mlx_driver_slices(
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
        assert!(mlx_f32_bit_equal(&a, &b));

        let entries = log_split.borrow();
        assert_eq!(entries.len(), 6);
        assert_eq!(log_unsplit.borrow().as_slice(), entries.as_slice());
        // Steps 0..=2 use High expert and unscaled base omegas; steps 3..=5 use Low and scale ONCE by 0.8.
        for &(step, expert, t, omegas, non_empty_bufs) in entries.iter() {
            assert_eq!(non_empty_bufs, 2);
            if t >= boundary {
                assert_eq!(expert, BerniniExpert::High, "step {step} >= boundary");
                assert_eq!(omegas, base_omegas);
            } else {
                assert_eq!(expert, BerniniExpert::Low, "step {step} < boundary");
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
                assert_eq!(
                    omegas.omega_tgt,
                    Some(base_omegas.omega_tgt.unwrap() * Defaults::OMEGA_SCALE)
                );
            }
        }
    }

    #[test]
    fn bernini_mlx_renderer_pre_load_plan_validates_output_contracts_and_refusals() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"model_type": "t2v", "dim": 5120, "dual_model": true}"#,
        )
        .unwrap();
        let g = crate::provider_registry()
            .unwrap()
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf())),
            )
            .unwrap();

        let still_req = GenerationRequest {
            prompt: "a lantern in the mist".into(),
            width: 512,
            height: 512,
            frames: Some(1),
            steps: Some(8),
            ..Default::default()
        };
        g.validate(&still_req).unwrap();
        let still_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Mlx,
            BerniniKnobs::default().to_plan_knobs(),
            &still_req,
        )
        .unwrap();
        assert_eq!(
            still_plan.output_contract,
            BerniniOutputContract::Images { expected_count: 1 }
        );

        let video_req = GenerationRequest {
            prompt: "a lantern drifting on water".into(),
            width: 512,
            height: 512,
            frames: Some(9),
            fps: Some(16),
            ..Default::default()
        };
        g.validate(&video_req).unwrap();
        let video_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Mlx,
            BerniniKnobs::default().to_plan_knobs(),
            &video_req,
        )
        .unwrap();
        assert_eq!(
            video_plan.output_contract,
            BerniniOutputContract::Video {
                fps: 16,
                latent_frames: 3
            }
        );

        // Invalid frame count (16 != 1 + 4k) and negative scheduler_shift fail pre-load.
        let bad_frames = GenerationRequest {
            frames: Some(16),
            ..still_req.clone()
        };
        assert!(g.validate(&bad_frames).is_err());
        let bad_shift = GenerationRequest {
            scheduler_shift: Some(-1.0),
            ..still_req
        };
        assert!(g.validate(&bad_shift).is_err());
    }
}
