//! PuLID-FLUX end-to-end generate (sc-3074) + native face wiring (sc-3073).
//!
//! Assembles the full face-identity path on top of the FLUX.1-dev backbone:
//!   1. **Face analysis** (native MLX, epic 3079): the reference face (`Conditioning::Reference`) →
//!      `mlx_gen_face::FaceAnalysis` → largest face's ArcFace embedding (512-d) + `face_features_image`
//!      (512² aligned, background-whitened grayscale). No Python/onnx.
//!   2. **EVA-CLIP** (sc-3070): `face_features_image` → resize/normalize → `id_cond_vit` (768-d,
//!      L2-normalized) + 5 hidden states.
//!   3. **IDFormer** (sc-3071): `id_cond = cat(arcface 512, id_cond_vit 768)` + hidden → `id_embedding`
//!      `[1,32,2048]`.
//!   4. **CA injection** (sc-3072): build `PulidCa` and run the FLUX flow-match denoise through
//!      `Flux1::generate_with_injector` (fake-CFG, true_cfg=1.0) → AE decode.
//!
//! The whole conditioning path runs in **f32**: mlx-gen-flux keeps the DiT image stream in f32 (mixed
//! precision, sc-2787), so f32 CA weights/id_embedding inject cleanly into the f32 hidden tokens (no
//! dtype mismatch) and at higher accuracy than the reference's bf16 — the e2e gate is ArcFace-cosine
//! (cross-encoder, loose), so this is strictly safe. Real-CFG / uncond-id is sc-3075; quant is sc-3076.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use mlx_rs::ops::{concatenate_axis, divide, maximum, sqrt, square, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, gen_core, CancelFlag,
    Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result, WeightsSource,
};
use mlx_gen_face::{Detection, FaceAnalysis};
use mlx_gen_flux::config::FluxVariant;
use mlx_gen_flux::model::{load_flux1, Flux1};

use crate::ca::{CaActivity, PulidCa};
use crate::eva_clip::{transform, EvaConfig, EvaVisionTransformer};
use crate::idformer::{IdFormer, IdFormerConfig};

/// FLUX.1-dev DiT block counts (the PuLID injection schedule is defined over these).
const NUM_DOUBLE_BLOCKS: usize = 19;
const NUM_SINGLE_BLOCKS: usize = 38;
/// Default step from which the real-CFG (and uncond-id) branch engages when the request leaves
/// `timestep_to_start_cfg` unset — the upstream PuLID default (the photoreal preset overrides to 4
/// via `req.timestep_to_start_cfg`).
const DEFAULT_TIMESTEP_TO_START_CFG: usize = 1;
/// ArcFace (antelopev2) face embedding width — the first half of the IdFormer `id_cond`
/// (`cat(arcface, id_cond_vit)`). The id_cond_vit half is the EVA head's `proj_dim`.
const ARCFACE_DIM: i32 = 512;
pub const MODEL_ID: &str = "pulid_flux";

/// Primary typed route declaration for PuLID-FLUX (`REG-1`, `AP5`).
pub(crate) fn route_pulid_flux() -> gen_core::RouteRecord {
    let mut samplers = curated_sampler_names();
    samplers.push("flow_match");
    let mut schedulers = curated_scheduler_names();
    schedulers.push("linear");

    gen_core::RouteRecord {
        route_id: MODEL_ID,
        family: "pulid",
        backend: "mlx",
        modality: Modality::Image,
        mac_only: true,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![
                    gen_core::TextEncoderDeclaration::new("clip_l", Some(77)),
                    gen_core::TextEncoderDeclaration::new("t5_xxl", Some(512)),
                ],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                // The reference face; PuLID drives its own real-CFG (sc-3075).
                kinds: vec![ConditioningKind::Reference],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::RequiresReference,
                supports_negative_prompt: true,
                modules: vec!["pulid_idformer", "eva_clip", "arcface"],
            },
            sampling: gen_core::SamplingSurface {
                samplers,
                schedulers,
                guidance_methods: Vec::new(),
                supports_guidance: true,
                supports_true_cfg: true,
                supports_cfg_interval: false,
                requires_sigma_shift: true,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new("flux1_mmdit_pulid")
                    .with_pag(FluxVariant::Dev.pag_capability()),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&mlx_gen::gen_core::FLUX1_LATENT_SPACE),
                pixel_stride: Some(mlx_gen_flux::SIZE_MULTIPLE),
                token_width: Some(mlx_gen_flux::PACKED_TOKEN_WIDTH as u32),
                supports_diffusion_decoder: false,
            },
            upsamplers: gen_core::UpsamplerSurface::default(),
            adapters: gen_core::AdapterSurface::unsupported(),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["clip_l", "t5_xxl"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::ConditioningEncode,
                        ["eva_clip", "pulid_idformer"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["flux1_mmdit_pulid"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["flux1_ae"],
                        false,
                    ),
                ],
                supports_sequential_offload: true,
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: true,
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::LatentProjection,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 256,
            max_size: 2048,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::BackendGlobalSeeded,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(FluxVariant::Dev.default_steps()),
            guidance: Some(4.0),
            sampler: Some("flow_match"),
            schedule: Some("linear"),
            shift: Some(1.15),
            strength: Some(1.0),
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings: gen_core::RouteBindings::generator(descriptor, |spec| {
            load_pulid_flux(spec).map_err(Into::into)
        })
        .with_render_plan(compile_plan)
        .with_memory_registration(MEMORY_REGISTRATION)
        .with_memory_fixture(
            gen_core::mlx_memory_contract_surface_specs,
            crate::memory_strategy::weights_free_contract,
        )
        .with_memory_behavior_registration(MEMORY_BEHAVIOR),
    }
}

/// Conditioned feature sub-route for PuLID over FLUX.1-dev (`REG-1`, `AP5`).
pub(crate) fn route_flux1_dev_pulid() -> gen_core::RouteRecord {
    let mut route = route_pulid_flux();
    route.route_id = "flux1_dev_pulid";
    route.bindings =
        gen_core::RouteBindings::feature_route(Some("flux1_dev"), descriptor)
            .with_render_plan(compile_plan);
    route
}

pub fn descriptor() -> ModelDescriptor {
    route_pulid_flux().compute_descriptor()
}

/// L2-normalize each row of `[B, D]` over the feature axis (the PuLID `id_cond_vit` normalization).
fn l2_normalize_rows(x: &Array) -> Result<Array> {
    let sumsq = sum_axes(&square(x)?, &[1], true)?; // [B, 1]
                                                    // Clamp the norm to a tiny epsilon (torch `F.normalize`'s default eps) so a degenerate zero-norm
                                                    // row — e.g. a bad ArcFace crop — yields a zero vector instead of NaN-poisoning the entire
                                                    // identity-conditioned generation. Byte-identical for real embeddings (norm >> eps) (F-078).
    let norm = maximum(&sqrt(&sumsq)?, Array::from_f32(1e-12))?;
    Ok(divide(x, &norm)?)
}

pub struct PulidFlux {
    descriptor: ModelDescriptor,
    flux: Flux1,
    eva: EvaVisionTransformer,
    idformer: IdFormer,
    /// The PuLID checkpoint weights (f32) — kept to build a per-generate [`PulidCa`] bound to the
    /// computed id_embedding. `pulid_encoder.*` already consumed by `idformer`; `pulid_ca.*` here.
    pulid: Weights,
    face: FaceAnalysis,
    memory_strategy: gen_core::MemoryProviderContract,
    loaded_spec: LoadSpec,
    identity_inventory: Option<crate::memory_strategy::IdentityArtifactInventory>,
}

impl PulidFlux {
    /// Build from already-loaded sub-models. `pulid` must hold both `pulid_encoder.*` and
    /// `pulid_ca.*` (cast to f32); `eva`/`idformer` must likewise be f32 (the conditioning path).
    /// `face` must have a parser attached (`with_parser`) for `face_features_image`.
    pub fn new(
        flux: Flux1,
        eva: EvaVisionTransformer,
        pulid: Weights,
        face: FaceAnalysis,
    ) -> Result<Self> {
        let idformer = IdFormer::from_weights(&pulid, "pulid_encoder", IdFormerConfig::default())?;
        let mut memory_strategy = flux.memory_strategy_contract().cloned().unwrap_or_else(|| {
            gen_core::MemoryProviderContract::compatibility_default(
                MODEL_ID,
                gen_core::MemoryBackendRealization::MlxMetal {
                    bounded_wired_residency: true,
                    lazy_or_mmap_materialization: true,
                    explicit_evaluation_and_synchronization: true,
                    cache_eviction: true,
                },
            )
        });
        memory_strategy.provider_id = MODEL_ID.into();
        memory_strategy.calibration = None;
        Ok(Self {
            descriptor: descriptor(),
            flux,
            eva,
            idformer,
            pulid,
            face,
            memory_strategy,
            loaded_spec: LoadSpec::new(WeightsSource::Dir("/constructed-pulid".into())),
            identity_inventory: None,
        })
    }

    fn new_loaded(
        flux: Flux1,
        eva: EvaVisionTransformer,
        pulid: Weights,
        face: FaceAnalysis,
        loaded_spec: LoadSpec,
        memory_strategy: gen_core::MemoryProviderContract,
        identity_inventory: crate::memory_strategy::IdentityArtifactInventory,
    ) -> Result<Self> {
        let mut model = Self::new(flux, eva, pulid, face)?;
        model.loaded_spec = loaded_spec;
        model.memory_strategy = memory_strategy;
        model.identity_inventory = Some(identity_inventory);
        Ok(model)
    }

    /// Face image (RGB, row-major, `h×w`) → `id_embedding` `[1,32,2048]`. Mirrors PuLID's
    /// `get_id_embedding` (the conditional side; cal_uncond is sc-3075).
    ///
    /// `cancel` (F-108): the identity tower is the priciest pre-denoise stage (SCRFD + BiSeNet +
    /// ArcFace, then the 24-block EVA-CLIP tower + IDFormer), and previously ran with zero cancel
    /// checks. We check between stages. The selected-face `embed` / `face_features_image` stages
    /// already materialize to host
    /// (`Face.embedding` is a `Vec<f32>`), so the check after them is effective as-is; the EVA tower
    /// output is lazy, so we `eval` it before the check ahead of the IDFormer (no lazy-eval false
    /// green). Returns [`Error::Canceled`] on trip.
    pub fn compute_id_embedding(
        &self,
        pixels: &[u8],
        h: usize,
        w: usize,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // F-138: `detect` preserves largest-first ordering without running ArcFace. PuLID needs only
        // that one identity, so embed it on demand instead of `analyze` embedding every face in a
        // group reference.
        let face = Self::largest_face_embedding(
            || self.face.detect(pixels, h, w),
            |detection| self.face.embed(pixels, h, w, detection),
        )?;
        // ArcFace 512-d (id_ante_embedding) — raw, un-normalized, matching the reference.
        let arcface = Array::from_slice(&face.embedding, &[1, face.embedding.len() as i32]);
        // face_features_image (512² aligned, bg-whitened gray) → EVA 336² transform → tower.
        let ffi = self.face.face_features_image(pixels, h, w, &face)?;
        // SCRFD/BiSeNet/ArcFace above are host-materialized; honor a cancel before the EVA tower.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let eva_in = transform::eva_transform(&ffi, self.eva_image_size())?;
        let eva_out = self.eva.forward(&eva_in)?;
        // Force the 24-block EVA tower so the check observes real progress before the IDFormer.
        mlx_rs::transforms::eval(&eva_out.hidden)?;
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let id_cond_vit = l2_normalize_rows(&eva_out.id_cond_vit)?; // [1,768]
        let id_cond = concatenate_axis(&[&arcface, &id_cond_vit], 1)?; // [1,1280]
        self.idformer.forward(&id_cond, &eva_out.hidden)
    }

    /// Detect faces in the face stack's largest-first order and perform exactly one requested
    /// embedding. Keeping this selection boundary separate makes the no-N-face-ArcFace-work
    /// guarantee testable without a real MLX model.
    fn largest_face_embedding<T>(
        detect: impl FnOnce() -> Result<Vec<Detection>>,
        embed: impl FnOnce(&Detection) -> Result<T>,
    ) -> Result<T> {
        let detections = detect()?;
        let largest = detections.first().ok_or_else(|| {
            Error::Msg("pulid_flux: no face detected in the reference image".into())
        })?;
        embed(largest)
    }

    /// The unconditional id_embedding — IDFormer over **zeroed** id_cond + zeroed hidden states (the
    /// PuLID `get_id_embedding(cal_uncond=True)` path), injected on the negative real-CFG branch.
    pub fn compute_uncond_id_embedding(&self) -> Result<Array> {
        // Derive the EVA token geometry from the loaded tower's config, not the default-tower
        // constants (F-082): seq = grid²+1 (CLS + patches) = 577, embed = embed_dim = 1024, and one
        // zeroed hidden state per captured block (5). `id_cond` is the IdFormer input width (ArcFace
        // 512 + the EVA head's proj_dim 768).
        let cfg = self.eva.config();
        let seq = cfg.grid() * cfg.grid() + 1;
        let embed = cfg.embed_dim;
        let id_cond_dim = ARCFACE_DIM + cfg.proj_dim;
        let id_cond = Array::from_slice(&vec![0f32; id_cond_dim as usize], &[1, id_cond_dim]);
        let hidden: Vec<Array> = cfg
            .hidden_capture
            .iter()
            .map(|_| Array::from_slice(&vec![0f32; (seq * embed) as usize], &[1, seq, embed]))
            .collect();
        self.idformer.forward(&id_cond, &hidden)
    }

    fn eva_image_size(&self) -> i32 {
        self.eva.config().image_size
    }

    fn reference_face<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        select_reference_face(&req.conditioning)
    }
}

/// Pick the single reference-face conditioning, rejecting any other kind. PuLID-FLUX advertises only
/// `ConditioningKind::Reference`, so a stray Control/Mask/etc. attached by a worker must **error**
/// rather than be silently dropped when `generate` clears `flux_req.conditioning` (F-094).
fn select_reference_face(conditioning: &[Conditioning]) -> Result<(&Image, f32)> {
    let mut found = None;
    for c in conditioning {
        match c {
            Conditioning::Reference { image, strength } => {
                if found.is_some() {
                    return Err(Error::Msg(
                        "pulid_flux: exactly one reference face is supported".into(),
                    ));
                }
                // The reference strength is the PuLID id_weight (0–3, default 1.0).
                found = Some((image, strength.unwrap_or(1.0)));
            }
            other => {
                return Err(Error::Msg(format!(
                    "pulid_flux: unsupported conditioning {:?} — only a reference face \
                     (Conditioning::Reference) is accepted",
                    other.kind()
                )));
            }
        }
    }
    found.ok_or_else(|| {
        Error::Msg(
            "pulid_flux: a reference face image (Conditioning::Reference) is required".into(),
        )
    })
}

/// Freeze the request trajectory for a model assembled with [`PulidFlux::new`]. Such a model owns
/// its identity stack but no load spec, so the caller compiles this plan before loading the
/// sub-models and executes it with [`Generator::generate_planned`]; [`Generator::generate`] refuses
/// a constructor-built model rather than compiling after load.
pub fn compile_constructed_plan(req: &GenerationRequest) -> gen_core::Result<gen_core::RenderPlan> {
    descriptor().capabilities.validate_request(MODEL_ID, req)?;
    let grid = mlx_gen_flux::model::SIZE_MULTIPLE;
    if !req.width.is_multiple_of(grid) || !req.height.is_multiple_of(grid) {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: width and height must be multiples of {grid}, got {}x{}",
            req.width, req.height
        )));
    }
    select_reference_face(&req.conditioning)?;
    if req.cancel.is_cancelled() {
        return Err(gen_core::Error::Canceled);
    }
    Ok(mlx_gen_flux::model::compile_trajectory_for(
        FluxVariant::Dev,
        &descriptor(),
        req,
    )?)
}

/// Freeze the face source, identity slots, sampling and CFG controls before loading PuLID.
pub(crate) fn compile_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    let identity = spec.identity.as_ref().ok_or_else(|| {
        gen_core::Error::Unsupported("pulid_flux needs identity weights before load".into())
    })?;
    if identity.encoder.is_none() || identity.eva.is_none() || identity.face_dir.is_none() {
        return Err(gen_core::Error::Unsupported(
            "pulid_flux needs encoder, EVA and face identity weights before load".into(),
        ));
    }
    if req.use_pid && spec.pid.is_none() {
        return Err(gen_core::Error::Unsupported(
            "pulid_flux PiD decode needs a loaded overlay".into(),
        ));
    }
    Ok(Some(compile_constructed_plan(req)?.bind_load(spec)))
}

impl PulidFlux {
    /// Load-time selections bound to every reported attempt.
    fn observe_loaded(&self) {
        mlx_gen::image_execution::policy_debug(
            "PulidFlux.load_precision_policy",
            &self.loaded_spec.precision,
        );
        mlx_gen::image_execution::policy_debug(
            "PulidFlux.load_quantization_policy",
            &self.loaded_spec.quantize,
        );
    }
}

impl Generator for PulidFlux {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Validate against PuLID's OWN descriptor, not the FLUX-dev backbone's (F-026). `generate`
        // delegates the denoise to `PulidFlux::flux` (the dev backbone), whose descriptor advertises
        // the `hyper` sampler — but PuLID deliberately does NOT load the dev Hyper LoRA, so a `hyper`
        // request would silently render WITHOUT it. PuLID's descriptor omits `hyper`, so delegating to
        // its own capability floor rejects that (and every other un-advertised sampler/scheduler,
        // negative_prompt, true_cfg, size/count/steps) instead of the backbone waving it through. The
        // `?` keeps the typed `Error::Unsupported` for capability gaps.
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        // PuLID-specific: a reference face image is required (consumed into the identity injector).
        self.reference_face(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        mlx_gen::image_execution::without_capture(|| self.generate_observed(req, on_progress))
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let pag = self
            .descriptor
            .capabilities
            .resolve_pag(self.descriptor.id, req)?;
        let total_steps = u32::try_from(plan.denoise_sigmas().len().saturating_sub(1))
            .map_err(|_| gen_core::Error::Unsupported("PuLID step count exceeds u32".into()))?;
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let mut report =
            gen_core::GenerationReport::from_output(self.generate_impl(req, &run, on_progress)?);
        report.pag = pag.map(|p| p.execution_report(total_steps));
        Ok(report)
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        mlx_gen::image_execution::without_capture(|| {
            self.generate_with_report_observed(req, on_progress)
        })
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.loaded_spec, &self.memory_strategy, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(&self.loaded_spec, &self.memory_strategy, context)
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::image_execution::complete_image_report_attempt(
            self.descriptor(),
            &req.cancel,
            || {
                self.observe_loaded();
                let mut progress =
                    |event| mlx_gen::image_execution::without_capture(|| on_progress(event));
                self.generate_with_report_observed(req, &mut progress)
            },
        )
    }

    /// The admitted runner's entry: the same planned body, captured as one attempt that covers
    /// the work inside its Conditioning/Denoise/Decode scopes.
    fn generate_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::image_execution::complete_planned_image_report(
            self,
            req,
            plan,
            scope,
            on_progress,
            || self.observe_loaded(),
        )
    }
}

impl PulidFlux {
    fn generate_with_report_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        if self.identity_inventory.is_none() {
            return Err(gen_core::Error::Unsupported(
                "pulid_flux: a constructor-built model has no load spec to plan against; compile \
                 `compile_constructed_plan` before loading its sub-models and call `generate_planned`"
                    .into(),
            ));
        }
        let plan = compile_plan(&self.loaded_spec, req)?
            .ok_or_else(|| gen_core::Error::Unsupported("PuLID plan is unavailable".into()))?;
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_observed(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report_observed(req, on_progress)?
            .output
            .ok_or_else(|| gen_core::Error::Msg("PuLID render produced no image".into()))
    }
}

impl PulidFlux {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        run: &gen_core::RenderRun<'_, '_>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.verify_identity_inventory()?;
        // Self-validate first, like every sibling generate_impl (chroma/svd/ideogram) — against
        // PuLID's OWN descriptor floor (F-026), so a caller that skips `validate` still gets the
        // typed `Unsupported` (e.g. `sampler: "hyper"`, which PuLID deliberately doesn't advertise
        // because it never loads the dev Hyper-FLUX LoRA) instead of a silently degraded render
        // (F-011). The `?` bridges `gen_core::Error` → `Error` with `Unsupported` kept typed.
        self.validate(req)?;
        // F-108: the identity stack (face analysis + EVA tower + IDFormer + CA build + the backbone
        // T5/CLIP encodes) is the priciest pre-denoise work and previously ran with zero cancel
        // checks. Bail up front, and `compute_id_embedding` checks between its own stages.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (image, id_weight) = self.reference_face(req)?;
        mlx_gen::image_execution::scalar("pulid.identity_weight", id_weight);
        let pag = self
            .descriptor
            .capabilities
            .resolve_pag(self.descriptor.id, req)?;
        mlx_gen::image_execution::policy_debug("pulid.pag", &pag);
        let bounded_attention_selected = req.memory.is_some_and(|memory| memory.chunk_attention);
        // The identity embedding is computed inside the Conditioning phase (below); its route is
        // observed there, once the embedding exists.
        let observe_embedding = |embedding: &Array| {
            observe_identity_route(
                (image.width, image.height),
                embedding.shape(),
                embedding.dtype(),
                bounded_attention_selected,
            );
        };
        let positive_activity = Arc::new(CaActivity::default());
        let negative_activity = Arc::new(CaActivity::default());
        let mk_ca = |emb: Array, activity: Arc<CaActivity>| {
            PulidCa::from_weights(
                &self.pulid,
                "pulid_ca",
                emb,
                id_weight,
                NUM_DOUBLE_BLOCKS,
                NUM_SINGLE_BLOCKS,
            )
            .map(|ca| {
                ca.with_bounded_attention(req.cancel.clone(), bounded_attention_selected)
                    .with_activity(activity)
            })
        };
        // The reference face is consumed into the injector; hand the FLUX backbone a plain request
        // (it rejects conditioning + negative_prompt it doesn't itself implement — both are handled
        // here / passed to the CFG denoise directly).
        let mut flux_req = req.clone();
        flux_req.conditioning = Vec::new();
        flux_req.negative_prompt = None;
        flux_req.true_cfg = None; // PuLID drives real-CFG itself; the backbone forbids it

        // The owner now prepares the face/EVA/IDFormer branch after staged T5/CLIP weights have
        // dropped, inside the same Conditioning phase as text. This avoids retaining the large
        // identity arrays beside text encoding and keeps the measured phase complete.
        let true_cfg = req.true_cfg.unwrap_or(1.0);
        mlx_gen::image_execution::scalar("pulid.true_cfg", true_cfg);
        let result = if true_cfg > 1.0 + 1e-3 {
            // Real-CFG (sc-3075): positive (id) + negative (uncond id) branches + a negative prompt.
            let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
            let start_cfg = req
                .timestep_to_start_cfg
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_TIMESTEP_TO_START_CFG);
            observe_cfg_branch(Some(start_cfg));
            self.flux.generate_with_prepared_injector_cfg_planned(
                &flux_req,
                neg_prompt,
                true_cfg,
                start_cfg,
                run,
                || {
                    let pos = self.compute_id_embedding(
                        &image.pixels,
                        image.height as usize,
                        image.width as usize,
                        &req.cancel,
                    )?;
                    observe_embedding(&pos);
                    let neg = self.compute_uncond_id_embedding()?;
                    mlx_rs::transforms::eval([&pos, &neg])?;
                    Ok((
                        mk_ca(pos, positive_activity.clone())?,
                        mk_ca(neg, negative_activity.clone())?,
                    ))
                },
                on_progress,
            )
        } else {
            // Fake-CFG (true_cfg = 1.0): single forward (sc-3074), bit-identical to that path.
            observe_cfg_branch(None);
            self.flux.generate_with_prepared_injector_planned(
                &flux_req,
                run,
                || {
                    let id_embedding = self.compute_id_embedding(
                        &image.pixels,
                        image.height as usize,
                        image.width as usize,
                        &req.cancel,
                    )?;
                    observe_embedding(&id_embedding);
                    mlx_rs::transforms::eval([&id_embedding])?;
                    mk_ca(id_embedding, positive_activity.clone())
                },
                on_progress,
            )
        };
        if result.is_ok() {
            observe_completed_route(&positive_activity, &negative_activity);
        }
        self.verify_identity_inventory()?;
        result
    }

    fn verify_identity_inventory(&self) -> Result<()> {
        self.identity_inventory
            .as_ref()
            .map_or(Ok(()), |inventory| {
                inventory.ensure_unchanged().map_err(Into::into)
            })
    }
}

fn observe_identity_route(
    reference_size: (u32, u32),
    embedding_shape: &[i32],
    embedding_dtype: Dtype,
    bounded_attention: bool,
) {
    mlx_gen::image_execution::policy_debug("pulid.reference_size", &reference_size);
    mlx_gen::image_execution::policy_debug("pulid.embedding_shape", &embedding_shape);
    mlx_gen::image_execution::policy_debug("pulid.embedding_dtype", &embedding_dtype);
    mlx_gen::image_execution::policy_debug("pulid.ca_budget_selected", &bounded_attention);
}

fn observe_cfg_branch(start_step: Option<usize>) {
    mlx_gen::image_execution::policy_debug("pulid.cfg_start_step", &start_step);
    mlx_gen::image_execution::policy_debug("pulid.dual_branch_cfg_selected", &start_step.is_some());
}

fn observe_completed_route(positive: &CaActivity, negative: &CaActivity) {
    mlx_gen::image_execution::policy_debug("pulid.real_cfg", &negative.forward_entered());
    mlx_gen::image_execution::policy_debug(
        "pulid.ca_active",
        &(positive.attention_used() || negative.attention_used()),
    );
    mlx_gen::image_execution::policy_debug(
        "pulid.ca_bounded_attention",
        &(positive.attention_chunked() || negative.attention_chunked()),
    );
}

// ---- registration -------------------------------------------------------------------------------

/// Extract the `PathBuf` from a spec-supplied identity [`WeightsSource`], validating existence with a
/// clear, slot-named message (F-114 / sc-13664). Accepts either `File` or `Dir` — the caller knows
/// which it expects. This is the ONLY identity-path resolver: every path is caller-supplied via
/// `LoadSpec::identity`, with no env / HF-cache fallback.
fn spec_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    let p = match src {
        WeightsSource::File(p) | WeightsSource::Dir(p) => p.clone(),
    };
    if !p.exists() {
        return Err(Error::Msg(format!(
            "pulid_flux: LoadSpec identity {what} path does not exist: {}",
            p.display()
        )));
    }
    Ok(p)
}

/// Load EVA weights (f32) from a converted safetensors (tools/convert_eva_clip.py output). Keys are
/// bare mlx-names (no prefix).
fn load_eva(path: &Path) -> Result<EvaVisionTransformer> {
    let mut w = Weights::from_file(path)?;
    w.cast_all(Dtype::Float32)?;
    let eva = EvaVisionTransformer::from_weights(&w, "", EvaConfig::default())?;
    // Materialize at load (after the f32 cast): left lazy, the first identity encode's command
    // buffers wait on the safetensors reads — past the GPU watchdog on a cold page cache (sc-24245;
    // see `mlx_gen_qwen_image::loader::load_transformer_with`).
    w.materialize_accessed()?;
    Ok(eva)
}

/// Registered loader for the `pulid_flux` target. Every identity sub-model path is **required** and
/// comes from `LoadSpec::identity` (sc-13664 — no `PULID_*` env vars, no on-disk HF-cache scan; an
/// absent slot or sub-field is a load-time, actionable error). Weight sources:
///   * FLUX.1-dev snapshot dir — `spec.weights` (Dir).
///   * identity encoder — `spec.identity.encoder` (`pulid_flux_v0.9.1.safetensors`).
///   * EVA tower — `spec.identity.eva` (converted EVA02-CLIP-L-14-336).
///   * face dir — `spec.identity.face_dir` (scrfd_10g / arcface_iresnet100 / bisenet_parsing).
pub fn load_pulid_flux(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // sc-13664: require the identity slot and each sub-field, and resolve (existence-check) every path
    // BEFORE the FLUX backbone load — so a missing identity slot/field fails fast at load with a
    // slot-named error, never a mid-load `PULID_*` env / HF-cache fetch.
    let identity = spec.identity.as_ref().ok_or_else(|| {
        Error::Msg(
            "pulid_flux requires the identity weights — set LoadSpec::identity with encoder / eva / \
             face_dir (WeightsSource paths); there is no PULID_* env or HF-cache fallback"
                .into(),
        )
    })?;
    let encoder = identity.encoder.as_ref().ok_or_else(|| {
        Error::Msg(
            "pulid_flux requires the identity encoder — set LoadSpec::identity.encoder to the \
             pulid_flux_v0.9.1.safetensors path"
                .into(),
        )
    })?;
    let eva_src = identity.eva.as_ref().ok_or_else(|| {
        Error::Msg(
            "pulid_flux requires the identity EVA tower — set LoadSpec::identity.eva to the \
             converted EVA02-CLIP-L-14-336 .safetensors path"
                .into(),
        )
    })?;
    let face_src = identity.face_dir.as_ref().ok_or_else(|| {
        Error::Msg(
            "pulid_flux requires the identity face-analysis dir — set LoadSpec::identity.face_dir to \
             the scrfd_10g / arcface_iresnet100 / bisenet_parsing weights dir"
                .into(),
        )
    })?;
    let encoder_path = spec_path(encoder, "encoder")?;
    let eva_path = spec_path(eva_src, "eva")?;
    let face_dir = spec_path(face_src, "face_dir")?;

    // FLUX.1-dev backbone (its loader validates the snapshot dir). Q8/Q4 (sc-3076) composes for free:
    // `spec.quantize` flows through `load_flux1`, quantizing ONLY the FLUX backbone linears. The PuLID
    // conditioning (EVA tower, IDFormer, the 20 CA modules) stays f32 — it runs once per image, not
    // per step, so the memory win is the backbone, and the f32 CA residual injects into the (still
    // f32) DiT image stream unchanged. No quant-specific wiring needed here.
    let identity_inventory = crate::memory_strategy::admitted_identity_inventory(spec)?;
    let memory_strategy =
        crate::memory_strategy::contract_with_inventory(spec, &identity_inventory)?;
    identity_inventory.ensure_unchanged()?;
    let flux = load_flux1(
        FluxVariant::Dev,
        &crate::memory_strategy::backbone_spec(spec),
    )?;

    // PuLID encoder + CA weights, cast f32 (conditioning path).
    let mut pulid = Weights::from_file(encoder_path)?;
    pulid.cast_all(Dtype::Float32)?;
    // Materialize at load, as `load_eva` does (sc-24245). The whole map is the IDFormer + the CA
    // modules, all read per generate, so it is materialized entire.
    pulid.materialize()?;
    identity_inventory.ensure_unchanged()?;

    // EVA-CLIP tower (f32).
    let eva = load_eva(&eva_path)?;
    identity_inventory.ensure_unchanged()?;

    // Native face stack.
    let scrfd = Weights::from_file(face_dir.join("scrfd_10g.safetensors"))?;
    let arcface = Weights::from_file(face_dir.join("arcface_iresnet100.safetensors"))?;
    let parser = Weights::from_file(face_dir.join("bisenet_parsing.safetensors"))?;
    let face = FaceAnalysis::load(&scrfd, &arcface)?.with_parser(&parser)?;
    // Materialize at load, as `load_eva` does (sc-24245).
    for weights in [&scrfd, &arcface, &parser] {
        weights.materialize_accessed()?;
    }
    identity_inventory.ensure_unchanged()?;

    Ok(Box::new(PulidFlux::new_loaded(
        flux,
        eva,
        pulid,
        face,
        spec.clone(),
        memory_strategy,
        identity_inventory,
    )?))
}

pub(crate) const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: crate::memory_strategy::contract_for,
    safety_check: crate::memory_strategy::safety_check,
};

pub(crate) const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: crate::memory_strategy::registered_fixture,
        begin_request: crate::memory_strategy::registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{IdentityWeights, LoadShape, OffloadPolicy, Quant};

    #[test]
    fn constructed_model_plans_without_identity_paths_but_keeps_request_binding() {
        let spec = LoadSpec::new(WeightsSource::Dir("/constructed-pulid".into()));
        let mut req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![3; 64 * 64 * 3],
                },
                strength: Some(1.0),
            }],
            ..Default::default()
        };
        let plan = compile_constructed_plan(&req).unwrap();
        plan.check_request(&descriptor(), &req).unwrap();
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&spec))
            .unwrap();
        assert!(compile_plan(&spec, &req).is_err());
        if let Conditioning::Reference { image, .. } = &mut req.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert!(plan.check_request(&descriptor(), &req).is_err());
    }

    #[test]
    fn pulid_plan_binds_face_and_identity_slots_before_load() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()));
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File("/missing/pulid".into())),
            eva: Some(WeightsSource::File("/missing/eva".into())),
            face_dir: Some(WeightsSource::Dir("/missing/face".into())),
        });
        let mut req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![3; 64 * 64 * 3],
                },
                strength: Some(1.5),
            }],
            ..Default::default()
        };
        let plan = compile_plan(&spec, &req)
            .unwrap()
            .expect("PuLID reference needs a pre-load trajectory");
        plan.check_request(&descriptor(), &req).unwrap();
        let mut changed_spec = spec.clone();
        changed_spec.identity.as_mut().unwrap().eva =
            Some(WeightsSource::File("/different/eva".into()));
        assert!(plan
            .check_load(&gen_core::RenderLoadBinding::from_spec(&changed_spec))
            .is_err());
        if let Conditioning::Reference { image, .. } = &mut req.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert!(plan.check_request(&descriptor(), &req).is_err());
        req.conditioning.push(Conditioning::Reference {
            image: Image {
                width: 64,
                height: 64,
                pixels: vec![0; 64 * 64 * 3],
            },
            strength: None,
        });
        assert!(compile_plan(&spec, &req).is_err());
        req.conditioning.pop();
        spec.identity.as_mut().unwrap().face_dir = None;
        assert!(compile_plan(&spec, &req).is_err());
    }

    /// Zero steps, an off-grid size and a below-minimum size are refused by both pre-load
    /// compilers (registry and constructed) at missing identity paths, so no weight is opened.
    #[test]
    fn pulid_plan_refuses_zero_steps_and_invalid_sizes_before_load() {
        let registry = crate::provider_registry().unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir("/missing/flux".into()));
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File("/missing/pulid".into())),
            eva: Some(WeightsSource::File("/missing/eva".into())),
            face_dir: Some(WeightsSource::Dir("/missing/face".into())),
        });
        let valid = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: Some(1.0),
            }],
            ..Default::default()
        };
        registry
            .compile_render_plan(MODEL_ID, &spec, &valid)
            .unwrap()
            .expect("PuLID reference needs a pre-load trajectory");
        for (width, height, steps, expected) in [
            (512, 512, Some(0), "steps must be >= 1"),
            (321, 512, Some(4), "multiples of 16"),
            (512, 321, Some(4), "multiples of 16"),
            (128, 512, Some(4), "outside supported range"),
            (512, 240, Some(4), "outside supported range"),
        ] {
            let req = GenerationRequest {
                width,
                height,
                steps,
                ..valid.clone()
            };
            let registered = registry
                .compile_render_plan(MODEL_ID, &spec, &req)
                .unwrap_err()
                .to_string();
            assert!(
                registered.contains(expected),
                "{width}x{height}: {registered}"
            );
            let constructed = compile_constructed_plan(&req).unwrap_err().to_string();
            assert!(
                constructed.contains(expected),
                "{width}x{height}: {constructed}"
            );
        }
    }

    /// A constructor-built model has no load spec, so `generate` refuses instead of compiling
    /// after load; the caller compiles first and uses `generate_planned`.
    #[test]
    fn constructed_model_generate_refuses_post_load_compile() {
        let model = weightless_pulid();
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        let err = Generator::generate(&model, &req, &mut |_| {}).unwrap_err();
        assert!(
            err.to_string().contains("compile_constructed_plan"),
            "got: {err:?}"
        );
    }

    #[test]
    fn completed_policy_distinguishes_ca_budget_and_cfg_start() {
        let capture = |bounded, start| {
            mlx_gen::image_execution::complete_image_attempt(
                &descriptor(),
                &CancelFlag::new(),
                || {
                    observe_identity_route((512, 512), &[1, 32, 2048], Dtype::Float32, bounded);
                    observe_cfg_branch(start);
                    mlx_gen::image_execution::sampled_native(
                        "flow_match",
                        "sigma",
                        Dtype::Float32,
                        &[1.0, 0.0],
                        42,
                    );
                    Ok(GenerationOutput::Images(vec![Image {
                        width: 1,
                        height: 1,
                        pixels: vec![0, 0, 0],
                    }]))
                },
            )
            .unwrap()
            .execution
        };
        let ordinary = capture(false, None);
        let bounded = capture(true, None);
        let real_early = capture(false, Some(1));
        let real_late = capture(false, Some(4));
        for identity in [&ordinary, &bounded, &real_early, &real_late] {
            assert!(identity.resolved().is_some());
        }
        let quality = |identity: &gen_core::generator::ExecutionIdentity| {
            identity.resolved().unwrap().quality_policy.clone()
        };
        assert_ne!(quality(&ordinary), quality(&bounded));
        assert_ne!(quality(&ordinary), quality(&real_early));
        assert_ne!(quality(&real_early), quality(&real_late));
    }

    #[test]
    fn completed_policy_does_not_claim_a_negative_forward_from_cfg_selection() {
        let completed = mlx_gen::image_execution::complete_image_attempt(
            &descriptor(),
            &CancelFlag::new(),
            || {
                // A one-step render with start=2 selects the dual-branch route but never
                // reaches its negative forward. The injector observations remain empty.
                observe_identity_route((512, 512), &[1, 32, 2048], Dtype::Float32, false);
                observe_cfg_branch(Some(2));
                let positive = CaActivity::default();
                let negative = CaActivity::default();
                positive.record_forward();
                observe_completed_route(&positive, &negative);
                mlx_gen::image_execution::sampled_native(
                    "flow_match",
                    "sigma",
                    Dtype::Float32,
                    &[1.0, 0.0],
                    42,
                );
                Ok(GenerationOutput::Images(vec![Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                }]))
            },
        )
        .unwrap();
        let policy = match &completed.execution.resolved().unwrap().quality_policy {
            gen_core::generator::ExecutionFact::Resolved { value } => value,
            other => panic!("expected resolved quality policy: {other:?}"),
        };
        assert_eq!(policy["pulid.dual_branch_cfg_selected"], "true");
        assert_eq!(policy["pulid.cfg_start_step"], "Some(2)");
        assert_eq!(policy["pulid.real_cfg"], "false");
        assert_eq!(policy["pulid.ca_active"], "false");
    }

    #[test]
    fn completed_policy_does_not_claim_ca_from_zero_identity_strength() {
        let completed = mlx_gen::image_execution::complete_image_attempt(
            &descriptor(),
            &CancelFlag::new(),
            || {
                observe_identity_route((512, 512), &[1, 32, 2048], Dtype::Float32, true);
                observe_cfg_branch(None);
                let positive = CaActivity::default();
                let negative = CaActivity::default();
                positive.record_forward();
                positive.record_attention(
                    0.0,
                    mlx_gen::attention::AttentionBudget::CONSTRAINED,
                    1,
                    16,
                    1024,
                    32,
                );
                observe_completed_route(&positive, &negative);
                mlx_gen::image_execution::sampled_native(
                    "flow_match",
                    "sigma",
                    Dtype::Float32,
                    &[1.0, 0.0],
                    42,
                );
                Ok(GenerationOutput::Images(vec![Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                }]))
            },
        )
        .unwrap();
        let policy = match &completed.execution.resolved().unwrap().quality_policy {
            gen_core::generator::ExecutionFact::Resolved { value } => value,
            other => panic!("expected resolved quality policy: {other:?}"),
        };
        assert_eq!(policy["pulid.ca_budget_selected"], "true");
        assert_eq!(policy["pulid.ca_active"], "false");
        assert_eq!(policy["pulid.ca_bounded_attention"], "false");
    }

    /// The load error string, matched (not `unwrap_err` — `Box<dyn Generator>` isn't `Debug`).
    fn load_err(spec: &LoadSpec) -> String {
        match load_pulid_flux(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected load_pulid_flux to error"),
        }
    }

    /// sc-13664: the identity slot is **required** — an absent `LoadSpec::identity`, or an absent
    /// `encoder` / `eva` / `face_dir` sub-field, is a **load-time** actionable error naming the slot,
    /// resolved before any FLUX backbone I/O. No `PULID_*` env / HF-cache fallback.
    #[test]
    fn load_requires_identity_slot_and_every_subfield() {
        // No identity slot at all — the actionable error names the slot (and confirms, for a migrating
        // caller, that the historical `PULID_*` env fallback is gone).
        let err = load_err(&LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/flux".into(),
        )));
        assert!(err.contains("requires the identity weights"), "got: {err}");
        assert!(
            !err.contains("PULID_FLUX_WEIGHTS"),
            "the message must not resurrect the deleted env var as usable: {err}"
        );

        // Identity present but each sub-field missing in turn → named error for the missing one.
        let base = |id: IdentityWeights| {
            let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/flux".into()));
            spec.identity = Some(id);
            spec
        };
        let file = |p: &str| Some(WeightsSource::File(p.into()));
        let dir = |p: &str| Some(WeightsSource::Dir(p.into()));

        let no_encoder = base(IdentityWeights {
            encoder: None,
            eva: file("/x/eva.safetensors"),
            face_dir: dir("/x/face"),
        });
        assert!(load_err(&no_encoder).contains("requires the identity encoder"));

        let no_eva = base(IdentityWeights {
            encoder: file("/x/pulid.safetensors"),
            eva: None,
            face_dir: dir("/x/face"),
        });
        assert!(load_err(&no_eva).contains("requires the identity EVA tower"));

        let no_face = base(IdentityWeights {
            encoder: file("/x/pulid.safetensors"),
            eva: file("/x/eva.safetensors"),
            face_dir: None,
        });
        assert!(load_err(&no_face).contains("requires the identity face-analysis dir"));
    }

    /// sc-13664: a set-but-nonexistent identity path errors up front with the spec-side message (the
    /// `spec_path` existence check), naming the sub-field — not a bare later I/O error, no env var name.
    #[test]
    fn identity_encoder_nonexistent_path_errors_with_spec_message() {
        let src = WeightsSource::File("/nonexistent/spec_pulid.safetensors".into());
        let err = spec_path(&src, "encoder").unwrap_err().to_string();
        assert!(err.contains("LoadSpec identity encoder"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");
    }

    fn img() -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![0u8; 3],
        }
    }

    /// F-094: a single reference face is accepted (its strength becomes the id_weight, default 1.0).
    #[test]
    fn select_reference_accepts_single_reference() {
        let cond = vec![Conditioning::Reference {
            image: img(),
            strength: Some(2.0),
        }];
        let (_, weight) = select_reference_face(&cond).unwrap();
        assert_eq!(weight, 2.0);
    }

    /// F-094: a non-Reference conditioning (e.g. a stray Mask/Control) is rejected rather than
    /// silently dropped when `generate` clears the conditioning.
    #[test]
    fn select_reference_rejects_unsupported_conditioning() {
        let cond = vec![
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
            Conditioning::Mask { image: img() },
        ];
        let err = select_reference_face(&cond).unwrap_err().to_string();
        assert!(err.contains("unsupported"), "got: {err}");
    }

    /// F-026: `validate` delegates to PuLID's OWN descriptor floor, which omits the `hyper` sampler
    /// (PuLID doesn't load the dev Hyper LoRA). A `hyper` request must be rejected — not silently
    /// rendered WITHOUT the LoRA the way the dev backbone descriptor would have allowed.
    #[test]
    fn own_descriptor_floor_rejects_hyper_sampler() {
        let caps = descriptor().capabilities;
        assert!(
            !caps.samplers.contains(&"hyper"),
            "PuLID's descriptor must omit the hyper sampler"
        );
        let base = GenerationRequest {
            prompt: "a portrait".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        // hyper is not advertised → typed Unsupported.
        let hyper = GenerationRequest {
            sampler: Some("hyper".into()),
            ..base.clone()
        };
        assert!(matches!(
            caps.validate_request(descriptor().id, &hyper),
            Err(gen_core::Error::Unsupported(_))
        ));
        // A curated sampler + PuLID's real-CFG knobs (true_cfg / negative_prompt) now validate, since
        // the descriptor advertises them (sc-3075).
        let ok = GenerationRequest {
            sampler: Some("euler".into()),
            true_cfg: Some(2.0),
            negative_prompt: Some("blurry".into()),
            ..base
        };
        assert!(
            caps.validate_request(descriptor().id, &ok).is_ok(),
            "curated sampler + real-CFG should pass PuLID's floor"
        );
    }

    /// Zero *scalar* arrays for every listed key — the EVA/IdFormer constructors only clone the
    /// required arrays (no shape validation at load), so scalars suffice for a tower that exists
    /// but is never forwarded.
    fn zero_weights(keys: Vec<String>) -> Weights {
        let mut w = Weights::empty();
        for k in keys {
            w.insert(k, Array::from_f32(0.0));
        }
        w
    }

    /// A fully weight-free `PulidFlux`: the weightless FLUX-dev backbone
    /// (`Flux1::new_for_tests`), a tiny zero-weight EVA tower + IdFormer, and the degenerate
    /// `FaceAnalysis::new_for_tests` fixture. Constructible via struct literal because this tests
    /// module is a child of `pulid_flux` (private fields in scope). Only the pre-weight `generate`
    /// prefix (validate-first) may run — any forward would error on the zero-weight shapes.
    fn weightless_pulid() -> PulidFlux {
        // Tiny EVA tower (depth 1, 4-wide) — the constructor honors any EvaConfig (F-082).
        let cfg = EvaConfig {
            image_size: 2,
            patch: 1,
            embed_dim: 4,
            depth: 1,
            num_heads: 1,
            proj_dim: 1,
            pt_seq_len: 2,
            rope_theta: 10000.0,
            hidden_capture: vec![0],
        };
        let mut keys: Vec<String> = [
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "cls_token",
            "pos_embed",
            "norm.weight",
            "norm.bias",
            "head.weight",
            "head.bias",
        ]
        .map(String::from)
        .to_vec();
        keys.extend(
            [
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "attn.q_proj.weight",
                "attn.q_bias",
                "attn.k_proj.weight",
                "attn.v_proj.weight",
                "attn.v_bias",
                "attn.inner_attn_ln.weight",
                "attn.inner_attn_ln.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "mlp.w1.weight",
                "mlp.w1.bias",
                "mlp.w2.weight",
                "mlp.w2.bias",
                "mlp.ffn_ln.weight",
                "mlp.ffn_ln.bias",
                "mlp.w3.weight",
                "mlp.w3.bias",
            ]
            .iter()
            .map(|l| format!("blocks.0.{l}")),
        );
        let eva = EvaVisionTransformer::from_weights(&zero_weights(keys), "", cfg).unwrap();

        // IdFormer with depth 0 — no perceiver layers; just latents/proj_out + the 6 mapping MLPs.
        let mlp = [
            "0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias", "4.weight", "4.bias",
            "6.weight", "6.bias",
        ];
        let mut ikeys: Vec<String> = vec!["enc.latents".into(), "enc.proj_out".into()];
        for m in [
            "id_embedding_mapping",
            "mapping_0",
            "mapping_1",
            "mapping_2",
            "mapping_3",
            "mapping_4",
        ] {
            ikeys.extend(mlp.iter().map(|l| format!("enc.{m}.{l}")));
        }
        let idformer = IdFormer::from_weights(
            &zero_weights(ikeys),
            "enc",
            IdFormerConfig {
                depth: 0,
                ..IdFormerConfig::default()
            },
        )
        .unwrap();

        let flux = Flux1::new_for_tests(FluxVariant::Dev);
        let mut memory_strategy = flux.memory_strategy_contract().unwrap().clone();
        memory_strategy.provider_id = MODEL_ID.into();
        memory_strategy.calibration = None;
        PulidFlux {
            descriptor: descriptor(),
            flux,
            eva,
            idformer,
            pulid: Weights::empty(),
            face: FaceAnalysis::new_for_tests().unwrap(),
            memory_strategy,
            loaded_spec: LoadSpec::new(WeightsSource::Dir("/weightless-pulid".into())),
            identity_inventory: None,
        }
    }

    #[test]
    fn registered_memory_fixture_is_an_executable_single_phase_request() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/flux".into()))
            .with_quant(Quant::Q4)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = crate::memory_strategy::weights_free_contract(&spec).unwrap();
        let fixture = crate::memory_strategy::registered_fixture(
            &spec,
            &contract,
            gen_core::MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);

        assert!(!fixture.context.has_phases);
        assert!(fixture.request.phases.is_none());
        assert_eq!(fixture.context.geometry.reference_count, 1);
        assert_eq!(fixture.request.conditioning.len(), 1);
        Generator::validate(&weightless_pulid(), &fixture.request).unwrap();
    }

    /// F-011 (sc-12463): `generate_impl` self-validates FIRST (`self.validate(req)?`, the sibling
    /// convention — chroma/svd/ideogram), so a caller that invokes `generate` WITHOUT a prior
    /// `validate` still gets the typed `Unsupported` for a `hyper` request instead of a silently
    /// degraded render (the FLUX-dev backbone would happily render WITHOUT the Hyper-FLUX LoRA
    /// PuLID never loads). Driven end-to-end through `Generator::generate_planned` (with a plan
    /// compiled before construction from the same request minus the sampler) on a weight-free
    /// instance: the capability floor rejects the request before any face/EVA/backbone work, so no
    /// weights are ever touched. Mutation-pins the regression — with the `self.validate(req)?` line
    /// deleted, the hyper request falls through to the plan check or the zero-weight identity
    /// stack and fails without the sampler-floor message, so this test FAILS.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn generate_without_prior_validate_rejects_hyper_typed_unsupported() {
        let model = weightless_pulid();
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 1024,
            height: 1024,
            sampler: Some("hyper".into()),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        // The reference-face half of `validate` accepts this request, so the sampler floor is the
        // ONLY rejection — and it must arrive through `generate` itself, typed.
        assert!(select_reference_face(&req.conditioning).is_ok());
        let plan = compile_constructed_plan(&GenerationRequest {
            sampler: None,
            ..req.clone()
        })
        .unwrap();
        let err = Generator::generate_planned(&model, &req, &plan, None, &mut |_| {}).unwrap_err();
        assert!(
            matches!(&err, gen_core::Error::Unsupported(message) if message.contains("unsupported sampler")),
            "got: {err:?}"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED pulid_flux::tests::generate_without_prior_validate_rejects_hyper_typed_unsupported");
    }

    /// Two reference faces are rejected, and an empty request is rejected as missing.
    #[test]
    fn select_reference_rejects_multiple_and_missing() {
        let two = vec![
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
        ];
        assert!(select_reference_face(&two)
            .unwrap_err()
            .to_string()
            .contains("exactly one"));
        assert!(select_reference_face(&[])
            .unwrap_err()
            .to_string()
            .contains("required"));
    }

    /// F-138: a multi-face reference reaches ArcFace exactly once, for the first detection. The
    /// face stack's `detect` contract sorts detections largest-first, so this is the selected face
    /// rather than the first detector-score result.
    #[test]
    fn largest_face_embedding_does_one_embed_for_multi_face_detection() {
        let detections = vec![
            Detection {
                bbox: [0.0, 0.0, 20.0, 20.0],
                kps: [[0.0; 2]; 5],
                score: 0.9,
            },
            Detection {
                bbox: [0.0, 0.0, 10.0, 10.0],
                kps: [[1.0; 2]; 5],
                score: 0.99,
            },
            Detection {
                bbox: [0.0, 0.0, 5.0, 5.0],
                kps: [[2.0; 2]; 5],
                score: 0.8,
            },
        ];
        let mut embed_calls = 0;
        let selected = PulidFlux::largest_face_embedding(
            || Ok(detections),
            |detection| {
                embed_calls += 1;
                Ok(detection.bbox)
            },
        )
        .unwrap();

        assert_eq!(embed_calls, 1, "only the selected face may reach ArcFace");
        assert_eq!(selected, [0.0, 0.0, 20.0, 20.0]);
    }

    #[test]
    fn descriptor_advertises_flux1_dev_pag_and_records_pag_policy() {
        let desc = descriptor();
        assert_eq!(
            desc.capabilities.pag,
            Some(FluxVariant::Dev.pag_capability())
        );

        let pag_req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 512,
            height: 512,
            steps: Some(4),
            pag: Some(gen_core::PagConfig {
                scale: 1.5,
                ..Default::default()
            }),
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        let resolved = desc
            .capabilities
            .resolve_pag(desc.id, &pag_req)
            .unwrap()
            .expect("active pag");
        let pag_report = resolved.execution_report(4);
        assert_eq!(pag_report.layers, vec![28]);
        assert_eq!(pag_report.active_steps, 4);
        assert_eq!(pag_report.total_steps, 4);

        let completed = mlx_gen::image_execution::complete_image_report_attempt(
            &desc,
            &CancelFlag::new(),
            || {
                let pag = Some(resolved.clone());
                mlx_gen::image_execution::policy_debug("pulid.pag", &pag);
                mlx_gen::image_execution::sampled_native(
                    "flow_match",
                    "sigma",
                    Dtype::Float32,
                    &[1.0, 0.0],
                    42,
                );
                let mut report =
                    gen_core::GenerationReport::from_output(GenerationOutput::Images(vec![img()]));
                report.pag = Some(pag_report.clone());
                Ok(report)
            },
        )
        .unwrap();
        assert_eq!(completed.generation.pag, Some(pag_report));
        let policy = match &completed.execution.resolved().unwrap().quality_policy {
            gen_core::generator::ExecutionFact::Resolved { value } => value,
            other => panic!("expected resolved quality policy: {other:?}"),
        };
        assert!(policy["pulid.pag"].as_str().unwrap().contains("Some("));
    }
}
