//! # candle-gen-seedvr2
//!
//! The **SeedVR2** provider crate for [`candle-gen`](candle_gen) — the Windows/CUDA sibling of
//! `mlx-gen-seedvr2` (epic 4811 / sc-5157). A native-candle port of the ByteDance one-step
//! diffusion-transformer super-resolution upscaler:
//!
//! 1. **DiT** ([`dit`]) — a dual-stream MMDiT with adaptive **spatiotemporal window attention**
//!    (`window=(T,H,W)=(4,3,3)`, shifted on odd layers), 3D axial RoPE, QK-norm, SwiGLU, AdaLN.
//! 2. **3D causal video VAE** ([`vae`]) — `CausalConv3d` (candle has no conv3d → conv2d temporal-sum,
//!    see [`conv3d`]) encoder/decoder with `temporal_down/up_blocks=2`, GroupNorm, per-frame attention.
//! 3. **One-step Euler** + a precomputed negative-prompt embedding (bundled, no runtime text encoder).
//!
//! **Surface (`Modality::Both`): a one-step super-resolution upscaler over image AND video**,
//! dispatched on the request's conditioning — a `Reference` LR image → `GenerationOutput::Images`
//! (sc-5157, the 3B engine), or a `VideoClip` LR frame sequence → `GenerationOutput::Video` (sc-5926:
//! the 5-D temporal pass with chunking/overlap cross-fade, a VRAM-budgeted chunk sizer, and HD spatial
//! tiling; see [`video`] + [`pipeline`]). 7B + int8/int4 quant is sc-5927; worker wiring/gating is
//! sc-5928. `backend = "candle"`, `mac_only = false`.

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_gen::candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_gen::candle_core::Tensor, ()>::Err(()));
};

pub mod color;
pub mod config;
pub mod conv3d;
pub mod convert;
pub mod dit;
pub mod nn;
pub mod pipeline;
pub mod quant;
pub mod vae;
pub mod video;
pub mod weights;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant,
    WeightsSource,
};

pub use config::{DitConfig, TIMESTEP, VAE_SCALE};
pub use pipeline::{Seedvr2Driver, Seedvr2Forward, Seedvr2Pipeline};
pub use video::{Seedvr2OutputLifecycle, Seedvr2RenderPlan, Seedvr2Route};

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
pub const MODEL_ID_7B: &str = "seedvr2_7b";
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";
const DIT_FILE_7B: &str = "seedvr2_ema_7b_fp16.safetensors";
/// Output fps when the request omits one (the worker normally supplies the source cadence).
const DEFAULT_FPS: u32 = 24;

/// The DiT checkpoint file + transformer config for a registered id (3B default; 7B is the
/// pixel-mode-RoPE variant — sc-5927). The VAE is shared across both.
fn variant(id: &str) -> (&'static str, DitConfig) {
    if id == MODEL_ID_7B {
        (DIT_FILE_7B, DitConfig::seedvr2_7b())
    } else {
        (DIT_FILE_3B, DitConfig::seedvr2_3b())
    }
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::SEEDVR2_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id,
        family: "seedvr2",
        backend: "candle",
        modality: Modality::Both, // image (Reference) + video (VideoClip) upscaling
        capabilities: Capabilities {
            // `supports_negative_prompt` is deferred to its `false` default rather than declared:
            // SeedVR2 carries a precomputed negative embedding and exposes no prompt surface.
            supports_guidance: false, // one-step, guidance fixed at 1.0
            supports_true_cfg: false,
            // the LR input image (image upscale) or LR frame sequence (video upscale)
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
            // Bespoke-by-architecture (epic 7114 P4, sc-7123): SeedVR2 is a ONE-STEP restoration
            // transformer (fixed timestep 1000.0, `denoised = noise − dit_out`), NOT a multi-step
            // rectified-flow ODE — there is no sigma schedule to integrate and no sampler/scheduler axis
            // to expose. It keeps its native single-step `seedvr2_euler` descriptor verbatim; the unified
            // curated solvers/schedulers do not apply.
            samplers: vec!["seedvr2_euler"],
            schedulers: vec!["seedvr2_euler"],
            min_size: VAE_SCALE,
            max_size: 4096,
            max_count: 8,
            supported_quants: &[Quant::Q4, Quant::Q8], // Linear-only DiT quant (sc-5927)
            component_precision_floors: &[],
            ..Default::default()
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}
pub fn descriptor_3b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_3B)
}
pub fn descriptor_7b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_7B)
}

/// The lazy candle SeedVR2 generator (one-step image/video upscaler). The pipeline is loaded on first
/// `generate` and cached behind a `Mutex` for the worker's `Arc<dyn Generator>` reuse. `dit_file`/`cfg`
/// select the 3B (default) or 7B variant; `quant` applies int8/int4 DiT quantization at load (sc-5927).
pub struct Seedvr2Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    dit_file: &'static str,
    cfg: DitConfig,
    quant: Option<Quant>,
    pipe: Mutex<Option<std::sync::Arc<Seedvr2Pipeline>>>,
}

/// The LR input image carried by the request's `Reference` conditioning.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

fn fold_tile_progress(
    image_idx: usize,
    tile_idx: usize,
    n_tiles: usize,
    count: usize,
) -> (u32, u32) {
    let total = count * n_tiles.max(1);
    let current = (image_idx * n_tiles.max(1) + tile_idx).min(total);
    (current as u32, total as u32)
}

impl Seedvr2Generator {
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<Seedvr2Pipeline>> {
        // `cached` recovers a poisoned lock (sc-9015) internally; the inner `?` bridges the
        // candle-side load/quantize errors into `gen_core::Error`.
        candle_gen::cached(&self.pipe, || {
            let mut p = Seedvr2Pipeline::load(
                &self.root,
                self.dit_file,
                &self.cfg,
                self.dtype,
                &self.device,
            )?;
            // sc-5927: int8/int4 quantize the DiT Linears at load (the VAE stays dense).
            if let Some(q) = self.quant {
                p.quantize(q)?;
            }
            Ok(std::sync::Arc::new(p))
        })
    }

    fn current_weights_bytes(&self) -> usize {
        self.pipe
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|pipe| pipe.weights_bytes()))
            .unwrap_or_else(|| video::estimated_weights_bytes(self.descriptor.id))
    }

    fn compile_render_plan(
        &self,
        req: &GenerationRequest,
        weights_bytes: usize,
    ) -> gen_core::Result<Seedvr2RenderPlan> {
        Seedvr2RenderPlan::compile(
            &self.descriptor,
            req,
            weights_bytes,
            video::safe_budget_gib(),
            None,
        )
    }
}

impl Generator for Seedvr2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.compile_render_plan(req, self.current_weights_bytes())?;
        Ok(())
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture(self, req, on_progress)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = self.pipeline()?;
        let plan = self.compile_render_plan(req, pipe.weights_bytes())?;
        let base_seed = plan.base_seed;
        let softness = plan.softness;

        if let Some(clip) = req.video_clips().into_iter().find(|c| !c.frames.is_empty()) {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let mut on_step = |done: usize, total: usize| {
                on_progress(Progress::Step {
                    current: done as u32,
                    total: total as u32,
                });
            };
            let frames = pipe.generate_video_for_route(
                clip.frames,
                req.width as usize,
                req.height as usize,
                base_seed,
                softness,
                &plan.route,
                Some(&req.cancel),
                Some(&mut on_step),
            )?;
            on_progress(Progress::Decoding);
            return plan.finish_output(GenerationOutput::Video {
                frames,
                fps: req.fps.unwrap_or(DEFAULT_FPS),
                audio: None,
            });
        }

        let image = reference_image(req).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{}: requires a Reference image (image upscale)",
                self.descriptor.id
            ))
        })?;
        let count = req.count as usize;
        let mut idx = 0usize;
        let out = candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            let image_idx = idx;
            idx += 1;
            let mut step_cb = |tile_idx: usize, n_tiles: usize| {
                let (current, total) = fold_tile_progress(image_idx, tile_idx, n_tiles, count);
                on_progress(Progress::Step { current, total });
            };
            Ok(pipe.generate_still_for_route(
                image,
                req.width as usize,
                req.height as usize,
                seed,
                softness,
                &plan.route,
                Some(&req.cancel),
                Some(&mut step_cb),
            )?)
        })?;
        on_progress(Progress::Decoding);
        plan.finish_output(GenerationOutput::Images(out))
    }
}

/// Construct a lazy candle SeedVR2 generator.
fn load_with(spec: &LoadSpec, id: &'static str) -> gen_core::Result<Box<dyn Generator>> {
    Seedvr2RenderPlan::validate_load_spec(id, spec)?;
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id}: expects a numz/SeedVR2_comfyUI checkpoint directory, not a single file"
            )))
        }
    };
    let device = candle_gen::default_device()?;
    // Precision::Bf16 is the "native dense" sentinel → bf16 on GPU, f32 on CPU (the parity dtype);
    // Fp32 forces full precision everywhere.
    let dtype = match spec.precision {
        Precision::Fp32 => DType::F32,
        Precision::Bf16 if device.is_cpu() => DType::F32,
        Precision::Bf16 => DType::BF16,
    };
    let (dit_file, cfg) = variant(id);
    Ok(Box::new(Seedvr2Generator {
        descriptor: descriptor_for(id),
        root,
        device,
        dtype,
        dit_file,
        cfg,
        quant: spec.quantize,
        pipe: Mutex::new(None),
    }))
}

fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID)
}
fn load_registered_3b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_3B)
}
fn load_registered_7b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_7B)
}

candle_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor => load_registered
}
candle_gen::register_generators! {
    pub(crate) const THREE_B_REGISTRATION = descriptor_3b => load_registered_3b
}
candle_gen::register_generators! {
    pub(crate) const SEVEN_B_REGISTRATION = descriptor_7b => load_registered_7b
}

/// Add all Candle SeedVR2 generators to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(BASE_REGISTRATION)
        .register_generator(THREE_B_REGISTRATION)
        .register_generator(SEVEN_B_REGISTRATION)
}

/// Build the complete explicit Candle SeedVR2 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["seedvr2", "seedvr2_3b", "seedvr2_7b"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_seedvr2_image_and_video() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "seedvr2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Both); // image (Reference) + video (VideoClip)
        assert!(!d.capabilities.mac_only);
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::VideoClip));
        assert_eq!(d.capabilities.min_size, VAE_SCALE);
        // sc-5927: int8/int4 DiT quant is now advertised.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn all_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B, MODEL_ID_7B] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/seedvr2".into()));
            let g = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .expect("seedvr2 is registered");
            assert_eq!(g.descriptor().family, "seedvr2");
            assert_eq!(g.descriptor().backend, "candle");
        }
    }

    #[test]
    fn variant_selects_7b_config_and_file() {
        let (file_3b, cfg_3b) = variant(MODEL_ID_3B);
        assert_eq!(file_3b, DIT_FILE_3B);
        assert_eq!(cfg_3b.num_layers, 32);
        assert!(!cfg_3b.rope_pixel);

        let (file_7b, cfg_7b) = variant(MODEL_ID_7B);
        assert_eq!(file_7b, DIT_FILE_7B);
        assert_eq!(cfg_7b.num_layers, 36);
        assert_eq!(cfg_7b.vid_dim, 3072);
        assert!(cfg_7b.rope_pixel); // pixel-mode RoPE — the 7B delta (sc-5197)
        assert!(!cfg_7b.rope_on_text);
        assert!(!cfg_7b.swiglu_mlp); // GELU MLP
    }

    #[test]
    fn load_accepts_quant_rejects_single_file() {
        use candle_gen::gen_core::Quant;
        // Quant is now wired (sc-5927) — load succeeds lazily and carries the level.
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(q);
            let g = load_with(&spec, MODEL_ID).expect("quant is wired");
            assert_eq!(g.descriptor().family, "seedvr2");
        }
        // A single-file weights source is still rejected.
        let file = LoadSpec::new(WeightsSource::File("/w.safetensors".into()));
        assert!(load_with(&file, MODEL_ID).is_err());
    }

    fn frame() -> Image {
        Image {
            width: 16,
            height: 16,
            pixels: vec![0u8; 16 * 16 * 3],
        }
    }

    /// An upscale request whose (single) VideoClip carries the given sc-20263 knob values.
    fn clip_request(frame_idx: i32, strength: f32) -> GenerationRequest {
        GenerationRequest {
            width: 32,
            height: 32,
            count: 1,
            conditioning: vec![Conditioning::VideoClip {
                frames: vec![frame()],
                frame_idx,
                strength,
            }],
            ..Default::default()
        }
    }

    /// A lazily-loaded generator — `validate` runs entirely before any weight is touched, so the
    /// bogus snapshot dir never matters.
    fn unloaded() -> Box<dyn Generator> {
        load_with(
            &LoadSpec::new(WeightsSource::Dir("/nonexistent-seedvr2".into())),
            MODEL_ID,
        )
        .expect("the seedvr2 loader is lazy")
    }

    /// sc-20263 — `VideoClip.strength` and `frame_idx` were silently dropped (`generate` binds only
    /// `clip.frames`). Each is now a typed `Unsupported` naming the field and the model, raised
    /// through `Generator::validate` — the seam `generate` calls first.
    ///
    /// Non-default values throughout: at the contract defaults the refusal must not fire, which the
    /// companion test pins.
    #[test]
    fn non_default_video_clip_knobs_are_refused_by_name() {
        let g = unloaded();
        for (req, field) in [
            (clip_request(0, 0.6), "strength"),
            (clip_request(4, 1.0), "frame_idx"),
        ] {
            let err = g
                .validate(&req)
                .expect_err("a non-default knob must be refused");
            assert!(
                matches!(err, gen_core::Error::Unsupported(_)),
                "{field}: typed Unsupported, got {err:?}"
            );
            let msg = err.to_string();
            assert!(msg.contains(field), "{field}: names the field: {msg}");
            assert!(msg.contains(MODEL_ID), "{field}: names the model: {msg}");
        }
    }

    /// sc-20263 — the refusal must inspect **every** clip, not the first. `generate` selects the
    /// first NON-empty clip, so a bad value on clip two is exactly the one that would otherwise
    /// stay silently dropped.
    #[test]
    fn a_bad_knob_on_the_second_clip_is_still_refused() {
        let req = GenerationRequest {
            width: 32,
            height: 32,
            count: 1,
            conditioning: vec![
                Conditioning::VideoClip {
                    frames: Vec::new(),
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::VideoClip {
                    frames: vec![frame()],
                    frame_idx: 7,
                    strength: 1.0,
                },
            ],
            ..Default::default()
        };
        let err = unloaded()
            .validate(&req)
            .expect_err("clip two's frame_idx must be refused");
        assert!(err.to_string().contains("frame_idx"), "got: {err}");
    }

    /// sc-20263 — the refusal fires only on a value a caller actually set. The contract defaults
    /// (`strength = 1.0`, `frame_idx = 0`) — every request SceneWorks builds today — validate, as
    /// does an image-upscale request that carries no clip at all.
    #[test]
    fn default_video_clip_knobs_still_pass_unchanged() {
        let g = unloaded();
        g.validate(&clip_request(0, 1.0))
            .expect("a default-valued VideoClip must still validate");
        g.validate(&GenerationRequest {
            width: 32,
            height: 32,
            count: 1,
            conditioning: vec![Conditioning::Reference {
                image: frame(),
                strength: None,
            }],
            ..Default::default()
        })
        .expect("an image upscale carries no clip and is unaffected");
    }

    #[test]
    fn seedvr2_render_plan_resolves_all_five_routes_and_validates_pre_load() {
        let g = unloaded();
        let desc = g.descriptor();
        let lr = frame();
        let wb = (7.3 * video::GIB) as usize;

        let still_req = GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            conditioning: vec![Conditioning::Reference {
                image: lr.clone(),
                strength: None,
            }],
            ..Default::default()
        };
        g.validate(&still_req).unwrap();
        let p_single = Seedvr2RenderPlan::compile(desc, &still_req, wb, 64.0, None).unwrap();
        assert_eq!(p_single.route, Seedvr2Route::StillSinglePass);

        let large_still = GenerationRequest {
            width: 2048,
            height: 2048,
            ..still_req
        };
        g.validate(&large_still).unwrap();
        let p_still_tiled =
            Seedvr2RenderPlan::compile(desc, &large_still, wb, 256.0, None).unwrap();
        assert!(matches!(
            p_still_tiled.route,
            Seedvr2Route::StillSpatialTiled { .. }
        ));

        let p_chunked = Seedvr2RenderPlan::compile(
            desc,
            &GenerationRequest {
                width: 512,
                height: 512,
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![lr.clone(); 20],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..clip_request(0, 1.0)
            },
            wb,
            108.0,
            None,
        )
        .unwrap();
        assert!(matches!(
            p_chunked.route,
            Seedvr2Route::VideoTemporalChunked { .. }
        ));

        let p_per_frame = Seedvr2RenderPlan::compile(
            desc,
            &GenerationRequest {
                width: 1024,
                height: 1024,
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![lr.clone(); 8],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..clip_request(0, 1.0)
            },
            wb,
            20.0,
            None,
        )
        .unwrap();
        assert!(matches!(
            p_per_frame.route,
            Seedvr2Route::VideoPerFrame { frame_count: 8 }
        ));

        let p_vid_tiled = Seedvr2RenderPlan::compile(
            desc,
            &GenerationRequest {
                width: 2048,
                height: 2048,
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![lr; 4],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..clip_request(0, 1.0)
            },
            wb,
            256.0,
            None,
        )
        .unwrap();
        assert!(matches!(
            p_vid_tiled.route,
            Seedvr2Route::VideoSpatialTiled { frame_count: 4, .. }
        ));
    }
}
