//! # candle-gen-mochi
//!
//! The **Mochi 1** (`genmo/mochi-1-preview`, Apache-2.0) text-to-video provider for
//! [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of `mlx-gen-mochi`. Mochi is a
//! T5-XXL-conditioned dual-stream MMDiT (the **AsymmDiT**) with an asymmetric 3-D causal-conv VAE
//! (6× temporal, 8× spatial). It has **no** `candle-transformers` reference: the masked T5 encode
//! ([`text_encoder`], run through the reused `candle_gen::text_encoder::T5Encoder` + the
//! tokenizer key-padding mask Mochi's `_get_t5_prompt_embeds` applies), the learned 3-D RoPE ([`rope`]),
//! the linear-quadratic flow-match scheduler ([`scheduler`]), the dual-stream AsymmDiT denoiser
//! ([`transformer`]), and the AsymmVAE decoder ([`vae`], on a from-scratch conv2d-tap [`conv3d`]) are
//! all ported here, preserving the exact `mlx-gen-mochi` math.
//!
//! **txt2video, true CFG:** Mochi is **not** distilled, so it exposes negative-prompt + `guidance`
//! true classifier-free guidance over the `[neg, pos]` batch. The denoise runs the inverted
//! linear-quadratic Euler loop with the CFG recombine `uncond + g·(cond − uncond)` inside the predict
//! step ([`pipeline`]).
//!
//! **Dtypes:** the AsymmDiT + T5 store weights at **bf16** (the checkpoint's native dtype; the 10B DiT
//! does not fit f32 on a single consumer GPU), the AsymmVAE runs **f32**. The AsymmDiT **computes in
//! f32** (the MLX parity regime — the `nn::linear_*` helpers upcast each bf16 weight to f32 at the
//! matmul). `backend = "candle"`, `mac_only = false`. Quant tiers ship as pre-quantized per-tier
//! checkpoints (epic 1788 / A6), *not* on-the-fly requant, so `supported_quants` is empty.

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_gen::candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_gen::candle_core::Tensor, ()>::Err(()));
};

pub mod config;
pub mod conv3d;
pub mod nn;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod tier;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use candle_gen::candle_core::DType;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, RenderLoadBinding, VideoPlanLayout, WeightsSource,
};

pub use config::{MochiConfig, MochiVaeConfig};
pub use pipeline::{denoise, frames_to_images, to_uint8_frames, Components, Pipeline};
pub use rope::{get_positions, MochiRope};
pub use scheduler::{cfg_combine, linear_quadratic_schedule, MochiScheduler};
pub use text_encoder::{encode_prompt, load_indexed_var_builder, MochiT5, MochiTextConditioning};
pub use tokenizer::{load_tokenizer, EOS_TOKEN_ID, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID};
pub use transformer::{
    load_transformer_var_builder, MochiAttention, MochiDitConfig, MochiTransformer3DModel,
    MochiTransformerBlock,
};
pub use vae::{MochiVaeDecoder, DEFAULT_DECODE_CHUNK_FRAMES};

/// Public provider id: `"mochi_1"`.
pub const MODEL_ID: &str = "mochi_1";

/// The AsymmDiT + T5 weight-storage dtype (the checkpoint's native bf16; the DiT then computes f32).
pub const DIT_DTYPE: DType = DType::BF16;
/// The AsymmVAE compute dtype (the decoder is numerically f32-only — bf16 intermediates reach O(100)).
pub const VAE_DTYPE: DType = DType::F32;

/// Width/height must be divisible by 16 (VAE 8× spatial × DiT patch 2). Exposed as the pinned-engine
/// stride SceneWorks ties `requiresDimensionsMultipleOf` to (sc-12587); mirrors mlx's
/// `mlx_gen_mochi::SIZE_MULTIPLE` and `wan::config::SIZE_MULTIPLE_14B`.
pub use gen_core_family::mochi::SIZE_MULTIPLE;

/// Diffusers `MochiPipeline` default `guidance_scale`.
pub(crate) const DEFAULT_GUIDANCE: f32 = 4.5;
/// Mochi renders ~30 fps.
pub(crate) const DEFAULT_FPS: u32 = 30;

/// Stable identity + advertised capabilities for Mochi 1 (text-to-video, true CFG, no audio).
///
/// Mochi is **text-to-video only** in the base preview (no audio, no I2V). It is **not** distilled, so
/// it exposes **true CFG** (negative prompt + `guidance` scale). Quant tiers ship as pre-quantized
/// per-tier checkpoints (epic 1788 / A6) — the tier dir IS the quant selection — *not* on-the-fly
/// requant, so [`Capabilities::supported_quants`] is empty and `load` rejects an on-the-fly
/// `spec.quantize` (honouring one only when it matches the tier dir's manifest, [`tier`]).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::MOCHI_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "mochi",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Not distilled → true classifier-free guidance over a [neg, pos] batch.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Text-to-video only in the base preview (I2V = a follow-on).
            conditioning: Vec::new(),
            // A single fixed flow-match Euler integrator is wired; no selectable sampler/scheduler axis.
            samplers: Vec::new(),
            // Width/height must be divisible by SIZE_MULTIPLE (VAE 8× spatial × DiT patch 2). 480p target = 848×480.
            min_size: SIZE_MULTIPLE,
            max_size: 1280,
            max_count: 1,
            // Quant tiers are pre-quantized per-tier checkpoints (epic 1788 / A6) — NOT on-the-fly requant.
            supported_quants: &[] as &[Quant],
            ..Default::default()
        },
    }
}

/// The lazy candle Mochi 1 generator. `spec.weights` is a `genmo/mochi-1-preview` diffusers snapshot
/// dir (`tokenizer/` vendored, `text_encoder/`, `transformer/`, `vae/`).
pub struct MochiGenerator {
    load_binding: RenderLoadBinding,
    descriptor: ModelDescriptor,
    pipe: Pipeline,
    components: Mutex<Option<Components>>,
}

impl MochiGenerator {
    /// Only a successful load owns a layout. Cold preflight must reread the resolved
    /// snapshot so a failed acquisition can be retried after its files are repaired.
    fn component_layout(&self) -> candle_gen::Result<Arc<pipeline::ComponentLayout>> {
        if let Some(resident) = candle_gen::lock_recover(&self.components).as_ref() {
            return Ok(Arc::clone(&resident.layout));
        }
        self.pipe.component_layout()
    }

    fn generate_scoped(
        &self,
        req: &GenerationRequest,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let layout = self.component_layout()?;
        let (frames, fps) = self.pipe.render_scoped(
            req,
            &layout.vae_cfg,
            || {
                let comps = candle_gen::cached(&self.components, || {
                    self.pipe.load_components_from(Arc::clone(&layout))
                })?;
                // Another first request can win the cold cache. Its actual codec must
                // agree with the geometry already entered by this request.
                if comps.layout.vae_cfg.temporal_expansions != layout.vae_cfg.temporal_expansions
                    || comps.layout.vae_cfg.spatial_expansions != layout.vae_cfg.spatial_expansions
                {
                    return Err(candle_gen::CandleError::Unsupported(
                        "Mochi codec geometry changed during component acquisition".into(),
                    ));
                }
                Ok(comps)
            },
            scope,
            on_progress,
        )?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

impl Generator for MochiGenerator {
    fn video_load_binding(&self) -> Option<&RenderLoadBinding> {
        Some(&self.load_binding)
    }

    fn validate_video_plan(
        &self,
        plan: &gen_core::VideoRenderPlan,
        req: &GenerationRequest,
    ) -> gen_core::Result<()> {
        plan.check_provider(self.descriptor(), self.video_load_binding())?;
        self.validate(req)?;
        let layout = self.component_layout()?;
        gen_core_family::mochi::check_layout(plan, &video_layout(req, &layout.vae_cfg)?)
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// Reject a request Mochi cannot serve: the shared capability floor plus the model-specific
    /// constraints — non-empty prompt, 16-divisible width/height, and `num_frames = 1 + 6·k`.
    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        gen_core_family::mochi::validate_request(&self.descriptor.capabilities, req)
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        self.generate_with_scoped_execution_report(req, None, on_progress)
    }

    fn generate_with_scoped_execution_report(
        &self,
        req: &GenerationRequest,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_generation(self.descriptor(), req, on_progress, |progress| {
            self.generate_scoped(req, scope, progress)
                .map(gen_core::GenerationReport::from_output)
        })
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_scoped(req, None, on_progress)
    }
}

/// Construct a lazy candle Mochi 1 generator. `spec.weights` is a `genmo/mochi-1-preview` snapshot dir
/// (a split-weight diffusers layout). Adapters / on-the-fly quantization / conditioning are rejected
/// (not wired — Mochi ships pre-quantized per-tier checkpoints).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = load_root(spec)?;
    let device = candle_gen::default_device()?;
    Ok(Box::new(MochiGenerator {
        load_binding: RenderLoadBinding::from_video_spec(spec),
        descriptor: descriptor(),
        pipe: Pipeline::new(&root, &device),
        components: Mutex::new(None),
    }))
}

fn load_root(spec: &LoadSpec) -> gen_core::Result<PathBuf> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(gen_core::Error::Msg(
                "mochi_1: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle mochi does not support LoRA/LoKr".into(),
        ));
    }
    // Mochi ships **pre-quantized per-tier** dirs (epic 1788 / A6): the tier dir IS the quant selection
    // (point WeightsSource at q4/q8/bf16), NOT on-the-fly requant. A stray `spec.quantize` is honoured
    // only when it AGREES with the tier the dir already is (assert-against-manifest, LTX-style); a raw
    // snapshot + `spec.quantize` stays rejected (there is no on-the-fly requant path).
    if let Some(q) = spec.quantize {
        match tier::MochiTierPaths::detect(&root) {
            Some(t) => {
                // A quantized tier at a non-64 group would mis-align the MLX→GGML repack — fail loudly.
                t.validate_group_size()?;
                let want_bits = match q {
                    Quant::Q4 => Some(4),
                    Quant::Q8 => Some(8),
                    other => {
                        return Err(gen_core::Error::Unsupported(format!(
                        "mochi_1: spec.quantize={other:?} has no Mochi tier (tiers are q4/q8/bf16)"
                    )))
                    }
                };
                let have_bits = t.manifest_bits()?;
                if want_bits != have_bits {
                    return Err(gen_core::Error::Unsupported(format!(
                        "mochi_1: spec.quantize={q:?} (bits {want_bits:?}) disagrees with the tier at {} \
                         (manifest bits {have_bits:?}). The tier dir IS the quant selection — point \
                         WeightsSource at the matching q4/q8/bf16 dir instead of requesting a requant.",
                        root.display()
                    )));
                }
            }
            None => {
                return Err(gen_core::Error::Unsupported(format!(
                    "mochi_1: spec.quantize={q:?} unsupported on a raw snapshot — Mochi ships \
                     pre-quantized per-tier checkpoints; point WeightsSource at the q4/q8/bf16 tier dir \
                     (no on-the-fly requant)"
                )));
            }
        }
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle mochi does not support image / I2V conditioning yet (txt2video only)".into(),
        ));
    }
    Ok(root)
}

fn compile_video_plan(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<VideoPlanLayout>> {
    gen_core_family::mochi::validate_request(&descriptor().capabilities, req)?;
    let root = load_root(spec)?;
    let cfg = Pipeline::read_layout(&root)?.vae_cfg;
    video_layout(req, &cfg).map(Some)
}

fn video_layout(
    req: &GenerationRequest,
    cfg: &MochiVaeConfig,
) -> gen_core::Result<VideoPlanLayout> {
    let video = gen_core_family::mochi::MochiVideo::new(
        req,
        &cfg.temporal_expansions,
        &cfg.spatial_expansions,
        req.seed.unwrap_or_default(),
    )?;
    video.layout(
        DEFAULT_DECODE_CHUNK_FRAMES,
        gen_core_family::mochi::MochiDecoder::Candle,
        candle_gen::default_memory_budget_domain(),
    )
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

/// Add the Candle Mochi 1 generator to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(REGISTRATION)
        .register_video_plan(gen_core::VideoPlanRegistration {
            provider_id: MODEL_ID,
            compile: compile_video_plan,
        })
}

/// Build the complete explicit Candle Mochi 1 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    use super::*;

    #[test]
    fn registered_video_plan_reads_shared_codec_metadata_before_weights() {
        use candle_gen::gen_core::{GenerationRequest, LoadSpec, WeightsSource};
        let root = tempfile::tempdir().unwrap();
        let tier = root.path().join("bf16");
        std::fs::create_dir_all(tier.join("transformer")).unwrap();
        std::fs::write(tier.join("split_model.json"), r#"{"quantized":false}"#).unwrap();
        std::fs::create_dir(root.path().join("text_encoder")).unwrap();
        std::fs::create_dir(root.path().join("vae")).unwrap();
        // Non-default metadata detects accidental hard-coding of the real 6x8 checkpoint.
        std::fs::write(
            root.path().join("vae/config.json"),
            r#"{"temporal_expansions":[1,1,3],"spatial_expansions":[1,2,2]}"#,
        )
        .unwrap();
        let req = GenerationRequest {
            prompt: "kitten".into(),
            width: 64,
            height: 48,
            frames: Some(7),
            ..Default::default()
        };
        let spec = LoadSpec::new(WeightsSource::Dir(tier));
        let registry = super::provider_registry().unwrap();
        let plan = registry
            .compile_video_plan(super::MODEL_ID, &spec, &req)
            .unwrap()
            .unwrap();
        let phase = &plan.layout().phases[0];
        assert_eq!(
            plan.minimum_denoise_live_bytes(plan.layout().domain)
                .unwrap(),
            16_891_904
        );
        assert_eq!(
            (phase.latent_frames, phase.latent_bytes),
            (3, 3 * 12 * 16 * 12 * 4)
        );
        assert_eq!((phase.decoded_frames, phase.decode_chunk_frames), (9, 3));
        assert_eq!(plan.layout().output.frames, 7);
        // One frame takes the non-chunked path. Its view retains three raw f32 frames
        // from this fixture's 3x codec. Conversion also holds the u8 tensor, a flat host
        // vector, and the returned RGB copy. CUDA charges the two host vectors separately.
        let single_req = GenerationRequest {
            frames: Some(1),
            ..req.clone()
        };
        let single = registry
            .compile_video_plan(super::MODEL_ID, &spec, &single_req)
            .unwrap()
            .unwrap();
        let rgb_frame = 64 * 48 * 3;
        let domain = candle_gen::default_memory_budget_domain();
        let host = gen_core::MemoryBudgetDomain::HostRam;
        if domain == gen_core::MemoryBudgetDomain::DeviceVram {
            assert_eq!(
                single.minimum_decode_live_bytes(host).unwrap(),
                2 * rgb_frame
            );
        }
        // Three raw f32 frames plus 37 bytes per cropped RGB sample at eager rounding.
        assert_eq!(
            single.minimum_decode_live_bytes(domain).unwrap(),
            49 * rgb_frame
        );
        let invalid = GenerationRequest {
            frames: Some(8),
            ..req
        };
        assert!(registry
            .compile_video_plan(super::MODEL_ID, &spec, &invalid)
            .is_err());
    }

    #[test]
    fn failed_cold_load_accepts_recompiled_plan_after_metadata_repair() {
        let root = tempfile::tempdir().unwrap();
        let vae = root.path().join("vae");
        std::fs::create_dir(&vae).unwrap();
        let config = vae.join("config.json");
        std::fs::write(&config, "{}").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let req = GenerationRequest {
            prompt: "kitten".into(),
            width: 64,
            height: 48,
            frames: Some(7),
            ..Default::default()
        };
        let registry = provider_registry().unwrap();
        let original = registry
            .compile_video_plan(MODEL_ID, &spec, &req)
            .unwrap()
            .unwrap();
        let generator = MochiGenerator {
            load_binding: RenderLoadBinding::from_video_spec(&spec),
            descriptor: descriptor(),
            pipe: Pipeline::new(root.path(), &candle_gen::candle_core::Device::Cpu),
            components: Mutex::new(None),
        };
        generator.validate_video_plan(&original, &req).unwrap();
        let err = generator
            .generate(&req, &mut |_| panic!("progress before weight load"))
            .unwrap_err();
        assert!(err.to_string().contains("text_encoder"), "{err}");
        assert!(candle_gen::lock_recover(&generator.components).is_none());
        std::fs::write(&config, r#"{"spatial_expansions":[1,2,2]}"#).unwrap();
        let changed = registry
            .compile_video_plan(MODEL_ID, &spec, &req)
            .unwrap()
            .unwrap();
        assert_eq!(original.geometry(), changed.geometry());
        assert_eq!(
            original.layout().phases[0].latent_frames,
            changed.layout().phases[0].latent_frames
        );
        assert_ne!(
            original.layout().phases[0].latent_bytes,
            changed.layout().phases[0].latent_bytes
        );
        assert!(
            matches!(generator.validate_video_plan(&original, &req), Err(gen_core::Error::Unsupported(msg)) if msg.contains("Mochi codec geometry"))
        );
        generator.validate_video_plan(&changed, &req).unwrap();
        assert!(candle_gen::lock_recover(&generator.components).is_none());
    }

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(generators, ["mochi_1"]);
    }

    #[test]
    fn registered_descriptor_conforms() {
        let registry = provider_registry().unwrap();
        assert!(
            registry.descriptor_conformance_errors().is_empty(),
            "mochi descriptor conformance violations: {:?}",
            registry.descriptor_conformance_errors()
        );
    }

    #[test]
    fn descriptor_identity_and_capabilities() {
        let d = descriptor();
        assert_eq!(d.id, "mochi_1");
        assert_eq!(d.family, "mochi");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        let c = &d.capabilities;
        assert!(c.supports_negative_prompt);
        assert!(c.supports_guidance);
        assert!(c.supports_true_cfg);
        assert!(!c.mac_only);
        assert_eq!(c.max_count, 1);
        assert!(c.conditioning.is_empty(), "t2v-only: no conditioning kinds");
        assert!(c.supported_quants.is_empty(), "no on-the-fly requant");
    }

    #[test]
    fn stride_is_the_pinned_size_multiple() {
        // sc-12587: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties `requiresDimensionsMultipleOf`
        // to. Pin the value and prove a multiple of 8 that is not a multiple of SIZE_MULTIPLE (16) is
        // rejected with the stride error — matching mlx's `mlx_gen_mochi::SIZE_MULTIPLE`.
        assert_eq!(SIZE_MULTIPLE, 16);
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = provider_registry().unwrap().load(MODEL_ID, &spec).unwrap();
        let base = GenerationRequest {
            prompt: "a calico kitten".into(),
            width: 64,
            height: 64,
            frames: Some(7),
            ..Default::default()
        };
        assert!(g.validate(&base).is_ok());
        let off = GenerationRequest {
            width: 72, // 9×8 — a multiple of 8 but not SIZE_MULTIPLE
            ..base.clone()
        };
        let err = g.validate(&off).unwrap_err().to_string();
        assert!(err.contains("divisible by 16"), "got: {err}");
    }
}
