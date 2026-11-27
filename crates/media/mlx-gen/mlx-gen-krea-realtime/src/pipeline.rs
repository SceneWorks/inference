//! Krea Realtime 14B provider: capability surface, registration, snapshot resolution, and the
//! [`Generator`] entrypoint (sc-8439, S6).
//!
//! [`Generator::generate`] maps a [`GenerationRequest`] onto a [`KreaRealtimeJob`] and runs the live
//! pipeline (prompt → UMT5 → AR few-step latents → z16 Wan VAE decode → clip). Krea Realtime is an
//! **autoregressive, self-forcing, CFG-off** model: no negative prompt, no guidance scale, a fixed
//! Self-Forcing few-step schedule. It advertises **i2v** ([`ConditioningKind::Reference`] — a still
//! warms the AR KV cache, [`crate::t2v::generate_i2v`]) and **v2v** ([`ConditioningKind::VideoClip`] — a
//! source clip drives the strength-controlled AR init, [`crate::t2v::generate_v2v`]) conditioning
//! (sc-8440 S7); `run` routes a request's conditioning to the matching pipeline. The streaming realtime
//! decode / webcam v2v deque is the streaming epic (8432).
//!
//! Registration mirrors the sibling Wan-2.1-14B video provider `mlx-gen-scail2`: an explicit
//! `ModelRegistration` composed by the platform catalog (never linker-discovered), so
//! `provider_registry().load("krea_realtime_14b", spec)` yields this generator.

use std::path::PathBuf;
use std::sync::Mutex;

use mlx_gen::gen_core::{
    AdapterInstallPolicy, AdapterSurface, ConditioningSurface, FamilyDriverSpec,
    LatentCodecDeclaration, ModelBinding, OutputSurface, Pass0ConditioningRequirement,
    PreviewSupport, ProductLimits, ResidencyPhaseKind, ResidencyPhaseNode, ResidencySurface,
    RouteBindings, RouteDefaults, RouteRecord, RouteStructure, SamplingSurface,
    SeedAndBatchSemantics, SeedScheme, StepSupport, TextEncoderDeclaration, TextEncoderSurface,
    WeightPlanDeclaration, WeightSourceKind, WAN_Z16_VIDEO_LATENT_SPACE,
};
use mlx_gen::{
    default_seed, AdapterApplyReport, AdapterSpec, CancelFlag, Conditioning, ConditioningKind,
    Error, GenerationOutput, GenerationRequest, Generator, Image, Modality, ModelDescriptor,
    Progress, Quant, Result, SizeFloor, WeightsSource,
};

use crate::config::{KreaRealtimeConfig, MODEL_ID};
use crate::plan::{KreaRealtimeRenderPlan, KreaRealtimeRoute};
use crate::t2v::{
    emit_whole_clip_to_sink, generate_i2v_reported, generate_t2v_reported, generate_v2v_reported,
    KreaRealtimeJob,
};

/// The Self-Forcing few-step sampler name Krea Realtime advertises (a fixed short per-block flow-match
/// renoise schedule, not a selectable classic solver). Distinct from Wan's UniPC/DPM++ so a consumer
/// knows this model runs the AR few-step regime.
pub const SELF_FORCING_SAMPLER: &str = "self_forcing";

/// Primary typed route declaration for Krea Realtime 14B (`REG-1`, `AP5`).
pub fn route() -> RouteRecord {
    RouteRecord {
        route_id: MODEL_ID,
        family: "krea_realtime",
        backend: "mlx",
        modality: Modality::Video,
        mac_only: true,
        structure: RouteStructure {
            text_encoders: TextEncoderSurface {
                encoders: vec![TextEncoderDeclaration::new("umt5_xxl", Some(512))],
                ..Default::default()
            },
            conditioning: ConditioningSurface {
                kinds: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
                control_kinds: None,
                pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: false,
                modules: vec!["causal_kv_cache"],
                ..Default::default()
            },
            sampling: SamplingSurface {
                samplers: vec![SELF_FORCING_SAMPLER],
                schedulers: vec![],
                guidance_methods: vec![],
                supports_guidance: false,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: ModelBinding::family_driver(FamilyDriverSpec::new(
                crate::generate::krea_realtime_driver_support(),
            )),
            latent_codec: LatentCodecDeclaration {
                codec_id: Some("wan_z16_vae"),
                output_latent_space: Some(&WAN_Z16_VIDEO_LATENT_SPACE),
                pixel_stride: Some(16),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            adapters: AdapterSurface::lora_and_lokr(AdapterInstallPolicy::ForwardResidual),
            residency: ResidencySurface {
                lifetime_graph: vec![
                    ResidencyPhaseNode::new(ResidencyPhaseKind::PromptEncode, ["umt5_xxl"], true),
                    ResidencyPhaseNode::new(
                        ResidencyPhaseKind::Denoise,
                        ["wan2_1_causal_dit"],
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
                ..Default::default()
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
            size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 16 },
            supported_steps: StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: SeedAndBatchSemantics {
            seed_scheme: SeedScheme::BackendGlobalSeeded,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: RouteDefaults {
            steps: Some(4),
            guidance: None,
            sampler: Some(SELF_FORCING_SAMPLER),
            schedule: None,
            shift: Some(5.0),
            strength: None,
            dimensions: Some((832, 480)),
            ..Default::default()
        },
        bindings: RouteBindings::generator(descriptor, |spec| load(spec).map_err(Into::into))
            .with_memory_registration(crate::memory_strategy::REGISTRATION)
            .with_resident_only_memory_registration(crate::memory_strategy::RESIDENT_ONLY_WITNESS),
    }
}

/// Stable identity + advertised capabilities for Krea Realtime 14B, computed from [`route`].
pub fn descriptor() -> ModelDescriptor {
    route().compute_descriptor()
}

/// The loaded Krea Realtime 14B model: the resolved config + the snapshot dir. The heavy components
/// (DiT / UMT5 / z16 VAE) are staged inside [`crate::t2v::generate_t2v`].
pub struct KreaRealtime {
    descriptor: ModelDescriptor,
    config: KreaRealtimeConfig,
    root: PathBuf,
    /// Inference LoRA(s) from [`LoadSpec::adapters`](mlx_gen::LoadSpec::adapters) (sc-15015, S14) —
    /// installed onto the DiT as forward-time residuals in [`crate::t2v::load_transformer`], the
    /// `apply_adapters_strict_with_diff_patch` path. Low-rank residuals stack over dense bf16 and
    /// packed Q4/Q8 alike; supported lightx2v diff-patch deltas land through tier-stable dense bias and
    /// norm parameters (sc-15326).
    adapters: Vec<AdapterSpec>,
    /// The requested load-time quantization from [`LoadSpec::quantize`](mlx_gen::LoadSpec::quantize)
    /// (sc-15203, S19). Reconciled in [`crate::t2v::load_transformer`] against the tier the snapshot
    /// actually ships at: a **dense bf16** snapshot is quantized in memory after load; a
    /// **pre-quantized** snapshot is already packed, so the request is a no-op at the same width and a
    /// hard error at a different one ("stored wins", loudly — `quantize` no-ops over packed weights, so
    /// a silent mismatch would serve a tier the caller did not ask for).
    quant: Option<Quant>,
    /// SC-20770 adopts the shared selector truthfully as resident-only. Existing staging,
    /// automatic decode tiling, and the fixed causal window are not request-selected rungs.
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    /// Exact prepared spec whose direct files produced `memory_strategy`. Retained so the loaded
    /// generator repeats the provider-owned receipt check at the request safety boundary.
    loaded_spec: mlx_gen::LoadSpec,
    /// Content- and tensor-geometry-bound receipt sealed before construction.
    loaded_artifact_identity: String,
    /// Actual engine-owned adapter outcomes from the most recent successful generation. The loaded
    /// provider is cached and `Generator::generate` takes `&self`, so the compatibility-safe report
    /// accessor uses interior mutability rather than changing generation's return type.
    adapter_reports: Mutex<Vec<AdapterApplyReport>>,
}

/// The three advertised request routes, selected once by [`KreaRealtime::run`] and executed through
/// one report-publishing funnel.
enum GenerationRoute<'a> {
    T2v,
    I2v(&'a Image),
    V2v { frames: &'a [Image], strength: f32 },
}

/// A reported generation backend. Production delegates to the real T2V/I2V/V2V entrypoints; tests
/// can drive the mandatory route/finalize contract with actual tiny-host adapter output and no model
/// snapshot.
trait ReportedGeneration {
    fn t2v(&mut self) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)>;
    fn i2v(&mut self, image: &Image) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)>;
    fn v2v(
        &mut self,
        frames: &[Image],
        strength: f32,
    ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)>;
}

struct ProductReportedGeneration<'a> {
    root: &'a std::path::Path,
    config: &'a KreaRealtimeConfig,
    job: &'a KreaRealtimeJob<'a>,
    output_latent: usize,
    generation_latent: usize,
    adapters: &'a [AdapterSpec],
    quant: Option<Quant>,
    cancel: &'a CancelFlag,
    on_progress: &'a mut dyn FnMut(Progress),
}

impl ReportedGeneration for ProductReportedGeneration<'_> {
    fn t2v(&mut self) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
        generate_t2v_reported(
            self.root,
            self.config,
            self.job,
            self.output_latent,
            self.adapters,
            self.quant,
            self.cancel,
            self.on_progress,
        )
    }

    fn i2v(&mut self, image: &Image) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
        generate_i2v_reported(
            self.root,
            self.config,
            self.job,
            self.output_latent,
            image,
            self.adapters,
            self.quant,
            self.cancel,
            self.on_progress,
        )
    }

    fn v2v(
        &mut self,
        frames: &[Image],
        strength: f32,
    ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
        generate_v2v_reported(
            self.root,
            self.config,
            self.job,
            frames,
            self.generation_latent,
            strength,
            self.adapters,
            self.quant,
            self.cancel,
            self.on_progress,
        )
    }
}

/// Load Krea Realtime from a converted MLX snapshot directory (`dit.safetensors` + the stock Wan
/// `t5_encoder.safetensors` / `vae.safetensors` / `tokenizer.json` — Krea Realtime ships
/// transformer-only, reusing stock Wan for the TE / VAE / tokenizer).
pub fn load(spec: &mlx_gen::LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "krea_realtime: expected a model directory (converted MLX snapshot), not a single file"
                .into(),
        )),
    };
    if !root.exists() {
        return Err(Error::Msg(format!(
            "krea_realtime: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(spec)?;
    let loaded_artifact_identity = crate::memory_strategy::canonical_artifact_identity(spec)?;
    let config = KreaRealtimeConfig::from_model_dir(&root)?;
    Ok(Box::new(KreaRealtime {
        descriptor: descriptor(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant: spec.quantize,
        memory_strategy,
        loaded_spec: spec.clone(),
        loaded_artifact_identity,
        adapter_reports: Mutex::new(Vec::new()),
    }))
}

// The registration constant the platform catalog composes (explicit, not linker-discovered); bridges
// the crate `Result` into backend-neutral `gen_core::Result`.


impl KreaRealtime {
    fn generate_execution_attempt(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.run(req, on_progress).map_err(Into::into)
    }
}

impl Generator for KreaRealtime {
    fn generate_with_report(
        &self,
        req: &mlx_gen::GenerationRequest,
        on_progress: &mut dyn FnMut(mlx_gen::Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::GenerationReport> {
        let output = mlx_gen::video_execution::without_capture(|| {
            self.generate_execution_attempt(req, on_progress)
        })?;
        let mut report = mlx_gen::gen_core::GenerationReport::from_output(output);
        report.adapter_apply_reports = self.current_adapter_apply_reports();
        Ok(report)
    }

    fn generate_with_execution_report(
        &self,
        req: &mlx_gen::GenerationRequest,
        on_progress: &mut dyn FnMut(mlx_gen::Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::generator::CompletedGeneration> {
        mlx_gen::video_execution::complete(self.descriptor(), &req.cancel, || {
            let output = self.generate_execution_attempt(req, &mut |event| {
                mlx_gen::video_execution::without_capture(|| on_progress(event));
            })?;
            let mut report = mlx_gen::gen_core::GenerationReport::from_output(output);
            report.adapter_apply_reports = self.current_adapter_apply_reports();
            Ok(report)
        })
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::loaded_safety_check(
            &self.loaded_spec,
            &self.memory_strategy,
            &self.loaded_artifact_identity,
            context,
        )
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_and_compile_plan(req)
            .map(|_| ())
            .map_err(Into::into)
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
}

impl KreaRealtime {
    pub(crate) fn current_adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        self.adapter_reports
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn validate_and_compile_plan(&self, req: &GenerationRequest) -> Result<KreaRealtimeRenderPlan> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        // sc-20265 & PIPE-16: compile and validate the complete pre-load render/residency plan
        // before opening or allocating any weights. Both `validate` and `run` route through here.
        KreaRealtimeRenderPlan::from_generation_request(&self.config, req)
    }

    /// Validate `req` and compile its pre-load [`KreaRealtimeRenderPlan`] without touching model
    /// weights (`PIPE-16`, `§6.9`).
    pub fn render_plan(&self, req: &GenerationRequest) -> Result<KreaRealtimeRenderPlan> {
        self.validate_and_compile_plan(req)
    }

    #[cfg(test)]
    fn validate_and_resolve_frames(
        &self,
        req: &GenerationRequest,
    ) -> Result<crate::plan::ResolvedFrames> {
        let plan = self.validate_and_compile_plan(req)?;
        Ok(crate::plan::ResolvedFrames {
            fps: plan.fps,
            output: plan.output_frames,
            output_latent: plan.output_latent_frames,
            generation_latent: plan.generation_latent_frames,
        })
    }

    fn finish_reported_generation(
        &self,
        result: Result<(GenerationOutput, Vec<AdapterApplyReport>)>,
    ) -> Result<GenerationOutput> {
        let (output, reports) = result?;
        *self
            .adapter_reports
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = reports;
        Ok(output)
    }

    fn finish_routed_generation_to_sink(
        &self,
        route: GenerationRoute<'_>,
        generation: &mut impl ReportedGeneration,
        sink: &mut dyn mlx_gen::gen_core::ChunkSink<GenerationOutput>,
        causal_codec_policy: Option<mlx_gen::gen_core::CausalCodecStatePolicy>,
    ) -> mlx_gen::gen_core::Result<()> {
        let reported = match route {
            GenerationRoute::T2v => generation.t2v(),
            GenerationRoute::I2v(image) => generation.i2v(image),
            GenerationRoute::V2v { frames, strength } => generation.v2v(frames, strength),
        };
        let output = self
            .finish_reported_generation(reported)
            .map_err(mlx_gen::gen_core::Error::from)?;
        emit_whole_clip_to_sink(output, causal_codec_policy, sink)
    }

    /// Execute every advertised modality through the same report-publishing finish seam and deliver
    /// the whole clip as one terminal chunk (`sequence = 0`, `is_final = true`) without cloning
    /// decoded frames. Keeping the route match inside this method makes it impossible for a
    /// T2V/I2V/V2V call site to consume a reported generation while forgetting to expose its adapter
    /// outcome.
    #[cfg(test)]
    fn finish_routed_generation(
        &self,
        route: GenerationRoute<'_>,
        generation: &mut impl ReportedGeneration,
    ) -> Result<GenerationOutput> {
        let mut collector = mlx_gen::gen_core::TerminalChunkCollector::new(CancelFlag::default());
        self.finish_routed_generation_to_sink(route, generation, &mut collector, None)?;
        collector.into_output().map_err(Into::into)
    }

    /// Map the text-to-video request onto a [`KreaRealtimeJob`] and run the AR pipeline. Self-validates
    /// the shared capability floor first (`impl_generator!`'s `generate` does not call `validate`), so a
    /// direct `Generator::generate` still rejects an out-of-surface request (count / size / sampler /
    /// finiteness) before any weight load.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Cached providers survive across jobs. Clear the previous run before validation/load so a
        // failed generation can never expose stale adapter evidence to its caller.
        self.adapter_reports
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let plan = self.validate_and_compile_plan(req)?;

        let job = KreaRealtimeJob {
            prompt: &req.prompt,
            width: plan.width,
            height: plan.height,
            num_frames: plan.output_frames,
            fps: plan.fps,
            seed: req.seed.unwrap_or_else(default_seed),
            steps: req.steps.map(|s| s as usize),
            memory: req.memory.unwrap_or_default(),
        };

        // Route on the pre-load `plan.route` (sc-8440 S7): a `VideoClip` source -> v2v; else a
        // `Reference` still -> i2v; else text-to-video.
        let route = match plan.route {
            KreaRealtimeRoute::V2v { strength, .. } => {
                let frames = req
                    .conditioning
                    .iter()
                    .find_map(|c| match c {
                        Conditioning::VideoClip { frames, .. } => Some(frames.as_slice()),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                            "{MODEL_ID} video_to_video requires a VideoClip carrier"
                        ))
                    })?;
                GenerationRoute::V2v { frames, strength }
            }
            KreaRealtimeRoute::I2v { .. } => {
                let image = req
                    .conditioning
                    .iter()
                    .find_map(|c| match c {
                        Conditioning::Reference { image, .. } => Some(image),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                            "{MODEL_ID} image_to_video requires a Reference carrier"
                        ))
                    })?;
                GenerationRoute::I2v(image)
            }
            KreaRealtimeRoute::T2v => GenerationRoute::T2v,
        };
        let mut generation = ProductReportedGeneration {
            root: &self.root,
            config: &self.config,
            job: &job,
            output_latent: plan.output_latent_frames,
            generation_latent: match plan.route {
                KreaRealtimeRoute::V2v {
                    source_latent_frames,
                    ..
                } => source_latent_frames,
                _ => plan.generation_latent_frames,
            },
            adapters: &self.adapters,
            quant: self.quant,
            cancel: &req.cancel,
            on_progress,
        };
        let mut collector = mlx_gen::gen_core::TerminalChunkCollector::new(req.cancel.clone());
        self.finish_routed_generation_to_sink(
            route,
            &mut generation,
            &mut collector,
            Some(plan.causal_codec_state),
        )?;
        collector.into_output().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{resolve_frames, ResolvedFrames};
    use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, DiffPatchPart};
    use mlx_gen::{AdapterKind, GenerationOutput};
    use mlx_rs::Array;

    /// A weights-free [`KreaRealtime`] whose [`KreaRealtime::run`] hits the shared floor before any
    /// load. Sound only because validation runs first: a *passing* request would then try to load
    /// `root` and fail.
    fn unloaded() -> KreaRealtime {
        KreaRealtime {
            descriptor: descriptor(),
            config: KreaRealtimeConfig::default(),
            root: PathBuf::from("/nonexistent-krea-realtime-snapshot"),
            adapters: Vec::new(),
            quant: None,
            memory_strategy: mlx_gen::gen_core::MemoryProviderContract::compatibility_default(
                MODEL_ID,
                mlx_gen::gen_core::MemoryBackendRealization::MlxMetal {
                    bounded_wired_residency: false,
                    lazy_or_mmap_materialization: true,
                    explicit_evaluation_and_synchronization: false,
                    cache_eviction: false,
                },
            ),
            loaded_spec: mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(PathBuf::from(
                "/nonexistent-krea-realtime-snapshot",
            ))),
            loaded_artifact_identity: String::new(),
            adapter_reports: Mutex::new(Vec::new()),
        }
    }

    /// sc-22738: the job the provider hands to the t2v/i2v/v2v routes carries the **request-scoped**
    /// memory selection. Every calibration fault in this crate reads `job.memory` (the Conditioning
    /// hook inside `stage_components`) or `params.memory` (derived from it), so a `run` that dropped
    /// the request scope — `Default::default()` instead of `req.memory.unwrap_or_default()` — would
    /// silently make every hook unreachable while leaving the phase-ordering tests green. Pinned as
    /// source text because `run` cannot be driven without the 28 GB snapshot.
    #[test]
    fn the_job_carries_the_request_scoped_memory_selection() {
        let source = include_str!("pipeline.rs");
        let region = source
            .split_once("let job = KreaRealtimeJob {")
            .expect("job construction")
            .1
            .split_once("};")
            .expect("job construction end")
            .0;
        assert!(
            region.contains("memory: req.memory.unwrap_or_default(),"),
            "KreaRealtimeJob must take its memory from the request, not a default: {region}"
        );
        let production = source
            .split_once("\n#[cfg(test)]\n")
            .map(|(head, _)| head)
            .unwrap_or(source);
        assert_eq!(
            production.matches("KreaRealtimeJob {").count(),
            1,
            "one production job construction — add the assertion above to any new one"
        );
    }

    #[test]
    fn generator_contract_returns_the_provider_owned_adapter_reports() {
        let provider = unloaded();
        let expected = AdapterApplyReport {
            adapter_path: PathBuf::from("/models/out_of_surface.safetensors"),
            applied: 1,
            skipped: vec!["blocks.0.cross_attn.norm_k_img".to_owned()],
        };
        provider
            .adapter_reports
            .lock()
            .unwrap()
            .push(expected.clone());
        assert_eq!(
            provider.current_adapter_apply_reports(),
            vec![expected],
            "severing the Krea override must not fall back to the empty compatibility default"
        );
    }

    struct TinyDiffPatchHost {
        norm: Array,
    }

    impl AdaptableHost for TinyDiffPatchHost {
        fn adaptable_mut(&mut self, _path: &[&str]) -> Option<&mut AdaptableLinear> {
            None
        }

        fn diff_patch_param_mut(
            &mut self,
            path: &[&str],
            part: DiffPatchPart,
        ) -> Option<&mut Array> {
            (path == ["norm"] && part == DiffPatchPart::Weight).then_some(&mut self.norm)
        }
    }

    fn actual_tiny_adapter_report(tmp: &tempfile::TempDir) -> AdapterApplyReport {
        let dir = tmp.path().join("mlx_gen_krea_pipeline_report_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("actual_report_{}.safetensors", std::process::id()));
        let delta = Array::from_slice(&[0.25f32, -0.5], &[2]);
        Array::save_safetensors(vec![("diffusion_model.norm.diff", &delta)], None, &path).unwrap();
        let mut host = TinyDiffPatchHost {
            norm: Array::from_slice(&[1.0f32, 1.0], &[2]),
        };
        let reports = mlx_gen::adapters::loader::apply_adapters_strict_with_diff_patch_reported(
            &mut host,
            &[AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora)],
            MODEL_ID,
        )
        .expect("the material diff-patch tensor applies");
        assert_eq!(reports.len(), 1);
        let report = reports.into_iter().next().unwrap();
        assert_eq!(report.applied, 1);
        assert!(report.diff_patch_unapplied.is_empty());
        AdapterApplyReport {
            adapter_path: path,
            applied: report.applied,
            skipped: report.diff_patch_unapplied,
        }
    }

    struct TinyReportedGeneration {
        report: AdapterApplyReport,
        called: Option<&'static str>,
    }

    impl TinyReportedGeneration {
        fn result(&self) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
            Ok((
                GenerationOutput::Video {
                    frames: Vec::new(),
                    fps: 1,
                    audio: None,
                },
                vec![self.report.clone()],
            ))
        }
    }

    impl ReportedGeneration for TinyReportedGeneration {
        fn t2v(&mut self) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
            self.called = Some("t2v");
            self.result()
        }

        fn i2v(&mut self, _image: &Image) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
            self.called = Some("i2v");
            self.result()
        }

        fn v2v(
            &mut self,
            _frames: &[Image],
            _strength: f32,
        ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
            self.called = Some("v2v");
            self.result()
        }
    }

    /// A material adapter is applied by the real loader, then its actual report crosses the single
    /// mandatory route→finish→Generator contract for every advertised generation mode. This is
    /// mutation-discriminating: returning a reported result from any route without
    /// `finish_reported_generation` leaves the accessor empty and fails.
    #[test]
    fn every_generation_route_publishes_actual_adapter_application_output() {
        let tmp = tempfile::tempdir().unwrap();
        let actual = actual_tiny_adapter_report(&tmp);
        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![1, 2, 3],
        };
        let frames = vec![image.clone()];
        for (route, expected_route) in [
            (GenerationRoute::T2v, "t2v"),
            (GenerationRoute::I2v(&image), "i2v"),
            (
                GenerationRoute::V2v {
                    frames: &frames,
                    strength: 0.5,
                },
                "v2v",
            ),
        ] {
            let provider = unloaded();
            let mut generation = TinyReportedGeneration {
                report: actual.clone(),
                called: None,
            };
            provider
                .finish_routed_generation(route, &mut generation)
                .expect("reported generation succeeds");
            assert_eq!(generation.called, Some(expected_route));
            assert_eq!(
                provider.current_adapter_apply_reports(),
                vec![actual.clone()],
                "{expected_route} must publish the actual loader report through the Generator contract"
            );
        }
    }

    /// Direct provider surface remains CFG-off T2V/I2V/V2V; SC-20770 narrows only the worker route.
    #[test]
    fn descriptor_is_cfg_off_video_with_i2v_v2v() {
        let d = descriptor();
        assert_eq!(d.id, "krea_realtime_14b");
        assert_eq!(d.family, "krea_realtime");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Video);
        let c = &d.capabilities;
        assert!(!c.supports_guidance, "Krea Realtime is CFG-off");
        assert!(!c.supports_negative_prompt, "CFG-off ⇒ no negative prompt");
        assert!(!c.supports_true_cfg);
        assert!(
            c.accepts(ConditioningKind::Reference),
            "i2v: a reference still is accepted (S7)"
        );
        assert!(c.accepts(ConditioningKind::VideoClip));
        assert_eq!(
            c.conditioning,
            vec![ConditioningKind::Reference, ConditioningKind::VideoClip]
        );
        assert_eq!(c.samplers, vec!["self_forcing"]);
        // S14: Wan-family style-LoRA / LoKr on the dense bf16 DiT is wired (LoadSpec::adapters →
        // the strict diff-patch-aware load path), so both knobs are advertised honestly.
        assert!(c.supports_lora, "S14 wires dense Wan-family style LoRA");
        assert!(c.supports_lokr, "S14 wires the dense LoKr install path too");
        assert!(c.supports_kv_cache, "the AR regime runs a rolling KV cache");
        assert!(!c.supports_sequential_offload);
        assert!(c.unconditionally_engages_staged_residency);
        assert_eq!(
            c.staged_residency_availability(),
            mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged,
        );
        assert!(c.mac_only);
        assert!(!c.supports_streaming, "batch form; streaming is epic 8432");
        // The descriptor passes the weights-free conformance sweep (Video modality admits VideoClip).
        assert!(
            mlx_gen::gen_core::registry::model_descriptor_errors(&d).is_empty(),
            "descriptor must be conformant: {:?}",
            mlx_gen::gen_core::registry::model_descriptor_errors(&d)
        );
    }

    /// Krea's z16 VAE reduces pixels by 8 and the DiT then packs 2x2 latent patches, so an explicit
    /// request must land on a 16px grid. Exercise the published descriptor and the loaded-provider
    /// preflight: neither path may accept 644x484 and silently render 640x480.
    #[test]
    fn explicit_off_grid_size_is_refused() {
        let request = |width, height| GenerationRequest {
            width,
            height,
            count: 1,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0; 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        let off_grid = request(644, 484);

        let advertised = descriptor()
            .capabilities
            .validate_request(MODEL_ID, &off_grid)
            .expect_err("the descriptor must reject an explicit off-grid size");
        assert!(
            matches!(advertised, mlx_gen::gen_core::Error::Msg(_)),
            "an off-grid request is a typed validation error, got: {advertised:?}"
        );
        let advertised_message = advertised.to_string();
        assert!(
            advertised_message.contains("multiples of 16")
                && advertised_message.contains("644×484"),
            "the rejection must name the 16px grid and requested size, got: {advertised_message}"
        );

        let provider = unloaded();
        let provider_error = Generator::validate(&provider, &off_grid)
            .expect_err("the provider preflight must reject an explicit off-grid size");
        assert_eq!(provider_error.to_string(), advertised_message);

        let on_grid = request(640, 480);
        descriptor()
            .capabilities
            .validate_request(MODEL_ID, &on_grid)
            .expect("a representative on-grid size remains advertised");
        Generator::validate(&provider, &on_grid)
            .expect("a representative on-grid size remains accepted by provider preflight");
    }

    /// S19 (sc-15203): the engine advertises the **three tiers** it actually ships — Q4 and Q8 as
    /// `supported_quants` entries, bf16 as their absence. Discriminating rather than a default check:
    /// `Capabilities::default()` is the *empty* slice, so this fails if the field is dropped, and the
    /// order + exact membership are pinned (NVFP4 must NOT appear — it is candle-only, and the load
    /// path rejects it as a typed capability gap rather than routing its `bits() == 4` through the MLX
    /// affine quantizer).
    #[test]
    fn descriptor_advertises_the_three_shipped_tiers() {
        let c = descriptor().capabilities;
        assert_eq!(c.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(
            !c.supported_quants.contains(&Quant::Nvfp4),
            "NVFP4 is a candle/CUDA tier with no MLX affine equivalent"
        );
        // The tiers and the adapter surface are advertised together: a LoRA must work at Q4/Q8 too
        // (the residual install is tier-agnostic — `tests/quant_tiers.rs` proves it on every tier).
        assert!(c.supports_lora && c.supports_lokr);
        // Matches the sibling Wan-2.1-14B video engines (`scail2_14b`, the Wan providers), which is the
        // family this reuses its packed-load path from.
        assert_eq!(
            c.supported_quants,
            mlx_gen_wan::model::descriptor_t2v_14b()
                .capabilities
                .supported_quants,
            "Krea Realtime reuses the Wan packed-load path, so it advertises the same tier surface \
             as the Wan-2.1/2.2 14B T2V sibling it shares that path with"
        );
    }

    /// The shared floor rejects an unadvertised sampler before any weight load.
    #[test]
    fn floor_rejects_unadvertised_sampler() {
        let m = unloaded();
        let mut noop = |_: Progress| {};
        let bad = GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            sampler: Some("unipc".into()),
            ..Default::default()
        };
        let err = m
            .run(&bad, &mut noop)
            .expect_err("unadvertised sampler must be rejected");
        assert!(err.to_string().contains("sampler"), "got: {err}");
    }

    /// A guidance scale is rejected (CFG-off): the floor gates `supports_guidance`.
    #[test]
    fn floor_rejects_guidance_on_cfg_off_model() {
        let m = unloaded();
        let mut noop = |_: Progress| {};
        let bad = GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            guidance: Some(5.0),
            ..Default::default()
        };
        let err = m
            .run(&bad, &mut noop)
            .expect_err("guidance must be rejected on a CFG-off model");
        assert!(err.to_string().contains("guidance"), "got: {err}");
    }

    #[test]
    fn frame_cap_boundary_is_shared_and_rejected_before_staging() {
        let provider = unloaded();
        let request = |frames| GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            frames: Some(frames),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0; 3],
                },
                strength: None,
            }],
            ..Default::default()
        };

        Generator::validate(&provider, &request(1_028))
            .expect("1,028 output frames resolve to the maximum 257 latent frames");
        let i2v_boundary = request(1_028);
        Generator::validate(&provider, &i2v_boundary)
            .expect("I2V keeps the same 1,028-output-frame boundary as T2V");
        let validation_error = Generator::validate(&provider, &request(1_029))
            .expect_err("1,029 output frames resolve to 258 latent frames and must be rejected");
        assert!(
            matches!(validation_error, mlx_gen::gen_core::Error::Unsupported(_)),
            "the model-local cap is a typed capability refusal, got: {validation_error:?}"
        );

        let mut progress_calls = 0;
        let run_error = provider
            .run(&request(1_029), &mut |_| progress_calls += 1)
            .expect_err("run must apply the same cap before touching the nonexistent snapshot");
        assert!(
            matches!(run_error, Error::Unsupported(_)),
            "got: {run_error:?}"
        );
        assert_eq!(
            progress_calls, 0,
            "component staging emits Loading progress, so no progress proves pre-staging rejection"
        );
    }

    #[test]
    fn frame_resolver_keeps_explicit_duration_and_default_paths_consistent() {
        let default = resolve_frames(&GenerationRequest::default()).unwrap();
        assert_eq!(
            default,
            ResolvedFrames {
                fps: 16,
                output: 81,
                output_latent: 21,
                generation_latent: 21,
            }
        );

        let from_duration = resolve_frames(&GenerationRequest {
            fps: Some(16),
            duration: Some(64.25),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(from_duration.output, 1_028);
        assert_eq!(from_duration.output_latent, 257);
        assert_eq!(from_duration.generation_latent, 257);

        let duration_error = resolve_frames(&GenerationRequest {
            fps: Some(16),
            duration: Some(64.3125),
            ..Default::default()
        })
        .expect_err("duration x fps resolving to 1,029 output frames must hit the same cap");
        assert!(matches!(duration_error, Error::Unsupported(_)));

        let explicit_wins = resolve_frames(&GenerationRequest {
            frames: Some(5),
            fps: Some(16),
            duration: Some(64.3125),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(explicit_wins.output, 5);
        assert_eq!(explicit_wins.output_latent, 2);
        assert_eq!(explicit_wins.generation_latent, 2);
    }

    fn v2v_request(source_frames: usize) -> GenerationRequest {
        let source = Image {
            width: 1,
            height: 1,
            pixels: vec![0; 3],
        };
        GenerationRequest {
            width: 512,
            height: 512,
            frames: Some(u32::try_from(source_frames).unwrap_or(u32::MAX)),
            count: 1,
            video_mode: Some("video_to_video".into()),
            conditioning: vec![Conditioning::VideoClip {
                frames: vec![source; source_frames],
                frame_idx: 0,
                strength: 0.5,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn v2v_source_frame_cap_is_resolved_and_rejected_before_staging() {
        let provider = unloaded();
        let accepted = v2v_request(1_028);
        Generator::validate(&provider, &accepted).expect("V2V boundary remains supported");
        let resolved = provider.validate_and_resolve_frames(&accepted).unwrap();
        assert_eq!(resolved.generation_latent, 257);
        let mut accepted_progress = 0;
        let error = provider
            .run(&accepted, &mut |_| accepted_progress += 1)
            .unwrap_err();
        assert!(error.to_string().contains("snapshot dir does not exist"));
        assert_eq!(accepted_progress, 0);

        let rejected = v2v_request(1_029);
        let validation_error = Generator::validate(&provider, &rejected)
            .expect_err("1,029 V2V source frames resolve to 258 generation latents");
        assert!(
            matches!(validation_error, mlx_gen::gen_core::Error::Unsupported(_)),
            "the V2V source cap must stay typed across the provider contract: {validation_error:?}"
        );
        let mut rejected_progress = 0;
        let run_error = provider
            .run(&rejected, &mut |_| rejected_progress += 1)
            .expect_err(
                "V2V source overflow must fail before the nonexistent snapshot is inspected",
            );
        assert!(
            matches!(run_error, Error::Unsupported(_)),
            "got: {run_error:?}"
        );
        assert_eq!(
            rejected_progress, 0,
            "no Loading progress proves the oversized source was rejected before staging"
        );
    }

    /// sc-20265 — `VideoClip.frame_idx` was silently swallowed by `run`'s `VideoClip { frames,
    /// strength, .. }` bind. A non-default offset is now the typed `Unsupported`, naming the field
    /// and the model, on BOTH the pre-flight and the run seam (both route through
    /// `validate_and_resolve_frames`).
    #[test]
    fn non_default_video_clip_frame_idx_is_refused_by_name() {
        let provider = unloaded();
        let mut offset = v2v_request(5);
        if let Conditioning::VideoClip { frame_idx, .. } = &mut offset.conditioning[0] {
            *frame_idx = 3;
        }

        let err = Generator::validate(&provider, &offset)
            .expect_err("a non-default frame_idx must be refused");
        assert!(
            matches!(err, mlx_gen::gen_core::Error::Unsupported(_)),
            "typed Unsupported, got {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("frame_idx"), "names the field: {msg}");
        assert!(msg.contains(MODEL_ID), "names the model: {msg}");

        // The run seam refuses it too, before any staging.
        let mut progress = 0;
        let run_error = provider
            .run(&offset, &mut |_| progress += 1)
            .expect_err("run must refuse it too");
        assert!(matches!(run_error, Error::Unsupported(_)), "{run_error:?}");
        assert!(run_error.to_string().contains("frame_idx"));
        assert_eq!(
            progress, 0,
            "no Loading progress proves the refusal ran before staging"
        );

        // Every clip is inspected, not just the one `run` routes on.
        let mut second = v2v_request(5);
        second.conditioning.insert(
            0,
            Conditioning::VideoClip {
                frames: Vec::new(),
                frame_idx: 0,
                strength: 1.0,
            },
        );
        if let Conditioning::VideoClip { frame_idx, .. } = &mut second.conditioning[1] {
            *frame_idx = 9;
        }
        assert!(Generator::validate(&provider, &second).is_err());
    }

    /// sc-20265 — the refusal is scoped to `frame_idx` alone. `strength` IS honored (it drives the
    /// AR init), so a NON-default strength must keep validating; and the default `frame_idx = 0` —
    /// what SceneWorks sends today — is untouched.
    #[test]
    fn video_clip_strength_is_untouched_and_default_frame_idx_still_passes() {
        let provider = unloaded();
        // `v2v_request` already carries a non-default strength of 0.5 at frame_idx 0.
        let honored = v2v_request(5);
        assert!(matches!(
            honored.conditioning[0],
            Conditioning::VideoClip { strength, .. } if strength == 0.5
        ));
        Generator::validate(&provider, &honored).unwrap();

        let mut default_strength = v2v_request(5);
        if let Conditioning::VideoClip { strength, .. } = &mut default_strength.conditioning[0] {
            *strength = 1.0;
        }
        Generator::validate(&provider, &default_strength).unwrap();
    }

    #[test]
    fn v2v_contract_rejects_crossed_carriers_strength_short_source_and_geometry() {
        let provider = unloaded();
        let valid = v2v_request(5);
        Generator::validate(&provider, &valid).expect("one exact clip is valid");

        let mut crossed = valid.clone();
        crossed.conditioning.push(Conditioning::Reference {
            image: Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
            strength: None,
        });
        assert!(Generator::validate(&provider, &crossed)
            .unwrap_err()
            .to_string()
            .contains("exactly one VideoClip"));

        for strength in [-0.01, 1.01, f32::NAN] {
            let mut invalid = valid.clone();
            let Conditioning::VideoClip {
                strength: actual, ..
            } = &mut invalid.conditioning[0]
            else {
                unreachable!()
            };
            *actual = strength;
            let error = Generator::validate(&provider, &invalid)
                .expect_err("out-of-range and non-finite clip strengths must fail");
            assert!(error.to_string().contains("strength"), "{error}");
        }

        let mut short = valid.clone();
        short.frames = Some(9);
        assert!(Generator::validate(&provider, &short)
            .unwrap_err()
            .to_string()
            .contains("source has 5 frames"));

        let mut crossed_geometry = valid;
        let Conditioning::VideoClip { frames, .. } = &mut crossed_geometry.conditioning[0] else {
            unreachable!()
        };
        frames[1].width = 2;
        assert!(Generator::validate(&provider, &crossed_geometry)
            .unwrap_err()
            .to_string()
            .contains("effective geometry"));
    }

    #[test]
    fn explicit_v2v_mode_cannot_fall_through_to_unconditioned_generation() {
        let provider = unloaded();
        let request = GenerationRequest {
            width: 512,
            height: 512,
            frames: Some(5),
            video_mode: Some("video_to_video".into()),
            ..Default::default()
        };
        assert!(Generator::validate(&provider, &request)
            .unwrap_err()
            .to_string()
            .contains("requires exactly one VideoClip"));
    }

    #[test]
    fn video_clip_requires_the_exact_v2v_mode() {
        let provider = unloaded();
        for crossed_mode in [None, Some("image_to_video")] {
            let mut request = v2v_request(5);
            request.video_mode = crossed_mode.map(Into::into);
            let error = Generator::validate(&provider, &request)
                .expect_err("a VideoClip must never infer or borrow a non-V2V public mode");
            assert!(
                error
                    .to_string()
                    .contains("requires video_mode=video_to_video"),
                "{crossed_mode:?}: {error}"
            );
        }
    }

    #[test]
    fn preload_render_plan_compiles_ar_geometry_kv_residency_and_refuses_invalid_requests() {
        use mlx_gen::gen_core::{CausalCodecStatePolicy, ChunkResidencyPolicy};

        let provider = unloaded();
        let req = GenerationRequest {
            width: 832,
            height: 480,
            count: 1,
            ..Default::default()
        };
        Generator::validate(&provider, &req).unwrap();
        let plan = provider.render_plan(&req).unwrap();

        assert_eq!(plan.route, KreaRealtimeRoute::T2v);
        assert_eq!(plan.width, 832);
        assert_eq!(plan.height, 480);
        assert_eq!(plan.fps, 16);
        assert_eq!(plan.output_frames, 81);
        assert_eq!(plan.output_latent_frames, 21);
        assert_eq!(plan.context_latent_frames, 0);
        assert_eq!(plan.generation_latent_frames, 21);
        assert_eq!(plan.frame_seq_length, 1560);
        assert_eq!(plan.seq_length, 21 * 1560);
        assert_eq!(plan.frames_per_block, 3);
        assert_eq!(plan.chunk_frame_counts, vec![3, 3, 3, 3, 3, 3, 3]);
        assert!(plan.context_chunk_frame_counts.is_empty());
        assert_eq!(plan.num_chunks, 7);
        assert_eq!(plan.n_steps, 5);
        assert_eq!(plan.total_outer_steps, 35);
        assert_eq!(
            plan.step_timesteps_f64,
            vec![1000.0, 937.5, 833.333_312_988_281_2, 625.0, 0.0]
        );
        assert_eq!(plan.execution_sigmas_f64.len(), 6);
        assert_eq!(plan.execution_sigmas_f64.last().copied(), Some(0.0));
        assert_eq!(plan.kv_residency.local_attn_frames, 6);
        assert_eq!(plan.kv_residency.max_attention_size, 6 * 1560);
        assert_eq!(plan.kv_residency.sink_frames, 0);
        assert_eq!(plan.kv_residency.sink_tokens, 0);
        assert_eq!(plan.kv_residency.steady_retained_kv_tokens, 6 * 1560);
        assert!(plan.kv_residency.do_kv_recomp);
        assert_eq!(plan.kv_residency.commits_per_generated_chunk, 1);
        assert_eq!(plan.kv_residency.context_commits, 0);
        assert_eq!(
            plan.chunk_residency,
            ChunkResidencyPolicy::TerminalWholeClip
        );
        assert_eq!(
            plan.causal_codec_state,
            CausalCodecStatePolicy::RollingKvWindow {
                max_attention_tokens: 6 * 1560,
                sink_tokens: 0,
            }
        );

        // I2V compiles context warm + remaining generated frames pre-load.
        let i2v_req = GenerationRequest {
            width: 512,
            height: 512,
            frames: Some(9),
            count: 1,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0; 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        let i2v_plan = provider.render_plan(&i2v_req).unwrap();
        assert_eq!(
            i2v_plan.route,
            KreaRealtimeRoute::I2v {
                reference_latent_frames: 1
            }
        );
        assert_eq!(i2v_plan.output_latent_frames, 3);
        assert_eq!(i2v_plan.context_latent_frames, 1);
        assert_eq!(i2v_plan.generation_latent_frames, 2);
        assert_eq!(i2v_plan.context_chunk_frame_counts, vec![1]);
        assert_eq!(i2v_plan.chunk_frame_counts, vec![2]);
        assert_eq!(i2v_plan.kv_residency.context_commits, 1);

        // V2V compiles source latent count + strength pre-load.
        let v2v_req = v2v_request(13); // 4 latent frames
        let v2v_plan = provider.render_plan(&v2v_req).unwrap();
        assert_eq!(
            v2v_plan.route,
            KreaRealtimeRoute::V2v {
                source_frames: 13,
                source_latent_frames: 4,
                strength: 0.5,
            }
        );
        assert_eq!(v2v_plan.chunk_frame_counts, vec![3, 1]);
        assert_eq!(v2v_plan.num_chunks, 2);

        // Pre-load refusals trigger before snapshot/weight access.
        let too_short_i2v = GenerationRequest {
            frames: Some(4),
            ..i2v_req
        };
        assert!(matches!(
            Generator::validate(&provider, &too_short_i2v),
            Err(mlx_gen::gen_core::Error::Unsupported(_))
        ));
        let zero_fps = GenerationRequest {
            fps: Some(0),
            ..req.clone()
        };
        assert!(matches!(
            Generator::validate(&provider, &zero_fps),
            Err(mlx_gen::gen_core::Error::Msg(_))
        ));
        assert!(matches!(
            KreaRealtimeRenderPlan::from_generation_request(&provider.config, &zero_fps),
            Err(Error::Unsupported(_))
        ));
        let zero_steps = GenerationRequest {
            steps: Some(0),
            ..req
        };
        assert!(matches!(
            Generator::validate(&provider, &zero_steps),
            Err(mlx_gen::gen_core::Error::Msg(_))
        ));
        assert!(matches!(
            KreaRealtimeRenderPlan::from_generation_request(&provider.config, &zero_steps),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn routed_terminal_chunk_emit_preserves_video_output_events_and_zero_copy_ptr() {
        use mlx_gen::gen_core::{
            CausalCodecStatePolicy, ChunkResidencyPolicy, GenerationReport, TerminalChunkCollector,
        };
        use mlx_gen::LoadPhase;

        struct ZeroCopyMockGeneration<'a> {
            report: AdapterApplyReport,
            frame_pixels_ptr: *const u8,
            frames_vec_ptr: *const Image,
            total_steps: u32,
            on_progress: &'a mut dyn FnMut(Progress),
        }

        impl ReportedGeneration for ZeroCopyMockGeneration<'_> {
            fn t2v(&mut self) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
                (self.on_progress)(Progress::Loading(LoadPhase::TextEncoder));
                (self.on_progress)(Progress::PromptEncodingComplete);
                (self.on_progress)(Progress::Loading(LoadPhase::Renderer));
                for current in 1..=self.total_steps {
                    (self.on_progress)(Progress::Step {
                        current,
                        total: self.total_steps,
                    });
                }
                (self.on_progress)(Progress::Decoding);

                let pixels = vec![123u8; 32 * 32 * 3];
                self.frame_pixels_ptr = pixels.as_ptr();
                let frames = vec![
                    Image {
                        width: 32,
                        height: 32,
                        pixels,
                    },
                    Image {
                        width: 32,
                        height: 32,
                        pixels: vec![7u8; 32 * 32 * 3],
                    },
                ];
                self.frames_vec_ptr = frames.as_ptr();
                Ok((
                    GenerationOutput::Video {
                        frames,
                        fps: 16,
                        audio: None,
                    },
                    vec![self.report.clone()],
                ))
            }

            fn i2v(
                &mut self,
                _image: &Image,
            ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
                self.t2v()
            }

            fn v2v(
                &mut self,
                _frames: &[Image],
                _strength: f32,
            ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
                self.t2v()
            }
        }

        let provider = unloaded();
        let req = GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            ..Default::default()
        };
        let plan = provider.render_plan(&req).unwrap();
        let report = AdapterApplyReport {
            adapter_path: PathBuf::from("/models/lora.safetensors"),
            applied: 3,
            skipped: vec![],
        };
        let mut events = Vec::new();
        let mut on_progress = |p: Progress| events.push(p);
        let mut mock = ZeroCopyMockGeneration {
            report: report.clone(),
            frame_pixels_ptr: std::ptr::null(),
            frames_vec_ptr: std::ptr::null(),
            total_steps: plan.total_outer_steps as u32,
            on_progress: &mut on_progress,
        };
        let mut collector = TerminalChunkCollector::new(CancelFlag::new());
        provider
            .finish_routed_generation_to_sink(
                GenerationRoute::T2v,
                &mut mock,
                &mut collector,
                Some(plan.causal_codec_state),
            )
            .unwrap();
        let expected_frames_ptr = mock.frames_vec_ptr;
        let expected_pixel_ptr = mock.frame_pixels_ptr;
        assert_eq!(provider.current_adapter_apply_reports(), vec![report]);

        let expected_events: Vec<Progress> = [
            Progress::Loading(LoadPhase::TextEncoder),
            Progress::PromptEncodingComplete,
            Progress::Loading(LoadPhase::Renderer),
        ]
        .into_iter()
        .chain(
            (1..=plan.total_outer_steps as u32).map(|current| Progress::Step {
                current,
                total: plan.total_outer_steps as u32,
            }),
        )
        .chain(std::iter::once(Progress::Decoding))
        .collect();
        assert_eq!(events, expected_events);

        let chunk = collector.chunk().unwrap();
        assert_eq!(chunk.sequence, 0);
        assert!(chunk.is_final);
        assert_eq!(
            chunk.residency_policy,
            Some(ChunkResidencyPolicy::TerminalWholeClip)
        );
        assert_eq!(
            chunk.causal_codec_policy,
            Some(CausalCodecStatePolicy::RollingKvWindow {
                max_attention_tokens: plan.kv_residency.max_attention_size,
                sink_tokens: 0,
            })
        );
        let output = collector.into_output().unwrap();
        let completed = GenerationReport::from_output(output);
        match completed.output {
            Some(GenerationOutput::Video { frames, fps, audio }) => {
                assert_eq!(fps, 16);
                assert!(audio.is_none());
                assert_eq!(frames.len(), 2);
                assert_eq!(frames.as_ptr(), expected_frames_ptr);
                assert_eq!(frames[0].pixels.as_ptr(), expected_pixel_ptr);
            }
            other => panic!("expected Some(Video) output, got {other:?}"),
        }
    }

    #[test]
    fn finish_routed_generation_to_sink_preserves_backpressure_error_on_saturated_bounded_sink() {
        use mlx_gen::gen_core::{BackpressureError, BoundedChunkSink, ChunkSink, OutputChunk};

        struct SequenceAdvancingSink<'a>(&'a mut BoundedChunkSink<GenerationOutput>);

        impl ChunkSink<GenerationOutput> for SequenceAdvancingSink<'_> {
            fn emit(
                &mut self,
                mut chunk: OutputChunk<GenerationOutput>,
            ) -> mlx_gen::gen_core::Result<()> {
                chunk.sequence = self.0.next_sequence();
                self.0.emit(chunk)
            }
        }

        struct SimpleMockGeneration {
            report: AdapterApplyReport,
        }

        impl ReportedGeneration for SimpleMockGeneration {
            fn t2v(&mut self) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
                Ok((
                    GenerationOutput::Video {
                        frames: Vec::new(),
                        fps: 16,
                        audio: None,
                    },
                    vec![self.report.clone()],
                ))
            }

            fn i2v(
                &mut self,
                _image: &Image,
            ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
                self.t2v()
            }

            fn v2v(
                &mut self,
                _frames: &[Image],
                _strength: f32,
            ) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
                self.t2v()
            }
        }

        let provider = unloaded();
        let report = AdapterApplyReport {
            adapter_path: PathBuf::from("/models/lora.safetensors"),
            applied: 1,
            skipped: vec![],
        };
        let mut bounded = BoundedChunkSink::new(1, CancelFlag::new()).unwrap();
        bounded
            .emit(OutputChunk::new(
                0,
                false,
                GenerationOutput::Video {
                    frames: Vec::new(),
                    fps: 16,
                    audio: None,
                },
            ))
            .unwrap();

        let err = {
            let mut mock = SimpleMockGeneration {
                report: report.clone(),
            };
            let mut adapter = SequenceAdvancingSink(&mut bounded);
            provider
                .finish_routed_generation_to_sink(
                    GenerationRoute::T2v,
                    &mut mock,
                    &mut adapter,
                    None,
                )
                .expect_err("saturated BoundedChunkSink must return typed BackpressureError")
        };
        assert_eq!(
            BackpressureError::from_error(&err),
            Some(&BackpressureError {
                sequence: 1,
                buffered: 1,
                max_buffered_chunks: 1,
            })
        );
        assert_eq!(bounded.backpressure_events(), 1);
        assert_eq!(provider.current_adapter_apply_reports(), vec![report]);
    }
}
