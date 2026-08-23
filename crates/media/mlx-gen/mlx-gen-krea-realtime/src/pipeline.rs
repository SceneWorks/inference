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

use mlx_gen::{
    default_seed, AdapterApplyReport, AdapterSpec, CancelFlag, Capabilities, Conditioning,
    ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image, Modality,
    ModelDescriptor, Progress, Quant, Result, SizeFloor, WeightsSource,
};

use crate::config::{KreaRealtimeConfig, MODEL_ID};
use crate::t2v::{
    bounded_latent_frame_count, generate_i2v_reported, generate_t2v_reported,
    generate_v2v_reported, KreaRealtimeJob,
};

/// The Self-Forcing few-step sampler name Krea Realtime advertises (a fixed short per-block flow-match
/// renoise schedule, not a selectable classic solver). Distinct from Wan's UniPC/DPM++ so a consumer
/// knows this model runs the AR few-step regime.
pub const SELF_FORCING_SAMPLER: &str = "self_forcing";

/// Default output cadence when a request omits `fps` (the reference realtime-video default).
const DEFAULT_FPS: u32 = 16;
/// Default requested **output** frame count when a request omits `frames`/`duration`. The AR path turns
/// this into `(81 − 1)/4 + 1 = 21` latent frames; the non-causal z16 Wan VAE decode then yields
/// `4 · 21 = 84` output frames (`output = 4 · latent`, not `(latent − 1)·4 + 1`), which the pipeline
/// trims back to this requested 81 (see [`crate::t2v::decode_latents_to_video`]). 81 is the reference
/// realtime-video canonical clip length.
const DEFAULT_FRAMES: u32 = 81;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedFrames {
    fps: u32,
    output: u32,
    output_latent: usize,
    generation_latent: usize,
}

/// Resolve the output cadence/count and its z16 latent count exactly once for provider validation and
/// execution. Keeping explicit-frames, duration-derived, and default precedence here prevents the
/// preflight and run paths from disagreeing about the model-local allocation cap.
fn resolve_frames(req: &GenerationRequest) -> Result<ResolvedFrames> {
    let fps = req.fps.unwrap_or(DEFAULT_FPS);
    let output = req.frames.unwrap_or_else(|| {
        req.duration
            .map(|duration| ((duration * fps as f32).round() as u32).max(1))
            .unwrap_or(DEFAULT_FRAMES)
    });
    let output_latent = bounded_latent_frame_count("requested output", output as usize)?;
    // V2V derives its actual full-noise allocation from the source clip, then trims decode to the
    // separately resolved requested output. Cap both dimensions without changing that output/trim
    // contract. Route selection below uses the same first VideoClip.
    let generation_latent = if let Some(source_frames) =
        req.conditioning
            .iter()
            .find_map(|conditioning| match conditioning {
                Conditioning::VideoClip { frames, .. } => Some(frames.as_slice()),
                _ => None,
            }) {
        bounded_latent_frame_count("V2V source", source_frames.len())?
    } else {
        output_latent
    };
    Ok(ResolvedFrames {
        fps,
        output,
        output_latent,
        generation_latent,
    })
}

/// Refuse the [`Conditioning::VideoClip`] knob Krea Realtime does not implement (sc-20265).
///
/// The variant's `strength` **is** honored here — it drives the strength-controlled AR init
/// ([`crate::t2v::generate_v2v`]), which is why `run` binds it on the v2v route. `frame_idx` is the
/// other half of the payload and it is **not**: it names the output latent frame an in-context clip
/// is appended at, and Krea Realtime is autoregressive — the source clip seeds the rolling causal KV
/// cache from step zero and the model generates forward from there. There is no output timeline to
/// splice a clip into partway through, so an offset has nothing to mean.
///
/// Until sc-20265 it was read past silently: `run`'s route match binds `VideoClip { frames,
/// strength, .. }` and the `..` swallowed it. The sc-19571 rule is that a control either works or is
/// refused with a clear error, so this is the refusal for the half that does not work.
///
/// It fires **only on a non-default value** — `frame_idx = 0` (the contract default, and what
/// SceneWorks sends today) passes through unchanged, and `strength` is untouched at any value.
///
/// Checked over **every** clip rather than the first: `run` routes on the first `VideoClip`, so a
/// bad value on clip two would otherwise be the one silently dropped.
///
/// Typed [`Error::Unsupported`], not [`Error::Msg`]: the worker classifies `Unsupported` as a
/// user-facing invalid-payload refusal and `Msg` as an opaque internal engine failure.
fn reject_unimplemented_video_clip_knobs(req: &GenerationRequest) -> Result<()> {
    for clip in req.video_clips() {
        if clip.frame_idx != 0 {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID} does not implement VideoClip frame_idx (got {}); remove it or leave it \
                 at the default 0 — Krea Realtime is autoregressive: the source clip seeds the \
                 rolling causal KV cache from step zero, so there is no output position to splice \
                 it at. (VideoClip strength IS honored and is unaffected.)",
                clip.frame_idx
            )));
        }
    }
    Ok(())
}

/// Krea V2V has exactly one conditioning carrier. Multiple clips, a clip plus a still, malformed
/// strength, or a source shorter than the requested output used to pass validation and let `run`
/// silently choose the first clip. Refuse that ambiguity before any staging or source allocation.
fn validate_v2v_contract(req: &GenerationRequest, resolved: ResolvedFrames) -> Result<()> {
    let clips = req
        .conditioning
        .iter()
        .filter_map(|conditioning| match conditioning {
            Conditioning::VideoClip {
                frames,
                strength,
                frame_idx,
            } => Some((frames, *strength, *frame_idx)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let explicitly_v2v = req.video_mode.as_deref() == Some("video_to_video");
    if clips.is_empty() {
        if explicitly_v2v {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID} video_to_video requires exactly one VideoClip carrier"
            )));
        }
        return Ok(());
    }
    if clips.len() != 1 || req.conditioning.len() != 1 {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID} video_to_video requires exactly one VideoClip and no other conditioning carriers"
        )));
    }
    let (frames, strength, _) = clips[0];
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID} VideoClip strength must be finite in [0, 1], got {strength}"
        )));
    }
    if frames.len() < resolved.output as usize {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID} video_to_video source has {} frames but the requested output requires {}",
            frames.len(),
            resolved.output
        )));
    }
    let Some(first) = frames.first() else {
        unreachable!("the source-length check rejects an empty clip")
    };
    if first.width == 0
        || first.height == 0
        || frames.iter().any(|frame| {
            frame.width != first.width
                || frame.height != first.height
                || frame.pixels.len()
                    != (u64::from(frame.width) * u64::from(frame.height) * 3) as usize
        })
    {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID} VideoClip frames must carry one non-zero, RGB8 effective geometry"
        )));
    }
    Ok(())
}

/// Stable identity + advertised capabilities for Krea Realtime 14B (Wan-2.1-T2V-14B backbone,
/// autoregressive self-forcing **text-to-video**; **CFG off** → no negative prompt / no guidance; a
/// fixed Self-Forcing few-step sampler; a rolling causal KV cache).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "krea_realtime",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG-off model: the AR few-step denoise runs a single batch-1 forward per step with no
            // unconditional branch, so there is no negative-prompt / guidance axis (sc-8437 S4).
            supports_negative_prompt: false,
            // Direct provider capability remains broader than SC-20770's worker/admission route.
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
            // Wan-family style-LoRA / LoKr (sc-15015 S14; extended to the packed tiers by sc-15203 S19):
            // Krea Realtime 14B is Wan-2.1-14B T2V weight-for-weight, so a diffusers / PEFT / kohya /
            // LoKr file installs onto the DiT as **forward-time residuals** via the shared
            // diff-patch-aware strict path. Low-rank residuals stay additive over dense or packed
            // bases, while lightx2v `.diff_b`/norm `.diff` targets fold through dense parameters that
            // exist on every tier (sc-15326).
            // `LoadSpec::adapters` is honored on every tier in the load path (`t2v::load_transformer`),
            // so advertising these is capability-honest at Q4/Q8 as well as bf16.
            supports_lora: true,
            supports_lokr: true,
            samplers: vec![SELF_FORCING_SAMPLER],
            // H/W align to patch×vae_stride = 16 (z16 VAE spatial stride 8, patch 2); mirror Wan's cap.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Three tiers (sc-15203, S19): **Q4** (~7 GB) / **Q8** (~14 GB) / **bf16** (~28 GB, the
            // absence of a `Quant`). A 14B bf16 DiT is ~28 GB resident and barely runnable on Mac, so the
            // quantized tiers are the practically-usable ones — the SceneWorks manifest picks **Q4** as
            // the default for this dense-video engine (sc-10750's dense-video convention); this slice is
            // the engine's advertised surface, the per-model default tier is a manifest field
            // (`mlx.quantize`) that rides S11 (sc-8444) in the SceneWorks repo, not a gen-core one.
            //
            // The tiers ship **pre-quantized (packed) on disk** — `dit.safetensors` carries the MLX
            // affine triple for the Wan `_quantize_predicate` Linears and the reused
            // `WanTransformer::from_weights` builds them packed (`load::resolve_snapshot_quant`). A
            // load-time `LoadSpec::quantize` over a *dense* bf16 snapshot is also honored (the
            // `AdaptableLinear::quantize` path the sibling Wan/SCAIL-2 providers use), and a request that
            // conflicts with a packed snapshot's own tier is a hard error rather than a silent downgrade
            // (`load::resolve_load_time_quant`). Both are what this slice advertises.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // The AR regime is built on a rolling causal KV cache (sc-8436 S3 / sc-8438 S5).
            supports_kv_cache: true,
            // Every route calls `stage_components`: UMT5 is loaded, evaluated, and dropped before
            // the DiT + VAE phase. There is no request-selectable Resident mode.
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: true,
            // Batch whole-clip form in S6; the realtime streaming decode is the streaming epic.
            supports_preview: false,
            // No audio surface: pure video model.
            audio_sample_rates: vec![],
            // z16 VAE stride 8 × the Wan DiT's 2×2 latent patch: explicit dimensions must land
            // on a 16px grid or integer division would silently render a smaller clip.
            size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 16 },
            ..Default::default()
        },
    }
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
mlx_gen::register_generators! { pub(crate) const REGISTRATION = descriptor => load }

impl Generator for KreaRealtime {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        self.adapter_reports
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
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
        self.validate_and_resolve_frames(req)
            .map(|_| ())
            .map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.run(req, on_progress).map_err(Into::into)
    }
}

impl KreaRealtime {
    fn validate_and_resolve_frames(&self, req: &GenerationRequest) -> Result<ResolvedFrames> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        // sc-20265 — refuse the per-clip knob this engine does not implement rather than binding it
        // under a `..` and dropping it. Both `validate` and `run` route through here, so the
        // pre-flight and the render cannot disagree.
        reject_unimplemented_video_clip_knobs(req)?;
        let frames = resolve_frames(req)?;
        validate_v2v_contract(req, frames)?;
        Ok(frames)
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

    /// Execute every advertised modality through the same report-publishing finish seam. Keeping the
    /// route match inside this method makes it impossible for a T2V/I2V/V2V call site to consume a
    /// reported generation while forgetting to expose its adapter outcome.
    fn finish_routed_generation(
        &self,
        route: GenerationRoute<'_>,
        generation: &mut impl ReportedGeneration,
    ) -> Result<GenerationOutput> {
        let reported = match route {
            GenerationRoute::T2v => generation.t2v(),
            GenerationRoute::I2v(image) => generation.i2v(image),
            GenerationRoute::V2v { frames, strength } => generation.v2v(frames, strength),
        };
        self.finish_reported_generation(reported)
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
        let frames = self.validate_and_resolve_frames(req)?;

        let job = KreaRealtimeJob {
            prompt: &req.prompt,
            width: req.width,
            height: req.height,
            num_frames: frames.output,
            fps: frames.fps,
            seed: req.seed.unwrap_or_else(default_seed),
            steps: req.steps.map(|s| s as usize),
        };

        // Route on the advertised conditioning (sc-8440 S7): a `VideoClip` source → v2v; else a
        // `Reference` still → i2v; else text-to-video. Advertising `Reference`/`VideoClip` in the
        // descriptor makes `validate_request` accept them, so `run` MUST honor them (capability honesty)
        // rather than silently generating t2v. The worker's mapping of its own reference inputs onto
        // these `GenerationRequest` conditioning entries is S10.
        let route = if let Some((frames, strength)) =
            req.conditioning.iter().find_map(|c| match c {
                Conditioning::VideoClip {
                    frames, strength, ..
                } => Some((frames.as_slice(), *strength)),
                _ => None,
            }) {
            GenerationRoute::V2v { frames, strength }
        } else if let Some(image) = req.conditioning.iter().find_map(|c| match c {
            Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        }) {
            GenerationRoute::I2v(image)
        } else {
            GenerationRoute::T2v
        };
        let mut generation = ProductReportedGeneration {
            root: &self.root,
            config: &self.config,
            job: &job,
            output_latent: frames.output_latent,
            generation_latent: frames.generation_latent,
            adapters: &self.adapters,
            quant: self.quant,
            cancel: &req.cancel,
            on_progress,
        };
        self.finish_routed_generation(route, &mut generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Generator::adapter_apply_reports(&provider),
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
                Generator::adapter_apply_reports(&provider),
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
            video_mode: Some("video_to_video".to_owned()),
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
            video_mode: Some("video_to_video".to_owned()),
            ..Default::default()
        };
        assert!(Generator::validate(&provider, &request)
            .unwrap_err()
            .to_string()
            .contains("requires exactly one VideoClip"));
    }
}
