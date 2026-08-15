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

/// Stable identity + advertised capabilities for Krea Realtime 14B (Wan-2.1-T2V-14B backbone,
/// autoregressive self-forcing **text-to-video**; **CFG off** → no negative prompt / no guidance; a
/// fixed Self-Forcing few-step sampler; a rolling causal KV cache).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
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
            supports_guidance: false,
            supports_true_cfg: false,
            // i2v / v2v conditioning (sc-8440 S7): a `Reference` still warms the AR KV cache
            // (image-to-video, like the sibling `svd_xt` image→video), and a `VideoClip` source drives
            // the strength-controlled AR init (video-to-video, like `bernini`'s v2v). Both are wired in
            // the engine (`generate_i2v` / `generate_v2v`) and routed by `run`, so advertising them is
            // honest. `VideoClip` is video-modality conditioning (allowed here — modality is Video).
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
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            // H/W align to patch×vae_stride = 16 (z16 VAE spatial stride 8, patch 2); mirror Wan's cap.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            // Not a distilled fixed-schedule model: any step count the shared sanity caps
            // admit is renderable (sc-19502).
            supported_steps: Vec::new(),
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
            component_precision_floors: &[],
            // The AR regime is built on a rolling causal KV cache (sc-8436 S3 / sc-8438 S5).
            supports_kv_cache: true,
            requires_sigma_shift: false,
            // Every route calls `stage_components`: UMT5 is loaded, evaluated, and dropped before
            // the DiT + VAE phase. There is no request-selectable Resident mode.
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: true,
            // Batch whole-clip form in S6; the realtime streaming decode is the streaming epic.
            supports_preview: false,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface: pure video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            // z16 VAE stride 8 × the Wan DiT's 2×2 latent patch: explicit dimensions must land
            // on a 16px grid or integer division would silently render a smaller clip.
            size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 16 },
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
    let config = KreaRealtimeConfig::from_model_dir(&root)?;
    Ok(Box::new(KreaRealtime {
        descriptor: descriptor(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant: spec.quantize,
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
        resolve_frames(req)
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

    /// The advertised surface is CFG-off video with the Self-Forcing few-step sampler, now advertising
    /// **i2v** (`Reference`) + **v2v** (`VideoClip`) conditioning (sc-8440 S7).
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
        // S7: advertises i2v (Reference) + v2v (VideoClip) — the engine wires + routes both.
        assert!(
            c.accepts(ConditioningKind::Reference),
            "i2v: a reference still is accepted (S7)"
        );
        assert!(
            c.accepts(ConditioningKind::VideoClip),
            "v2v: a source clip is accepted (S7)"
        );
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
            ..Default::default()
        };

        Generator::validate(&provider, &request(1_028))
            .expect("1,028 output frames resolve to the maximum 257 latent frames");
        let mut i2v_boundary = request(1_028);
        i2v_boundary.conditioning = vec![Conditioning::Reference {
            image: Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
            strength: None,
        }];
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
            count: 1,
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
        Generator::validate(&provider, &accepted)
            .expect("1,028 V2V source frames resolve to the maximum 257 generation latents");
        let resolved = provider.validate_and_resolve_frames(&accepted).unwrap();
        assert_eq!(
            resolved.output, DEFAULT_FRAMES,
            "the output trim stays defaulted"
        );
        assert_eq!(resolved.output_latent, 21);
        assert_eq!(resolved.generation_latent, 257);
        let mut accepted_progress = 0;
        let accepted_run_error = provider
            .run(&accepted, &mut |_| accepted_progress += 1)
            .expect_err("an accepted boundary request reaches the nonexistent snapshot guard");
        assert!(
            matches!(accepted_run_error, Error::Msg(_))
                && accepted_run_error.to_string().contains("snapshot dir does not exist"),
            "the accepted boundary must pass the cap and fail only on missing weights: {accepted_run_error:?}"
        );
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
}
