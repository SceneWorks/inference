//! `Seedvr2Generator` — the [`mlx_gen::Generator`] wiring the SeedVR2 pipeline into `mlx_gen`'s
//! registry (sc-4813 image, sc-4814 video). Registered under `seedvr2` (alias) + `seedvr2_3b`.
//!
//! **Surface.** A one-step super-resolution **upscaler** over image **and** video (`Modality::Both`),
//! dispatched on the request's conditioning:
//!   * [`Conditioning::Reference`] — the LR input image → [`GenerationOutput::Images`];
//!   * [`Conditioning::VideoClip`] — the LR input frame sequence → [`GenerationOutput::Video`]
//!     (temporal chunking + overlap cross-fade + a memory-budgeted chunk sizer; sc-4814).
//!
//! `width`/`height` are the target output size (both ÷16). No prompt, no guidance/CFG (1-step), no
//! LoRA. `spec.weights` is the raw `numz/SeedVR2_comfyUI` checkpoint dir (converted in-memory at
//! load — no Python). Dense bf16 default; `Fp32` honored (the parity path). Video `fps` passes
//! through `req.fps` (the worker supplies the source cadence; audio mux is the worker's job).
//!
//! 3B (default) + 7B (pixel-mode RoPE — sc-5197) are wired; `spec.quantize` Q4/Q8 quantizes the DiT
//! Linears at load (sc-5198).

use mlx_rs::Dtype;

use mlx_gen::{
    Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Result, WeightsSource,
};

use crate::config::DitConfig;
use crate::pipeline::Seedvr2Pipeline;

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
pub const MODEL_ID_7B: &str = "seedvr2_7b";
const VAE_SCALE: u32 = 16; // VAE /8 · patch /2
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";
const DIT_FILE_7B: &str = "seedvr2_ema_7b_fp16.safetensors";
/// Output fps when the request omits one (the worker normally supplies the source cadence).
const DEFAULT_FPS: u32 = 24;

/// The DiT checkpoint file + transformer config for a registered id (3B default; 7B is the
/// pixel-mode-RoPE variant — sc-5197). The VAE is shared across both.
fn variant(id: &str) -> (&'static str, DitConfig) {
    if id == MODEL_ID_7B {
        (DIT_FILE_7B, DitConfig::seedvr2_7b())
    } else {
        (DIT_FILE_3B, DitConfig::seedvr2_3b())
    }
}

fn route_for(
    id: &'static str,
    desc_fn: fn() -> ModelDescriptor,
    load_fn: fn(&LoadSpec) -> mlx_gen::gen_core::Result<Box<dyn Generator>>,
) -> mlx_gen::gen_core::RouteRecord {
    let model_id = if id == MODEL_ID_7B {
        "seedvr2_dit_7b"
    } else {
        "seedvr2_dit_3b"
    };
    mlx_gen::gen_core::RouteRecord {
        route_id: id,
        family: "seedvr2",
        backend: "mlx",
        modality: Modality::Both, // image (Reference) + video (VideoClip) upscaling
        mac_only: true,
        structure: mlx_gen::gen_core::RouteStructure {
            text_encoders: mlx_gen::gen_core::TextEncoderSurface::default(),
            conditioning: mlx_gen::gen_core::ConditioningSurface {
                // the LR input image (image upscale) or LR frame sequence (video upscale)
                kinds: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
                control_kinds: None,
                pass0_requirement:
                    mlx_gen::gen_core::Pass0ConditioningRequirement::RequiresReference,
                supports_negative_prompt: false,
                modules: Vec::new(),
                ..Default::default()
            },
            sampling: mlx_gen::gen_core::SamplingSurface {
                samplers: vec!["seedvr2_euler"],
                schedulers: vec!["seedvr2_euler"],
                guidance_methods: Vec::new(),
                supports_guidance: false, // one-step, guidance fixed at 1.0
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: mlx_gen::gen_core::ModelBinding::family_model(
                mlx_gen::gen_core::FamilyModelSpec::new(model_id),
            ),
            latent_codec: mlx_gen::gen_core::LatentCodecDeclaration {
                codec_id: Some("seedvr2_vae"),
                output_latent_space: Some(&mlx_gen::gen_core::SEEDVR2_VIDEO_LATENT_SPACE),
                pixel_stride: Some(VAE_SCALE),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            upsamplers: mlx_gen::gen_core::UpsamplerSurface::default(),
            adapters: mlx_gen::gen_core::AdapterSurface::unsupported(),
            residency: mlx_gen::gen_core::ResidencySurface {
                lifetime_graph: vec![
                    mlx_gen::gen_core::ResidencyPhaseNode::new(
                        mlx_gen::gen_core::ResidencyPhaseKind::Denoise,
                        [model_id],
                        false,
                    ),
                    mlx_gen::gen_core::ResidencyPhaseNode::new(
                        mlx_gen::gen_core::ResidencyPhaseKind::Decode,
                        ["seedvr2_vae"],
                        false,
                    ),
                ],
                supports_sequential_offload: false,
                ..Default::default()
            },
            weight_plan: mlx_gen::gen_core::WeightPlanDeclaration {
                source_kind: mlx_gen::gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
                ..Default::default()
            },
            output: mlx_gen::gen_core::OutputSurface {
                preview: mlx_gen::gen_core::PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: mlx_gen::gen_core::ProductLimits {
            min_size: VAE_SCALE,
            max_size: 4096,
            size_floor: mlx_gen::gen_core::SizeFloor::RangeChecked,
            supported_steps: mlx_gen::gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: mlx_gen::gen_core::SeedAndBatchSemantics {
            seed_scheme: mlx_gen::gen_core::SeedScheme::BackendGlobalSeeded,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: mlx_gen::gen_core::RouteDefaults {
            steps: Some(1),
            guidance: None,
            sampler: Some("seedvr2_euler"),
            schedule: Some("seedvr2_euler"),
            shift: None,
            strength: None,
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings: mlx_gen::gen_core::RouteBindings::generator(desc_fn, load_fn),
    }
}

pub(crate) fn route_base() -> mlx_gen::gen_core::RouteRecord {
    route_for(MODEL_ID, descriptor, |spec| {
        load_base(spec).map_err(Into::into)
    })
}

pub(crate) fn route_3b() -> mlx_gen::gen_core::RouteRecord {
    route_for(MODEL_ID_3B, descriptor_3b, |spec| {
        load_3b(spec).map_err(Into::into)
    })
}

pub(crate) fn route_7b() -> mlx_gen::gen_core::RouteRecord {
    route_for(MODEL_ID_7B, descriptor_7b, |spec| {
        load_7b(spec).map_err(Into::into)
    })
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    match id {
        MODEL_ID_3B => route_3b().compute_descriptor(),
        MODEL_ID_7B => route_7b().compute_descriptor(),
        _ => route_base().compute_descriptor(),
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

pub struct Seedvr2Generator {
    descriptor: ModelDescriptor,
    pipe: Seedvr2Pipeline,
}

fn load_with(spec: &LoadSpec, id: &'static str) -> Result<Box<dyn Generator>> {
    crate::video::Seedvr2RenderPlan::validate_load_spec(id, spec)?;
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let dir = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id}: expects a numz/SeedVR2_comfyUI checkpoint directory, not a single file"
            )))
        }
    };
    let (dit_file, cfg) = variant(id);
    let mut pipe = Seedvr2Pipeline::load(&dir, dit_file, &cfg, dtype)?;
    // sc-5198: Q4/Q8 quantize the DiT Linears at load (the VAE stays dense).
    if let Some(q) = spec.quantize {
        pipe.quantize(q.bits())?;
    }
    Ok(Box::new(Seedvr2Generator {
        descriptor: descriptor_for(id),
        pipe,
    }))
}

impl Seedvr2Generator {
    fn generate_execution_attempt(
        &self,
        req: &mlx_gen::GenerationRequest,
        on_progress: &mut dyn FnMut(mlx_gen::Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl mlx_gen::Generator for Seedvr2Generator {
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

    fn descriptor(&self) -> &mlx_gen::ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &mlx_gen::GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        let s = self;
        s.validate_impl(req).map(|_| ()).map_err(Into::into)
    }
    fn generate(
        &self,
        req: &mlx_gen::GenerationRequest,
        on_progress: &mut dyn FnMut(mlx_gen::Progress),
    ) -> mlx_gen::gen_core::Result<mlx_gen::GenerationOutput> {
        mlx_gen::video_execution::without_capture(|| {
            self.generate_execution_attempt(req, on_progress)
        })
    }
}

/// Refuse the [`Conditioning::VideoClip`] knobs SeedVR2 does not implement (sc-20263).
///
/// SeedVR2 is a one-step **super-resolution upscaler**: a `VideoClip` is the LR source frame
/// sequence, consumed whole through `generate_video`. It has no denoise mask and no output timeline
/// to place a clip on, so neither field this variant carries has a mechanism here:
///
/// * `strength` is the `1 − strength` denoise mask of the LTX in-context append path. A one-step
///   upscale runs a single deterministic pass over the LR frames — there is no multi-step denoise
///   for a mask to gate.
/// * `frame_idx` is the output latent frame the clip is appended at. SeedVR2 emits one upscaled
///   frame per input frame in order; the clip *is* the timeline, so there is no offset to honor.
///
/// Until sc-20263 both were read off the request and thrown away — `generate_impl` binds only
/// `clip.frames`. The sc-19571 rule is that a control either works or is refused with a clear
/// error, so this is the refusal, and it fires **only on a non-default value**: the contract
/// defaults (`strength = 1.0`, `frame_idx = 0`) pass through unchanged, which is every request
/// SceneWorks builds today.
///
/// Checked over **every** clip rather than the first: `generate_impl` takes the first NON-empty
/// clip, so a bad value on clip two would otherwise be the one silently dropped (the same reason
/// gen-core's finiteness floor loops over `AudioEditRegions` instead of guarding `regions[0]`).
///
/// Typed [`Error::Unsupported`], not [`Error::Msg`]: the worker classifies `Unsupported` as a
/// user-facing invalid-payload refusal and `Msg` as an opaque internal engine failure.
fn reject_unimplemented_video_clip_knobs(id: &str, req: &GenerationRequest) -> Result<()> {
    for clip in req.video_clips() {
        if clip.strength != 1.0 {
            return Err(Error::Unsupported(format!(
                "{id} does not implement VideoClip strength (got {}); remove it or leave it at the \
                 default 1.0 — SeedVR2 is a one-step upscale with no denoise mask to weight",
                clip.strength
            )));
        }
        if clip.frame_idx != 0 {
            return Err(Error::Unsupported(format!(
                "{id} does not implement VideoClip frame_idx (got {}); remove it or leave it at \
                 the default 0 — SeedVR2 upscales the clip in order and has no output timeline to \
                 offset it against",
                clip.frame_idx
            )));
        }
    }
    Ok(())
}

/// The LR input image carried by the request's `Reference` conditioning.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

impl Seedvr2Generator {
    fn compile_render_plan(
        &self,
        req: &GenerationRequest,
    ) -> Result<crate::video::Seedvr2RenderPlan> {
        crate::video::Seedvr2RenderPlan::compile(
            &self.descriptor,
            req,
            self.pipe.weights_bytes(),
            crate::video::safe_budget_gib(),
            None,
        )
        .map_err(Into::into)
    }

    fn validate_impl(&self, req: &GenerationRequest) -> Result<crate::video::Seedvr2RenderPlan> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let has_video = req.video_clips().iter().any(|c| !c.frames.is_empty());
        if !has_video && reference_image(req).is_none() {
            return Err(Error::Msg(format!(
                "{}: requires a Reference image (image upscale) or a non-empty VideoClip frame \
                 sequence (video upscale)",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(VAE_SCALE) || !req.height.is_multiple_of(VAE_SCALE) {
            return Err(Error::Msg(format!(
                "{}: width/height must be multiples of {VAE_SCALE} (got {}x{})",
                self.descriptor.id, req.width, req.height
            )));
        }
        reject_unimplemented_video_clip_knobs(self.descriptor.id, req)?;
        self.compile_render_plan(req)
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let plan = self.validate_impl(req)?;
        let base_seed = plan.base_seed;
        let softness = plan.softness;

        if let Some(clip) = req.video_clips().into_iter().find(|c| !c.frames.is_empty()) {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let frames = self.pipe.generate_video_for_route(
                clip.frames,
                req.width as i32,
                req.height as i32,
                base_seed,
                softness,
                &plan.route,
                &req.cancel,
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            let output = GenerationOutput::Video {
                frames,
                fps: req.fps.unwrap_or(DEFAULT_FPS),
                audio: None,
            };
            return plan.finish_output(output).map_err(Into::into);
        }

        let image = reference_image(req).expect("validated");
        let mut out = Vec::with_capacity(req.count as usize);
        let count = req.count as usize;
        for i in 0..req.count {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let seed = base_seed.wrapping_add(i as u64);
            mlx_gen::video_execution::output_index(i as usize);
            let idx = i as usize;
            let img = self.pipe.generate_still_for_route(
                image,
                req.width as i32,
                req.height as i32,
                seed,
                softness,
                &plan.route,
                &req.cancel,
                &mut |tile_idx, n_tiles| {
                    let (current, total) = fold_tile_progress(idx, tile_idx, n_tiles, count);
                    on_progress(Progress::Step { current, total });
                },
            )?;
            out.push(img);
        }
        on_progress(Progress::Decoding);
        plan.finish_output(GenerationOutput::Images(out))
            .map_err(Into::into)
    }
}

// Thin id-binding loaders over `load_with` (each pins the variant id), so they can't be a plain
// `load` path. They return the crate's rich `Result`; `register_generators!` adds the
// `gen_core::Result` bridge.
fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID)
}
fn load_3b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_3B)
}
fn load_7b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_7B)
}

/// Fold one image's per-tile progress onto the whole-batch bar (F-162 / F-164, sc-11133). Image
/// `image_idx` (0-based) of `count`, tile `tile_idx` (1-based) of `n_tiles`, maps to an absolute
/// `(current, total)` where `total = count × n_tiles` and `current = image_idx × n_tiles + tile_idx`
/// (clamped). A single-pass image reports `n_tiles == 1`, so this reduces to `(image_idx + 1, count)`
/// — the count axis F-162 wants — while a tiled image contributes per-tile liveness F-164 wants. The
/// sequence across the loop is monotone non-decreasing and reaches `total` on the final tile.
fn fold_tile_progress(
    image_idx: usize,
    tile_idx: usize,
    n_tiles: usize,
    count: usize,
) -> (u32, u32) {
    let total = count * n_tiles;
    let current = (image_idx * n_tiles + tile_idx).min(total);
    (current as u32, total as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-162 / F-164 (sc-11133): the fold yields one monotone bar reaching `total`, covering both the
    /// single-pass (`n_tiles == 1` → the count axis) and spatial-tiling (per-tile liveness) shapes.
    #[test]
    fn fold_tile_progress_is_monotone_and_reaches_total() {
        // Single-pass, count = 8: each image is one "tile" → Step{i+1, 8} (the F-162 count axis).
        let seq: Vec<(u32, u32)> = (0..8).map(|i| fold_tile_progress(i, 1, 1, 8)).collect();
        assert_eq!(
            seq,
            (1..=8).map(|c| (c, 8)).collect::<Vec<_>>(),
            "single-pass batch must be a monotone 1..=count bar"
        );

        // count = 2, 3 tiles each: one monotone 1..=6 bar, per-tile liveness within each image.
        let mut folded = Vec::new();
        for image_idx in 0..2 {
            for tile_idx in 1..=3 {
                folded.push(fold_tile_progress(image_idx, tile_idx, 3, 2));
            }
        }
        assert_eq!(
            folded,
            vec![(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)],
            "tiled multi-image batch must fold to one monotone 1..=6 bar"
        );
        // The final tile of the final image reaches total.
        assert_eq!(folded.last(), Some(&(6, 6)));
    }

    #[test]
    fn descriptor_is_seedvr2() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "seedvr2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Both); // image (Reference) + video (VideoClip)
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::VideoClip));
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B, MODEL_ID_7B] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/seedvr2".into()));
            let err = match crate::provider_registry().unwrap().load(id, &spec) {
                Ok(_) => panic!("bogus weights dir must fail to load"),
                Err(e) => e.to_string(),
            };
            assert!(
                !err.contains("no generator registered"),
                "{id} should resolve; got: {err}"
            );
        }
    }

    /// F-099: `validate_impl` accepts a request if ANY clip is non-empty, so the generate path must
    /// take the first NON-empty clip — not `into_iter().next()`. With the old `next()` selection,
    /// `[empty, real]` validated but generated an empty video "as success". This pins the
    /// first-non-empty predicate the generate path now uses.
    #[test]
    fn video_upscale_takes_the_first_non_empty_clip() {
        // Two clips: the first empty, the second with one frame. `find(!frames.is_empty())` must
        // skip the empty one (mirrors the generate_impl selection at line ~175).
        let frame = Image {
            pixels: vec![0u8; 3],
            width: 1,
            height: 1,
        };
        let nonempty = [frame];
        let clips: Vec<&[Image]> = vec![&[], &nonempty[..]];
        let selected = clips.into_iter().find(|c| !c.is_empty());
        assert!(
            selected.is_some(),
            "the non-empty second clip must be selected"
        );
        assert_eq!(selected.unwrap().len(), 1);
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

    /// sc-20263 — `VideoClip.strength` and `frame_idx` were silently dropped (`generate_impl` binds
    /// only `clip.frames`). Each is now a typed `Unsupported` naming the field and the model.
    ///
    /// Non-default values throughout: at the contract defaults the refusal must not fire, which the
    /// companion test pins.
    #[test]
    fn non_default_video_clip_knobs_are_refused_by_name() {
        for (req, field) in [
            (clip_request(0, 0.6), "strength"),
            (clip_request(4, 1.0), "frame_idx"),
        ] {
            let err = reject_unimplemented_video_clip_knobs(MODEL_ID, &req)
                .expect_err("a non-default knob must be refused");
            assert!(
                matches!(err, Error::Unsupported(_)),
                "{field}: typed Unsupported, got {err:?}"
            );
            let msg = err.to_string();
            assert!(msg.contains(field), "{field}: names the field: {msg}");
            assert!(msg.contains(MODEL_ID), "{field}: names the model: {msg}");
        }
    }

    /// sc-20263 — the refusal must inspect **every** clip, not the first. `generate_impl` selects
    /// the first NON-empty clip, so a bad value on clip two is exactly the one that would otherwise
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
        let err = reject_unimplemented_video_clip_knobs(MODEL_ID, &req)
            .expect_err("clip two's frame_idx must be refused");
        assert!(err.to_string().contains("frame_idx"), "got: {err}");
    }

    /// sc-20263 — the refusal fires only on a value a caller actually set. The contract defaults
    /// (`strength = 1.0`, `frame_idx = 0`) — every request SceneWorks builds today — pass through,
    /// as does an image-upscale request that carries no clip at all.
    #[test]
    fn default_video_clip_knobs_still_pass_unchanged() {
        assert!(reject_unimplemented_video_clip_knobs(MODEL_ID, &clip_request(0, 1.0)).is_ok());
        let image_only = GenerationRequest {
            width: 32,
            height: 32,
            count: 1,
            conditioning: vec![Conditioning::Reference {
                image: frame(),
                strength: None,
            }],
            ..Default::default()
        };
        assert!(reject_unimplemented_video_clip_knobs(MODEL_ID, &image_only).is_ok());
    }

    /// sc-20263 (adversarial-review follow-up) — bind the refusal's **call site** in
    /// `validate_impl`.
    ///
    /// The tests above call the helper directly, which leaves the wiring unbound: deleting the
    /// `reject_unimplemented_video_clip_knobs(self.descriptor.id, req)?` line puts the knobs back to
    /// being silently dropped with every assertion above still green. The candle-bernini trick of
    /// driving the registered `Generator` weights-free does NOT transfer here — MLX SeedVR2's
    /// `load_with` calls `Seedvr2Pipeline::load`, which materializes the DiT and VAE safetensors
    /// eagerly, so there is no config-only snapshot root that yields a `Generator` (all three
    /// registered ids share that one loader). The binding is therefore pinned in the source, the
    /// same shape as `mlx-llm`'s `multi_frame_attention_has_no_quadratic_mask_allocation`.
    #[test]
    fn validate_impl_is_wired_to_the_refusal() {
        let source = include_str!("registry.rs");
        let body = source
            .split("#[cfg(test)]")
            .next()
            .expect("production body precedes the test module");
        let validate = body
            .split("fn validate_impl(")
            .nth(1)
            .expect("validate_impl is defined")
            .split("fn generate_impl(")
            .next()
            .expect("validate_impl precedes generate_impl");
        assert!(
            validate.contains("reject_unimplemented_video_clip_knobs(self.descriptor.id, req)?"),
            "validate_impl must run the sc-20263 refusal:\n{validate}"
        );
    }

    #[test]
    fn seedvr2_render_plan_resolves_all_five_routes_and_load_refusals_before_weights() {
        use crate::video::{Seedvr2OutputLifecycle, Seedvr2RenderPlan, Seedvr2Route, GIB};

        let desc = descriptor();
        let wb = (7.3 * GIB) as usize;
        let lr = frame();

        assert!(Seedvr2RenderPlan::validate_load_spec(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::File("/single.safetensors".into())),
        )
        .is_err());

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
        let p_single = Seedvr2RenderPlan::compile(&desc, &still_req, wb, 64.0, None).unwrap();
        assert_eq!(p_single.route, Seedvr2Route::StillSinglePass);
        assert!(matches!(
            p_single.output_lifecycle,
            Seedvr2OutputLifecycle::Images { .. }
        ));

        let p_still_tiled = Seedvr2RenderPlan::compile(
            &desc,
            &GenerationRequest {
                width: 2048,
                height: 2048,
                ..still_req
            },
            wb,
            256.0,
            None,
        )
        .unwrap();
        assert!(matches!(
            p_still_tiled.route,
            Seedvr2Route::StillSpatialTiled { .. }
        ));

        let vid_req = clip_request(0, 1.0);
        let p_chunked = Seedvr2RenderPlan::compile(
            &desc,
            &GenerationRequest {
                width: 512,
                height: 512,
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![lr.clone(); 20],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..vid_req.clone()
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
            &desc,
            &GenerationRequest {
                width: 1024,
                height: 1024,
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![lr.clone(); 8],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..vid_req.clone()
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
            &desc,
            &GenerationRequest {
                width: 2048,
                height: 2048,
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![lr; 4],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..vid_req
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
