//! The **`minimax_h3` generator on candle/CUDA** (sc-17156): prompt → real video frames plus a real
//! synchronized stereo soundtrack, `t2va` and `fl2va`.
//!
//! The Windows/Linux sibling of `mlx_gen_minimax_h3::model`. The checkpoint is guidance-distilled —
//! no negative prompt, no guidance scale, one transformer forward per step.
//!
//! # Phases, and why nothing is held resident
//!
//! The three heavy components are **66.71 GB** of Qwen3-VL-32B text encoder, **66.28 GB** of DiT
//! (40.43 GB once the AdaLN projections are evicted) and **10.42 GB** of video VAE. Holding any two
//! of them at once does not fit a card that exists, so [`MiniMaxH3::generate_impl`] builds each
//! phase's component, uses it, drops it and synchronizes the device before the next.
//! [`load`](MiniMaxH3::load) therefore holds **paths**, not tensors: it validates that every
//! partition the render needs is present and defers the reads.
//!
//! # The offload policy is FORCED, and that is what makes the VRAM gate honest
//!
//! [`OffloadPolicy`] is advisory in `gen-core` — "a provider that has not wired it falls back to
//! `Resident`". This provider inverts that: [`OFFLOAD_POLICY`] is `Sequential` **whatever the caller
//! asks for**, and [`MiniMaxH3::offload_policy`] reports what the render will actually do rather
//! than what was requested.
//!
//! That is not a convenience. `candle.vramGbByTier` in SceneWorks' manifest is a *measured*
//! sequential-phase peak, and the worker admits a card by comparing its free VRAM against it. If a
//! caller could ask for `Resident` and get it, the same manifest number would admit a render whose
//! true peak is the **sum** of the three components rather than their max — 143 GB against a
//! published ~40 GB. The number is only truthful because the load is pinned to match it, so the pin
//! is enforced here and [`crate::tests`]' tripwire fails if it stops holding.
//!
//! # Geometry: `frames` is a lattice point, `duration` is a request
//!
//! Two request fields reach the same quantity and are deliberately treated differently:
//!
//! * **`frames`** names a point on the model's own `17n + 5` lattice. An off-lattice value is
//!   **rejected** ([`crate::pipeline::resolve_geometry`]) — SceneWorks normalizes dimensions
//!   upstream, so a gate that silently refits is a gate that can never be observed to fire, and at
//!   video scale a refit is three quarters of a second of picture the caller never asked for.
//! * **`duration`** is a continuous seconds value with no lattice of its own, so it is aligned
//!   **upward** to the next legal frame count ([`crate::pipeline::align_frames_for_duration`]).
//!
//! When both are present `frames` wins, because it is the exact one.
//!
//! # Not here
//!
//! `ref2va` (sc-17157) — the `transformer_ref` partition, the omni-reference presentation and the
//! audio VAE encoder — and the turbo LoRA seam (the candle twin of `mlx_gen_minimax_h3::adapters`).
//! Both are refused rather than silently ignored, but by **two different mechanisms**, and the
//! distinction matters because only one of them is carried by the descriptor:
//!
//! * **`ref2va` is refused by the descriptor.** The three `ref2va` conditioning kinds are absent
//!   from [`descriptor`]'s `conditioning` allowlist, and `Capabilities::validate_request` — which
//!   [`MiniMaxH3::generate_impl`] calls on every render — enforces that allowlist, so a reference
//!   request gets a typed `Unsupported` naming the kind.
//! * **The turbo LoRA seam and the quant tiers are refused by [`load`].** `supports_lora`,
//!   `supports_lokr` and `supported_quants` are *declarations that gen-core reads on no path*: no
//!   validator and no registry code inspects them. They are advertisements to a weights-free
//!   consumer, not gates. The gates are the explicit `spec.adapters` / `spec.quantize` checks in
//!   [`load`]; without them a spec carrying either loaded clean and rendered with the caller's knob
//!   silently discarded. See [`load`] for the full note.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    reject_unknown_components, Capabilities, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, MemoryProviderContract, Modality, ModelDescriptor, OffloadPolicy, Progress,
    SizeFloor, WeightsSource,
};
use candle_gen::gen_core::{ConditioningKind, Image};
use candle_gen::{CandleError, Result};

use crate::audio_config::{AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::denoise::{JointSchedule, AUDIO_SIGMA_SHIFT, MINIMAX_H3_FPS, VIDEO_SIGMA_SHIFT};
use crate::dit::adaln::AdaLnResidency;
use crate::dit::model::{JointDit, MiniMaxH3Dit};
use crate::dit::positions::KeyframeAnchor;
use crate::pipeline::{
    align_frames_for_duration, fit_audio_to_video, fl2va_layout, frames_to_images, initial_latents,
    prepend_condition_rows, render_latents, resolve_geometry, revert_pixel_normalization,
    t2va_layout, RequestGeometry, CANVAS_SHORT_EDGE, PATCH_SIZE, SPATIAL_STRIDE,
};
use crate::text_encoder::{
    lm_prefixes, load_vision_tower, MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer,
    LM_PREFIX, TE_STORE_DTYPE,
};
use crate::vae::MiniMaxH3VideoVae;
use crate::MODEL_ID;

/// Model evaluations a request runs when it names no step count.
///
/// The reference declares no default; 50 is what the sc-17242 spike rendered at and what the model
/// card's own examples use. **Evaluations**, not `num_inference_steps` — that count includes the
/// terminal `σ = 0` the model is never evaluated at, so the schedule is built with `steps + 1`.
pub const DEFAULT_STEPS: u32 = 50;

/// Upper bound on requested steps. A provider-local guard: every step is a full 33 B forward and the
/// AdaLN cache grows with the distinct-timestep count, so an unbounded value is a resource hazard
/// rather than a slow render.
pub const MAX_STEPS: u32 = 200;

/// **The residency this provider runs at, regardless of what a caller asked for.**
///
/// See the module docs: the manifest's measured `vramGbByTier` is a sequential-phase peak, so the
/// number is only truthful if the load cannot be talked out of the staging it was measured under.
pub const OFFLOAD_POLICY: OffloadPolicy = OffloadPolicy::Sequential;

/// The DiT partition a `t2va` / `fl2va` render denoises from. (`transformer_ref` is sc-17157's.)
pub const BASE_DIT_PARTITION: &str = "transformer";

/// The component directories a render reads, in phase order.
const REQUIRED_COMPONENT_DIRS: [&str; 4] = ["text_encoder", BASE_DIT_PARTITION, "vae", "audio_vae"];

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "minimax_h3",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch exists anywhere in the checkpoint.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // **One kind covers all three keyframe shapes.** `Keyframe` carries a `frame_idx`, and
            // first-only / last-only / first+last are one, one and two of them.
            //
            // The `ref2va` kinds the MLX sibling advertises (`Reference`, `ReferenceVideo`,
            // `ReferenceAudio`) are deliberately ABSENT until sc-17157 lands the
            // `transformer_ref` path here: default-deny turns a reference request into a typed
            // `Unsupported` instead of a render that silently ignored it.
            conditioning: vec![ConditioningKind::Keyframe],
            // The candle turbo-LoRA seam is not ported (the MLX sibling's `adapters`), and
            // advertising it would accept a file this lane cannot fold.
            supports_lora: false,
            supports_lokr: false,
            // The pre-quantized tiers are packed offline by the MLX lane's `convert`; this lane has
            // no tier loader of its own yet, so it advertises none rather than accepting a
            // `spec.quantize` it would ignore.
            supported_quants: &[],
            max_count: 1,
            component_precision_floors: &[],
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: SPATIAL_STRIDE,
            max_size: 1344,
            // The stride is advertised, not hidden in provider code, so a weights-free consumer can
            // predict that an unaligned canvas is REFUSED rather than quietly refit.
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SPATIAL_STRIDE,
            },
            ..Default::default()
        },
    }
}

/// The loaded `minimax_h3` provider — **paths only**.
///
/// Nothing heavy is resident between renders, and nothing heavy is resident between *phases* of one
/// render either. That is the whole memory design, so the struct that carries it holds no tensors to
/// make the property structural rather than a habit.
pub struct MiniMaxH3 {
    descriptor: ModelDescriptor,
    root: PathBuf,
    dit_dir: PathBuf,
    device: Device,
    dtype: DType,
    offload: OffloadPolicy,
    contract: MemoryProviderContract,
}

impl MiniMaxH3 {
    /// Validate the snapshot and record the paths a render will read.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        reject_unknown_components(spec, &[], MODEL_ID)?;
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(p) => p
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| p.clone()),
        };
        for component in REQUIRED_COMPONENT_DIRS {
            let dir = root.join(component);
            if !dir.is_dir() {
                return Err(CandleError::Msg(format!(
                    "{MODEL_ID}: the snapshot at {} has no `{component}/` component — a render \
                     reads all of {REQUIRED_COMPONENT_DIRS:?}",
                    root.display()
                )));
            }
        }
        let device = candle_gen::default_device()?;
        Ok(Self {
            descriptor: descriptor(),
            dit_dir: root.join(BASE_DIT_PARTITION),
            root,
            device,
            // The DiT ships mixed f32/bf16 top-level tensors; bf16 is the block store, matching the
            // published checkpoint rather than widening it.
            dtype: DType::BF16,
            // **Not `spec.offload_policy`.** See [`OFFLOAD_POLICY`] and the module docs.
            offload: OFFLOAD_POLICY,
            contract: crate::memory_strategy::contract_for(spec)?,
        })
    }

    /// The residency this provider will actually run at.
    ///
    /// Reported rather than echoed: a caller reading this back gets the enforced policy, so
    /// "requested Resident, got Sequential" is observable instead of silent.
    pub fn offload_policy(&self) -> OffloadPolicy {
        self.offload
    }

    /// The snapshot root a render reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The DiT partition directory this provider denoises from.
    pub fn dit_dir(&self) -> &Path {
        &self.dit_dir
    }

    /// The device every phase materializes on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resolve the request's geometry: `frames` wins over `duration`, and the canvas falls back to
    /// the checkpoint's own 16:9 default when the caller names none.
    fn request_geometry(&self, req: &GenerationRequest) -> Result<RequestGeometry> {
        let frames = match (req.frames, req.duration) {
            (Some(f), _) => f as usize,
            (None, Some(seconds)) => align_frames_for_duration(seconds)?,
            (None, None) => align_frames_for_duration(MIN_DEFAULT_SECONDS)?,
        };
        let (width, height) = if req.width == 0 || req.height == 0 {
            (
                CANVAS_SHORT_EDGE * 16 / 9 / SPATIAL_STRIDE * SPATIAL_STRIDE,
                CANVAS_SHORT_EDGE,
            )
        } else {
            (req.width, req.height)
        };
        resolve_geometry(width, height, frames)
    }

    /// **Phase 1 (t2va): encode the prompt and release the 66.7 GB conditioner.**
    ///
    /// The prefix trim is what keeps this to the 50 layers the tap runs: `lm_prefixes` names
    /// `embed_tokens` and layers `0..50`, so the header-only mmap never materializes layers 50-63,
    /// the final norm or `lm_head`.
    fn encode_prompt(&self, prompt: &str) -> Result<candle_gen::candle_core::Tensor> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) = tok.encode_prompt(prompt, &self.device)?;
        let cfg = MiniMaxH3TeConfig::from_snapshot(&self.root)?;
        let shards = candle_gen::loader::sorted_safetensors(
            &self.root.join("text_encoder"),
            "minimax-h3 te",
        )?;
        let prefixes = lm_prefixes(LM_PREFIX, &cfg);
        let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
        let w =
            candle_gen::Weights::from_files_filtered(&shards, &self.device, TE_STORE_DTYPE, &refs)?;
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_STORE_DTYPE)?;
        drop(w);
        let context = te.forward(&ids, &mask)?;
        // candle is eager, so `context` is already materialized here and dropping the encoder
        // genuinely frees its weights — there is no lazy graph holding them alive.
        drop(te);
        crate::dit::release_device_memory(&self.device)?;
        Ok(context)
    }

    /// **Phase 1 (fl2va): the vision-grounded presentation.**
    ///
    /// Runs the Qwen3-VL vision tower over the fitted keyframes, splices their embeddings into the
    /// `"<Picture i>: "` presentation, and returns the context together with its **per-row modality
    /// tags** — a vision block's rows are tagged *video* and address a different block of the AdaLN
    /// modulation table than the text rows around them.
    ///
    /// This is one of the two paths a keyframe takes. The other is the VAE encode in
    /// [`crate::conditioning`]; the reference runs **both**, and neither substitutes for the other —
    /// the tower supplies semantic context to the prompt stream, the VAE supplies the pixel-space
    /// anchor the video rows are conditioned on.
    fn encode_prompt_grounded(
        &self,
        prompt: &str,
        keyframes: &[&Image],
    ) -> Result<(candle_gen::candle_core::Tensor, Vec<u32>)> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let cfg = MiniMaxH3TeConfig::from_snapshot(&self.root)?;

        let vision = load_vision_tower(&self.root, &self.device)?;
        let grounded = crate::text_encoder::run_vision(&vision, keyframes, &self.device)?;
        drop(vision);
        crate::dit::release_device_memory(&self.device)?;

        let (ids, mask, tags) = tok.encode_fl2va(prompt, &grounded.counts, &self.device)?;
        let shards = candle_gen::loader::sorted_safetensors(
            &self.root.join("text_encoder"),
            "minimax-h3 te",
        )?;
        let prefixes = lm_prefixes(LM_PREFIX, &cfg);
        let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
        let w =
            candle_gen::Weights::from_files_filtered(&shards, &self.device, TE_STORE_DTYPE, &refs)?;
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_STORE_DTYPE)?;
        drop(w);
        let context = te.forward_with_images(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        drop(te);
        drop(grounded);
        crate::dit::release_device_memory(&self.device)?;

        // The tags describe the presentation row for row; a mismatch would mis-tag every row after
        // the divergence and is silent, because both lengths build a runnable sequence.
        if context.dims()[1] != tags.len() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: the grounded context has {} rows but {} modality tags",
                context.dims()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Generator::validate(self, req)?;
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let geometry = self.request_geometry(req)?;
        let evaluations = req.steps.unwrap_or(DEFAULT_STEPS);
        if evaluations > MAX_STEPS {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: {evaluations} steps exceeds the provider bound of {MAX_STEPS}"
            )));
        }
        // `num_inference_steps` counts the terminal σ = 0, at which the model is never evaluated.
        //
        // **The video shift is a per-request knob.** The base checkpoint's published 12.0 is the
        // default, but the distilled turbo variants are trained against their own shift and are
        // simply wrong on 12.0. Only the **video** shift is overridable, because only the video
        // shift moves across the published set: every documented variant keeps audio at 3.0.
        let video_shift = req.scheduler_shift.unwrap_or(VIDEO_SIGMA_SHIFT);
        let schedule =
            JointSchedule::with_shifts(evaluations as usize + 1, video_shift, AUDIO_SIGMA_SHIFT)?;
        let seed = req.seed.unwrap_or(0);

        // --- 1. conditioning -----------------------------------------------------------------
        let keyframes = req.keyframes();
        let anchors = keyframe_anchors(&keyframes, geometry.joint.num_frames)?;
        // Fitted ONCE, here, and shared by both keyframe paths. The vision tower and the VAE encode
        // must see the same pixels — resizing separately per path would let a resampling difference
        // put the two conditioning signals fractionally out of register, which is exactly the class
        // of divergence nothing downstream can detect.
        let fitted = if anchors.is_empty() {
            Vec::new()
        } else {
            let images: Vec<&Image> = keyframes.iter().map(|k| k.image).collect();
            crate::keyframe::fit_keyframes(&images, geometry.width, geometry.height)?
        };

        let (context, text_tags) = if anchors.is_empty() {
            let context = self.encode_prompt(&req.prompt)?;
            let tags = vec![crate::denoise::TEXT_TAG; context.dims()[1]];
            (context, tags)
        } else {
            let refs: Vec<&Image> = fitted.iter().collect();
            self.encode_prompt_grounded(&req.prompt, &refs)?
        };
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 1b. keyframe conditioning latents (fl2va) ----------------------------------------
        // The VAE encode, the second of the keyframe's two paths. Done before the DiT is mapped so
        // the ~10 GB VAE and the ~66 GB DiT are never both resident.
        let condition_rows = if anchors.is_empty() {
            None
        } else {
            let pixels = fitted
                .iter()
                .map(|f| crate::keyframe::keyframe_to_vae_pixels(f, &self.device))
                .collect::<Result<Vec<_>>>()?;
            let vae = MiniMaxH3VideoVae::load(&self.root, &self.device, self.dtype)?;
            let rows = crate::conditioning::build_condition_rows(
                &vae,
                &pixels,
                &anchors,
                PATCH_SIZE,
                &crate::conditioning::KeyframeNoise::Seeded,
                &self.device,
            )?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            rows
        };
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 2. denoise -----------------------------------------------------------------------
        let dit = MiniMaxH3Dit::load(&self.root, BASE_DIT_PARTITION, &self.device, self.dtype)?;
        let patch = dit.config().patch_size;
        let layout = if anchors.is_empty() {
            t2va_layout(&geometry, context.dims()[1], patch, &self.device)?
        } else {
            fl2va_layout(&geometry, &text_tags, &anchors, patch, &self.device)?
        };
        let (video_rows, audio_rows) = initial_latents(&geometry, patch, seed, &self.device)?;
        // The anchors LEAD the video row stream; the scheduler then writes only the tail.
        let video_rows = prepend_condition_rows(&layout, condition_rows.as_ref(), &video_rows)?;
        let adaln = crate::denoise::adaln_schedule(&schedule)?;
        // The 26.02 GB lever: project the whole schedule's modulation and release `adaln_proj`.
        // Every timestep this run evaluates at is enumerated in `adaln`, so the eviction is safe.
        let mut model = JointDit::new(
            dit,
            layout.clone(),
            &context,
            adaln,
            AdaLnResidency::PrecomputeAndEvict,
        )?;
        drop(context);

        let total = schedule.num_evals() as u32;
        let mut on_step = |completed: usize| {
            on_progress(Progress::Step {
                current: completed as u32,
                total,
            });
        };
        let rendered = render_latents(
            &mut model,
            &layout,
            &schedule,
            &video_rows,
            &audio_rows,
            patch,
            &self.device,
            &req.cancel,
            &mut on_step,
        )?;
        drop(model);
        drop(video_rows);
        drop(audio_rows);
        crate::dit::release_device_memory(&self.device)?;

        // --- 3. decode ------------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let frames = {
            let vae = MiniMaxH3VideoVae::load_decode_only(&self.root, &self.device, self.dtype)?;
            let video = vae.decode(&rendered.video)?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            frames_to_images(&revert_pixel_normalization(&video)?)?
        };
        let audio = {
            let (cfg, w) = self.load_audio_vae()?;
            let vae = crate::audio_vae::MiniMaxH3AudioVae::from_weights(
                &w,
                &cfg,
                &self.device,
                self.dtype,
            )?;
            drop(w);
            let track = vae.decode_audio_track(&rendered.audio)?;
            drop(vae);
            crate::dit::release_device_memory(&self.device)?;
            track
        };
        let audio = fit_audio_to_video(audio, &geometry)?;
        if audio.sample_rate != AUDIO_SAMPLE_RATE || audio.channels != AUDIO_OUTPUT_CHANNELS {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: the decoded soundtrack is {} Hz / {} channels, expected \
                 {AUDIO_SAMPLE_RATE} / {AUDIO_OUTPUT_CHANNELS}",
                audio.sample_rate, audio.channels
            )));
        }
        if frames.len() != geometry.joint.num_frames {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
    }
}

impl MiniMaxH3 {
    /// The audio VAE's config triple plus its weight map.
    ///
    /// The config comes from the FL2VA **source triple** (`config.json` + `config.yaml` +
    /// `metadata.json`) when the snapshot ships it, because that is where the reference's
    /// constructor kwargs actually live; the diffusers-repackaged `audio_vae/config.json` alone is
    /// not sufficient (see `crate::audio_config`). A snapshot without the triple falls back to the
    /// declared defaults, which those documents are asserted to reproduce exactly.
    fn load_audio_vae(
        &self,
    ) -> Result<(
        crate::audio_config::MiniMaxH3AudioVaeConfig,
        candle_gen::Weights,
    )> {
        let src = self.root.join("FL2VA").join("audio_vae");
        let cfg = if src.join("config.yaml").is_file() {
            let read = |name: &str| -> Result<String> {
                std::fs::read_to_string(src.join(name)).map_err(|e| {
                    CandleError::Msg(format!(
                        "{MODEL_ID}: read {}: {e}",
                        src.join(name).display()
                    ))
                })
            };
            crate::audio_config::MiniMaxH3AudioVaeConfig::from_source_files(
                &read("config.json")?,
                &read("config.yaml")?,
                &read("metadata.json")?,
            )?
        } else {
            crate::audio_config::MiniMaxH3AudioVaeConfig::default()
        };
        let shards = candle_gen::loader::sorted_safetensors(
            &self.root.join("audio_vae"),
            "minimax-h3 audio vae",
        )?;
        let w = candle_gen::Weights::from_files(&shards, &self.device, self.dtype)?;
        Ok((cfg, w))
    }
}

/// Duration a request that names neither `frames` nor `duration` renders at — the lattice floor.
const MIN_DEFAULT_SECONDS: f32 = 5.1667;

/// Which end of the clip each keyframe anchors, from the request's `frame_idx` values.
///
/// `fl2va` has exactly two slots — the first frame and the last — and the anchors are **positional**
/// against the fitted keyframe list, so a request that anchors the same end twice, or names a frame
/// index that is neither end, is a typed error rather than a render that quietly moved it.
pub fn keyframe_anchors(
    keyframes: &[candle_gen::gen_core::KeyframeRef<'_>],
    num_frames: usize,
) -> Result<Vec<KeyframeAnchor>> {
    let last = num_frames.saturating_sub(1);
    let mut out = Vec::with_capacity(keyframes.len());
    for k in keyframes {
        let idx = k.frame_idx.max(0) as usize;
        let anchor = if idx == 0 {
            KeyframeAnchor::First
        } else if idx == last {
            KeyframeAnchor::Last
        } else {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: a keyframe anchors frame {idx}, but fl2va has exactly two slots — the \
                 FIRST frame (0) and the LAST ({last}). An interior keyframe has no conditioning \
                 row to occupy"
            )));
        };
        out.push(anchor);
    }
    // `[First, First]` is arity-legal and semantically impossible; `crate::conditioning` rejects it
    // too, but catching it here names the request field the caller got wrong.
    if out.len() == 2 && out[0] == out[1] {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: both keyframes anchor {:?}; fl2va's two slots are the FIRST and the LAST \
             frame",
            out[0]
        )));
    }
    Ok(out)
}

impl Generator for MiniMaxH3 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.trim().is_empty() {
            return Err(candle_gen::gen_core::Error::Msg(format!(
                "{MODEL_ID}: prompt must not be empty"
            )));
        }
        Ok(())
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.contract)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<GenerationOutput> {
        Ok(self.generate_impl(req, on_progress)?)
    }
}

/// Registry entry point.
///
/// # The two load-time refusals, and why the descriptor flags cannot stand in for them
///
/// [`descriptor`] sets `supports_lora` / `supports_lokr` to `false` and `supported_quants` to `&[]`,
/// and the module docs above promise a request carrying either "gets a typed `Unsupported` at the
/// contract boundary instead of a plausible render that quietly dropped it". **Those three fields are
/// declarations that gen-core reads on no path.** `Capabilities::validate_request` gates the
/// conditioning allowlist (which is what actually default-denies the `ref2va` kinds) but never looks
/// at `supports_lora` or `supported_quants`, and `ProviderRegistryBuilder` does not inspect a
/// `LoadSpec` either — so without the checks below, a spec carrying `adapters` or `quantize` loaded
/// clean and rendered with the caller's knob silently discarded. That is the precise failure the
/// docs claimed was impossible.
///
/// `reject_unknown_components` does not cover this: it validates `spec.components`, a different field
/// from `spec.adapters` and `spec.quantize`.
///
/// This is the `candle-gen-mochi` idiom — refuse at the registry entry point, which returns
/// `gen_core::Result` and can therefore carry the contract-load-bearing
/// [`gen_core::Error::Unsupported`] that `CandleError` has no variant for.
pub fn load(spec: &LoadSpec) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    if !spec.adapters.is_empty() {
        return Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: the candle lane does not support LoRA/LoKr adapters — {} staged, and the \
             turbo seam (the MLX sibling's `adapters`) is not ported here. Refused rather than \
             rendered without them.",
            spec.adapters.len()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: spec.quantize={q:?} is not supported on the candle lane — the pre-quantized \
             tiers are packed offline by the MLX lane's `convert` and this lane has no tier loader, \
             so a tier request is refused rather than silently rendered dense."
        )));
    }
    Ok(Box::new(MiniMaxH3::load(spec)?))
}

candle_gen::register_generators! {
    pub const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The offload policy is forced, not echoed** — the tripwire the measured `vramGbByTier`
    /// depends on.
    ///
    /// `OffloadPolicy` is advisory in gen-core, so the plausible regression is a `load` that reads
    /// `spec.offload_policy` "like every other provider does". That would leave the manifest number
    /// admitting a card against a sequential peak while the render ran resident — a ~3.5x
    /// under-estimate that fails as an OOM in the middle of a 20-minute render, not at admission.
    ///
    /// Asserting `Sequential` alone would be a false green (it is also the value a `Sequential`
    /// request would echo), so the load below asks for **`Resident`** and the assertion is that it
    /// did not get it.
    #[test]
    fn the_offload_policy_is_forced_sequential_even_when_resident_is_requested() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for c in REQUIRED_COMPONENT_DIRS {
            std::fs::create_dir_all(root.join(c)).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_offload_policy(OffloadPolicy::Resident);
        assert_eq!(spec.offload_policy, OffloadPolicy::Resident);

        let model = MiniMaxH3::load(&spec).unwrap();
        assert_eq!(
            model.offload_policy(),
            OffloadPolicy::Sequential,
            "a Resident request must NOT be honored: the manifest's measured vramGbByTier is a \
             sequential-phase peak"
        );
        assert_eq!(OFFLOAD_POLICY, OffloadPolicy::Sequential);
        assert_ne!(model.offload_policy(), spec.offload_policy);
    }

    /// A staged snapshot root with all four component dirs — enough for `load` to succeed.
    fn staged_root(tmp: &tempfile::TempDir) -> PathBuf {
        let root = tmp.path().to_path_buf();
        for c in REQUIRED_COMPONENT_DIRS {
            std::fs::create_dir_all(root.join(c)).unwrap();
        }
        root
    }

    /// **A staged LoRA is REFUSED, not dropped.**
    ///
    /// `supports_lora: false` is a declaration gen-core reads on no path — neither
    /// `validate_request` nor the registry builder inspects it — so before this guard a spec
    /// carrying `adapters` loaded clean and rendered with the adapter silently discarded, which is
    /// exactly what the module docs promised could not happen.
    ///
    /// The control is the point: the SAME root with no adapters must still load, so this proves the
    /// adapter is what refuses rather than the fixture being broken.
    #[test]
    fn a_staged_lora_is_refused_rather_than_silently_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);

        // Control: without adapters the very same spec loads.
        load(&LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("the bare spec must load — otherwise the refusal below proves nothing");

        let spec = LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![
            candle_gen::gen_core::AdapterSpec {
                path: PathBuf::from("/turbo.safetensors"),
                scale: 1.0,
                kind: candle_gen::gen_core::AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            },
        ]);
        // `Box<dyn Generator>` is not `Debug`, so `expect_err` is unavailable.
        let err = match load(&spec) {
            Ok(_) => panic!("a staged LoRA must be refused, not silently dropped"),
            Err(e) => e,
        };
        assert!(
            matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
            "must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("LoRA/LoKr"), "{msg}");
    }

    /// **A `ref2va` request is REFUSED on the RENDER PATH, not merely absent from the descriptor.**
    ///
    /// `the_descriptor_advertises_only_the_ported_surface` asserts the three reference kinds are
    /// missing from the allowlist — but a missing entry is a DECLARATION, and this crate has already
    /// been bitten once by assuming a declaration gates anything (`supports_lora` gates nothing at
    /// all, which is why [`load`] carries an explicit check). This exercises the actual seam:
    /// `Generator::validate`, which `generate_impl` calls on every render, must turn a reference
    /// request into a typed `Unsupported` that NAMES the kind.
    #[test]
    fn a_ref2va_request_is_refused_on_the_render_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let generator = load(&LoadSpec::new(WeightsSource::Dir(root))).unwrap();

        let base = GenerationRequest {
            prompt: "a cellist on a rooftop".into(),
            width: 1344,
            height: 768,
            ..Default::default()
        };
        // Control: the same request without reference conditioning passes validation, so the
        // refusal below is attributable to the conditioning and not to some other floor check.
        generator
            .validate(&base)
            .expect("the bare request must validate — otherwise the refusal proves nothing");

        for kind in [
            candle_gen::gen_core::Conditioning::Reference {
                image: Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                },
                strength: Some(1.0),
            },
            candle_gen::gen_core::Conditioning::ReferenceVideo {
                frames: vec![Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                }],
                fps: 24.0,
                audio: None,
            },
        ] {
            let named = format!("{:?}", kind.kind());
            let req = GenerationRequest {
                conditioning: vec![kind],
                ..base.clone()
            };
            let err = generator
                .validate(&req)
                .expect_err("a ref2va request must be refused");
            assert!(
                matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
                "{named}: must be a typed Unsupported, got {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(&named),
                "the refusal must NAME the kind so a caller knows what to drop: {msg}"
            );
        }
    }

    /// **A tier request is REFUSED, not silently rendered dense.**
    ///
    /// `supported_quants: &[]` is likewise enforced nowhere in gen-core. Mutating this guard away
    /// makes a `q4` request load the dense bf16 checkpoint and return a render the caller believes
    /// is quantized.
    #[test]
    fn a_quantize_request_is_refused_rather_than_silently_rendered_dense() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);

        load(&LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("the bare spec must load — otherwise the refusal below proves nothing");

        let spec =
            LoadSpec::new(WeightsSource::Dir(root)).with_quant(candle_gen::gen_core::Quant::Q4);
        let err = match load(&spec) {
            Ok(_) => panic!("a tier request must be refused, not silently rendered dense"),
            Err(e) => e,
        };
        assert!(
            matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
            "must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("no tier loader"), "{msg}");
    }

    /// A snapshot missing any one of the four component directories fails at `load`, naming the
    /// component — not twenty minutes later inside a phase.
    #[test]
    fn a_missing_component_fails_the_load_and_names_itself() {
        for missing in REQUIRED_COMPONENT_DIRS {
            let tmp = tempfile::tempdir().unwrap();
            let root = tmp.path();
            for c in REQUIRED_COMPONENT_DIRS.iter().filter(|c| **c != missing) {
                std::fs::create_dir_all(root.join(c)).unwrap();
            }
            let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
            let e = match MiniMaxH3::load(&spec) {
                Ok(_) => panic!("a snapshot without `{missing}/` must not load"),
                Err(e) => e.to_string(),
            };
            assert!(e.contains(missing), "{missing}: {e}");
        }
    }

    /// The descriptor advertises what this lane can actually do — and, just as load-bearing, does
    /// **not** advertise what it cannot.
    #[test]
    fn the_descriptor_advertises_only_the_ported_surface() {
        let d = descriptor();
        assert_eq!(d.id, "minimax_h3");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        // Guidance-distilled.
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_true_cfg);
        // fl2va only; ref2va (sc-17157) is default-denied rather than silently ignored.
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Keyframe]
        );
        assert!(!d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(!d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::ReferenceVideo));
        assert!(!d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::ReferenceAudio));
        // The turbo LoRA seam is not ported here.
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_lokr);
        assert!(d.capabilities.supported_quants.is_empty());
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 32 }
        );
    }

    /// Keyframe anchors come from `frame_idx` and only the two ends exist. An interior index is a
    /// typed error naming the field, not a keyframe quietly moved to frame 0.
    #[test]
    fn keyframe_anchors_admit_only_the_two_ends() {
        let img = Image {
            width: 32,
            height: 32,
            pixels: vec![0; 32 * 32 * 3],
        };
        let kf = |idx: i32| candle_gen::gen_core::KeyframeRef {
            image: &img,
            frame_idx: idx,
            strength: 1.0,
        };
        assert_eq!(keyframe_anchors(&[], 124).unwrap(), Vec::new());
        assert_eq!(
            keyframe_anchors(&[kf(0)], 124).unwrap(),
            vec![KeyframeAnchor::First]
        );
        // A last-frame-only request is a payload shape, not a mode of its own.
        assert_eq!(
            keyframe_anchors(&[kf(123)], 124).unwrap(),
            vec![KeyframeAnchor::Last]
        );
        assert_eq!(
            keyframe_anchors(&[kf(0), kf(123)], 124).unwrap(),
            vec![KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        // An interior frame has no conditioning row to occupy.
        let e = keyframe_anchors(&[kf(60)], 124).unwrap_err().to_string();
        assert!(e.contains("two slots"), "{e}");
        // The same end twice is arity-legal and semantically impossible.
        let e = keyframe_anchors(&[kf(0), kf(0)], 124)
            .unwrap_err()
            .to_string();
        assert!(e.contains("both keyframes anchor"), "{e}");
    }

    /// The step bound is a real gate, not a comment: every step is a full 33 B forward.
    #[test]
    fn the_step_bound_is_enforced() {
        assert_eq!(DEFAULT_STEPS, 50);
        assert_eq!(MAX_STEPS, 200);
        const { assert!(DEFAULT_STEPS < MAX_STEPS) };
    }
}
