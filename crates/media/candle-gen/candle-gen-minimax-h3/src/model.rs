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
//! of them at once does not fit a card that exists, so `MiniMaxH3::generate_impl` builds each
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
//! is enforced here and **two** guards in this module's `tests` watch it, because the policy field
//! and the staging it stands for are separate facts:
//!
//! * `the_offload_policy_is_forced_sequential_even_when_resident_is_requested` pins the reported
//!   policy — a `load` that started echoing `spec.offload_policy` reds.
//! * `every_heavy_component_is_released_before_the_next_one_is_mapped` pins the **staging itself**.
//!   `self.offload` is read by no render path, so the first guard alone stays green if every
//!   `drop` / `release_device_memory` below is deleted. The second one reads this file's own
//!   production source and reds on the deletion of any single one of them. See its doc comment for
//!   what a source scan can and cannot see.
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
//! audio VAE encoder. It is refused **by the descriptor**: the three `ref2va` conditioning kinds are
//! absent from [`descriptor`]'s `conditioning` allowlist, and `Capabilities::validate_request` —
//! which `MiniMaxH3::generate_impl` calls on every render — enforces that allowlist, so a reference
//! request gets a typed `Unsupported` naming the kind.
//!
//! # The turbo LoRA seam IS here (sc-18728), and the declaration still gates nothing
//!
//! [`crate::adapters`] is the candle twin of `mlx_gen_minimax_h3::adapters`, installed through the
//! single [`MiniMaxH3::load_task_dit`] seam. `supports_lora` is now `true` — but it is worth keeping
//! the original finding on the record, because it is what shapes the code around it:
//!
//! **`supports_lora`, `supports_lokr` and `supported_quants` are declarations gen-core reads on no
//! path.** No validator and no registry code inspects them. They are advertisements to a
//! weights-free consumer, not gates. So flipping `supports_lora` to `true` grants nothing and
//! flipping it to `false` would forbid nothing; the behaviour is entirely in
//! [`crate::adapters::apply_minimax_h3_adapters`], which is strict about unmatched targets and
//! about a file that folded nothing. Symmetrically, `supports_lokr: false` is enforced twice in
//! real code — by kind in [`load`] and by file content in the installer — precisely because the
//! flag itself cannot refuse anything.
//!
//! The quant tiers and every unread `LoadSpec` slot are still refused by [`load`]: the explicit
//! `spec.quantize` check plus `reject_unread_slots` for the seven slots this provider does not
//! read. Without them a spec carrying any of those loaded clean and rendered with the caller's knob
//! silently discarded.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    reject_unknown_components, Capabilities, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, MemoryProviderContract, Modality, ModelDescriptor, OffloadPolicy, Precision,
    Progress, SizeFloor, WeightsSource,
};
use candle_gen::gen_core::{AdapterSpec, ConditioningKind, Image};
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
            // **The turbo-LoRA seam is ported** (sc-18728): `crate::adapters` is the candle twin
            // of `mlx_gen_minimax_h3::adapters`, installed on the job-local DiT in
            // `generate_impl`. This bit is a declaration gen-core reads on no path, so it is an
            // advertisement to a weights-free consumer; the enforcement is
            // `crate::adapters::apply_minimax_h3_adapters`, which is strict about both unmatched
            // targets and a file that matched nothing.
            supports_lora: true,
            // **LoKr stays false, and here that is enforced twice.** This DiT's adapter seam
            // installs `scale·((x·A)·B)` residuals only; a Kronecker delta is a different
            // operation, so `load` refuses `AdapterKind::Lokr` and the installer refuses a file
            // carrying `lokr_*` factors regardless of how it was declared.
            supports_lokr: false,
            // The pre-quantized tiers are packed offline by the MLX lane's `convert`; this lane has
            // no tier loader of its own yet, so it advertises none rather than accepting a
            // `spec.quantize` it would ignore.
            supported_quants: &[],
            // **The discovery signal, and it is true here.** gen-core makes
            // `OffloadPolicy::Sequential` advisory, so an unwired engine silently stays resident and
            // the fallback is invisible from outside; this bit is what a consumer reads to know
            // which it got. `candle-gen-krea` states the rule as a lockstep contract — "never flip
            // this on ahead of the wiring", because advertising a lane that would really run
            // resident makes the fit-gate under-predict its peak and admit a job that then OOMs.
            //
            // The wiring is here and is *stronger* than honoring: `generate_impl` releases every
            // heavy component and synchronizes the device before mapping the next, and
            // `OFFLOAD_POLICY` forces `Sequential` so a caller cannot even opt out. Leaving this
            // `false` was the divergence worth closing — SceneWorks' manifest already declares
            // `supportsSequentialOffload: true`, and a `false` bit here would tell the worker to
            // size this render at the ~143 GB SUM of the components rather than their max.
            //
            // Note this is a *separate* question from the memory-ladder rung-1 declaration in
            // `crate::memory_strategy`, which stays `Missing` pending its behavior seam (sc-18660).
            // That declaration gates ladder ADMISSION; this bit describes the residency lifecycle.
            supports_sequential_offload: true,
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
    /// The staged adapter specs, held as **paths and strengths** rather than tensors — the same
    /// discipline the rest of this struct keeps. They are read once per render, in
    /// [`MiniMaxH3::load_task_dit`], onto the job-local DiT.
    adapters: Vec<AdapterSpec>,
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
            adapters: spec.adapters.clone(),
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

    /// The adapter specs this provider was loaded with.
    pub fn adapters(&self) -> &[AdapterSpec] {
        &self.adapters
    }

    /// **The single render seam that maps a DiT partition and folds the staged adapters onto it**
    /// (sc-18728).
    ///
    /// One function rather than an install at each call site, so a future `ref2va` denoise path
    /// (sc-17157) cannot acquire a DiT that skipped the adapters — the MLX sibling landed the same
    /// single seam for exactly that reason. The DiT is **job-local**: residuals are pushed onto this
    /// render's own copy, never onto a shared resident base, so two concurrent renders at different
    /// strengths cannot see each other's fold.
    ///
    /// The install is strict — an unmatched target, or a file that folded nothing, is an error
    /// rather than a partial render — and it runs **before** `JointDit` precomputes the AdaLN
    /// tables. That ordering is required rather than incidental: `adaln_proj` is deliberately
    /// unreachable to adapters ([`crate::dit::block::DitBlock::adaptable_mut`]), so the precompute
    /// must observe the same projection whether or not a LoRA is staged.
    /// **`pub` on purpose.** The property "a staged adapter reaches the DiT this render denoises
    /// with" cannot be asserted from outside the crate unless this seam is callable, and a source
    /// scan asserting the call site *looks* right is not evidence that it folds anything —
    /// `tests/turbo_lora.rs::the_render_seam_folds_the_staged_adapter_onto_the_dit` calls this and
    /// reads the residual back off the returned model.
    pub fn load_task_dit(&self, partition: &str) -> Result<MiniMaxH3Dit> {
        let mut dit = MiniMaxH3Dit::load(&self.root, partition, &self.device, self.dtype)?;
        if !self.adapters.is_empty() {
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.adapters)?;
        }
        Ok(dit)
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
        let dit = self.load_task_dit(BASE_DIT_PARTITION)?;
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
/// # The load-time refusals, and why the descriptor flags cannot stand in for them
///
/// [`descriptor`] sets `supports_lokr` to `false` and `supported_quants` to `&[]`. **Those fields
/// are declarations that gen-core reads on no path.** `Capabilities::validate_request` gates the
/// conditioning allowlist (which is what actually default-denies the `ref2va` kinds) but never looks
/// at `supports_lora` or `supported_quants`, and `ProviderRegistryBuilder` does not inspect a
/// `LoadSpec` either — so without the checks below, a spec carrying `quantize`, a LoKr adapter, or a
/// per-pass / MoE adapter knob loaded clean and rendered with the caller's knob silently discarded.
///
/// `reject_unknown_components` does not cover this: it validates `spec.components`, a different
/// field from `spec.adapters` and `spec.quantize`.
///
/// Since sc-18728 `spec.adapters` is **read** rather than refused — see
/// [`MiniMaxH3::load_task_dit`] — so the checks here narrowed to the three adapter-shaped knobs this
/// lane still cannot honor: `AdapterKind::Lokr`, `pass_scales` (LTX's two-stage schedule) and
/// `moe_expert` (Wan's dual-expert selector). Each is a field a caller can set on an
/// otherwise-valid LoRA, and each would otherwise be dropped without a word.
///
/// This is the `candle-gen-mochi` idiom — refuse at the registry entry point, which returns
/// `gen_core::Result` and can therefore carry the contract-load-bearing
/// [`candle_gen::gen_core::Error::Unsupported`] that `CandleError` has no variant for.
///
/// # Every other `LoadSpec` slot is decided too
///
/// The checks below were the ones this crate shipped first, but "declared and unenforced" is a
/// *class*, not two instances. `reject_unread_slots` closes the rest of it, and the three slots
/// that are **not** refused are each accounted for there rather than left to inference.
pub fn load(spec: &LoadSpec) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    for a in &spec.adapters {
        // **LoKr is still refused**, by kind here and by file content in
        // [`crate::adapters::apply_minimax_h3_adapters`]. This DiT's seam installs
        // `scale·((x·A)·B)` residuals; a Kronecker delta is a different operation, so accepting one
        // would be a wrong fold rather than a weak one. Refused at the registry entry point because
        // that is the boundary carrying `Error::Unsupported`.
        if a.kind == candle_gen::gen_core::AdapterKind::Lokr {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: {} is staged as LoKr; the candle lane installs LoRA residuals only. \
                 Use a LoRA export of the same adapter.",
                a.path.display()
            )));
        }
        // `pass_scales` is LTX's per-distilled-stage knob and `moe_expert` is Wan's dual-expert
        // selector. This checkpoint has one denoise pass and one expert, so either is a knob that
        // would be silently discarded — the exact class `reject_unread_slots` exists to close, one
        // field deeper.
        if a.pass_scales.is_some() {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} sets `pass_scales`, which only LTX's two-stage denoise \
                 reads. MiniMax-H3 runs ONE forward per step, so a per-pass schedule has nowhere \
                 to apply and would be silently dropped.",
                a.path.display()
            )));
        }
        if a.moe_expert.is_some() {
            return Err(candle_gen::gen_core::Error::Unsupported(format!(
                "{MODEL_ID}: adapter {} names a MoE expert, which only the Wan2.2 A14B dual-expert \
                 denoiser reads. This checkpoint is single-stream, so the selection would be \
                 silently dropped.",
                a.path.display()
            )));
        }
    }
    if let Some(q) = spec.quantize {
        return Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{MODEL_ID}: spec.quantize={q:?} is not supported on the candle lane — the pre-quantized \
             tiers are packed offline by the MLX lane's `convert` and this lane has no tier loader, \
             so a tier request is refused rather than silently rendered dense."
        )));
    }
    reject_unread_slots(spec)?;
    Ok(Box::new(MiniMaxH3::load(spec)?))
}

/// **Refuse every `LoadSpec` slot this provider does not read**, naming the ones the caller set.
///
/// The same defect class as the `adapters` / `quantize` checks above, one field over. A caller that
/// stages a ControlNet, an IP-Adapter, a PiD decoder, a face-ID bundle or an external text-encoder
/// snapshot for this model gets weights that **nothing in this crate ever opens** — grep `src/` and
/// `tests/` for any of these fields and the only hits are this function and the guard that proves
/// it. Silently discarding them is precisely what this provider was already caught doing with
/// `spec.adapters`.
///
/// This is the `candle-gen-flux2` accumulate-and-name idiom (`edit_provider.rs`), which reports
/// *every* offending slot in one message rather than only the first: a caller fixing them one round
/// trip at a time is a worse contract than one that lists them. The refusal is
/// `Error::Unsupported`, matching the sibling refusals in `candle-gen-mochi`, `candle-gen-wan` and
/// `candle-gen-ltx` for `control` / `extra_controls` / `ip_adapter`.
///
/// # The three slots that are deliberately NOT here
///
/// * **`spec.components`** — already refused, by `reject_unknown_components` inside
///   [`MiniMaxH3::load`], against this model's (empty) `required_components`.
/// * **`spec.offload_policy`** — deliberately *overridden*, not ignored: [`OFFLOAD_POLICY`] wins and
///   [`MiniMaxH3::offload_policy`] reports the enforced value, so the override is observable rather
///   than silent. Refusing a `Resident` request would break every caller that never asked for
///   staging; see the module docs.
/// * **`spec.load_shape`** — same shape. `crate::memory_strategy::LOAD_SHAPE` is pinned to the
///   loader this crate actually has and published on the memory contract whatever the spec says,
///   which `load_shape_is_pinned_to_the_loader_not_taken_from_the_spec` asserts. sc-18662 changes
///   the loader, not this decision.
fn reject_unread_slots(spec: &LoadSpec) -> candle_gen::gen_core::Result<()> {
    let mut unread: Vec<&str> = Vec::new();
    if spec.control.is_some() {
        unread.push("control");
    }
    if !spec.extra_controls.is_empty() {
        unread.push("extra_controls");
    }
    if spec.ip_adapter.is_some() {
        unread.push("ip_adapter");
    }
    if spec.pid.is_some() {
        unread.push("pid");
    }
    if spec.identity.is_some() {
        unread.push("identity");
    }
    if spec.text_encoder.is_some() {
        unread.push("text_encoder");
    }
    // `Precision::Bf16` is gen-core's "no override" sentinel, not a literal request, so only `Fp32`
    // is an ask. This provider hardcodes `DType::BF16` to match the published checkpoint's block
    // store; gen-core's own field docs say a provider that has not wired the override "rejects it
    // at `load`", which is what this is.
    if spec.precision != Precision::Bf16 {
        unread.push("precision");
    }
    if unread.is_empty() {
        return Ok(());
    }
    Err(candle_gen::gen_core::Error::Unsupported(format!(
        "{MODEL_ID}: the candle lane reads none of these LoadSpec slots — {}. No code in this \
         crate opens them, so honoring the load would render as though they were never set. \
         `control`/`extra_controls`/`ip_adapter` need the conditioning branches this lane has not \
         ported; `pid` and `identity` have no seam in this latent space; `text_encoder` is \
         co-located under the snapshot root and is not relocatable here; `precision` is pinned to \
         the checkpoint's bf16 block store. Drop the slot or pick a model that reads it.",
        unread.join(", ")
    )))
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

    /// This file's **production** source: the test module removed, then full-line comments stripped.
    ///
    /// Both exclusions are load-bearing, and each was observed to matter:
    ///
    /// * **The test module goes first.** This module's own staging table names every anchor the
    ///   guard searches for, so scanning the whole file finds the guard's own string literals and
    ///   the scan proves nothing about the code. The MLX sibling drops it for the same reason.
    /// * **Then comments.** Prose legitimately narrates the call sites — the module docs discuss
    ///   `drop` and the device release directly — so counting comment lines would make the gate
    ///   track the documentation rather than the code.
    fn production_source() -> String {
        let whole = include_str!("model.rs");
        let production = whole
            .split_once("\n#[cfg(test)]\n")
            .map_or(whole, |(before, _)| before);
        assert!(
            production.len() < whole.len(),
            "expected a `#[cfg(test)]` module in model.rs; if it moved, this scan is reading the \
             whole file and would match the test module's own anchor literals"
        );
        production
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The brace-matched body of one function, located by a signature that must be unique.
    fn body_of(src: &str, signature: &str) -> String {
        let start = src.find(signature).unwrap_or_else(|| {
            panic!(
                "model.rs no longer contains `{signature}`. This guard would otherwise silently \
                 stop watching a renamed function, so it fails instead of passing vacuously"
            )
        });
        assert!(
            !src[start + signature.len()..].contains(signature),
            "`{signature}` appears more than once; this scan reads only the first"
        );
        let open = start
            + src[start..]
                .find('{')
                .expect("a function signature is followed by a body");
        let mut depth = 0usize;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return src[open..open + i + 1].to_owned();
                    }
                }
                _ => {}
            }
        }
        panic!("unbalanced braces while extracting `{signature}`")
    }

    /// **The staging itself is pinned — not merely the policy field.**
    ///
    /// The test above asserts `MiniMaxH3::offload_policy()`, and that is a real guard against a
    /// `load` that starts echoing `spec.offload_policy`. It is also *all* it is: `self.offload` is
    /// read by nothing on any render path. Delete every `drop` and every device release from
    /// `generate_impl` and that test stays green, while the manifest's measured `vramGbByTier`
    /// silently stops describing the render — the true peak becomes the ~143 GB **sum** of the
    /// components instead of their max, and the worker admits cards against a number 3.5x too small.
    ///
    /// So this guard walks the staged phases of the three staging functions in this file's own
    /// production source. Each phase must map its component, drop it, and release the device —
    /// **before** the next phase maps anything. Deleting any single `drop(...)`, or any single
    /// `crate::dit::release_device_memory(...)`, makes this red on its own; each was mutated
    /// individually to prove it, not as a set.
    ///
    /// # What a source scan can and cannot see
    ///
    /// It **cannot** observe device bytes. It proves the releases are written and ordered, not that
    /// the allocator returned pages, and it says nothing about a component held alive by a path it
    /// does not name. The thing that *would* observe that is a weights-free behavioral seam — the
    /// MLX sibling's `MemoryBehaviorRegistration` shape, with per-route fixtures a harness executes
    /// without a checkpoint — and that is sc-18660's scope. Nothing in this crate can execute
    /// `generate_impl` without a ~143 GB snapshot, so this is the strongest guard available at this
    /// size: deliberately blunt, in exchange for being exact about the regression that was live.
    #[test]
    fn every_heavy_component_is_released_before_the_next_one_is_mapped() {
        /// The explicit device release. Dropping the Rust binding alone is not enough — candle
        /// hands the pages back to its own allocator, so without this the next phase maps on top.
        const RELEASE: &str = "crate::dit::release_device_memory(";
        /// The two bindings every text-encoder phase holds: the weight map and the tower built
        /// from it. The tower is built *before* the map is dropped, necessarily, so a phase is a
        /// group of maps and a group of releases rather than one pair.
        const TE_MAPS: &[&str] = &[
            "Weights::from_files_filtered",
            "MiniMaxH3TextEncoder::from_weights",
        ];

        /// One staged phase: a name for the message, the source anchors that MAP its component,
        /// and the source anchors that RELEASE it.
        type Phase = (
            &'static str,
            &'static [&'static str],
            &'static [&'static str],
        );

        // (function, its phases in source order).
        let staged: [(&str, &[Phase]); 3] = [
            (
                "fn encode_prompt(",
                &[(
                    "the 66.7 GB Qwen3-VL-32B text encoder",
                    TE_MAPS,
                    &["drop(w);", "drop(te);"],
                )],
            ),
            (
                "fn encode_prompt_grounded(",
                &[
                    (
                        "the Qwen3-VL vision tower",
                        &["load_vision_tower("],
                        &["drop(vision);"],
                    ),
                    (
                        "the 66.7 GB Qwen3-VL-32B text encoder",
                        TE_MAPS,
                        &["drop(w);", "drop(te);", "drop(grounded);"],
                    ),
                ],
            ),
            (
                "fn generate_impl(",
                &[
                    (
                        "the 10.4 GB video VAE, keyframe encode",
                        &["MiniMaxH3VideoVae::load("],
                        &["drop(vae);"],
                    ),
                    (
                        // `load_task_dit` is the single seam that maps the partition AND folds any
                        // staged LoRA (sc-18728), so the anchor tracks it rather than the inner
                        // `MiniMaxH3Dit::load` — which is now one call deeper and outside this
                        // function's body, where a scan of `generate_impl` cannot see it.
                        "the 66.3 GB DiT",
                        &["self.load_task_dit("],
                        &["drop(model);"],
                    ),
                    (
                        "the 10.4 GB video VAE, decode",
                        &["MiniMaxH3VideoVae::load_decode_only("],
                        &["drop(vae);"],
                    ),
                    (
                        "the audio VAE, decode",
                        &["self.load_audio_vae(", "MiniMaxH3AudioVae::from_weights("],
                        &["drop(w);", "drop(vae);"],
                    ),
                ],
            ),
        ];

        let src = production_source();
        for (signature, phases) in staged {
            let body = body_of(&src, signature);
            // An unbalanced brace inside a string literal defeats the matcher, and a body that ran
            // long would scan a sibling's staging as its own. Fail loudly on that rather than
            // passing on borrowed evidence. (Overrunning the last function instead hits EOF, which
            // `body_of` reports as unbalanced braces.)
            for (other, _) in staged {
                assert!(
                    other == signature || !body.contains(other),
                    "the extracted body of `{signature}` ran into `{other}`"
                );
            }

            let mut cursor = 0usize;
            for (phase, maps, releases) in phases {
                let mut first_map = usize::MAX;
                for m in *maps {
                    let at = cursor
                        + body[cursor..].find(m).unwrap_or_else(|| {
                            panic!(
                                "{signature} / {phase}: `{m}` no longer appears after the previous \
                                 phase's device release. Either the component is gone or the phase \
                                 order changed, and the measured sequential peak describes neither"
                            )
                        });
                    first_map = first_map.min(at);
                }
                let mut last_release = first_map;
                for r in *releases {
                    let at = first_map
                        + body[first_map..].find(r).unwrap_or_else(|| {
                            panic!(
                                "{signature} / {phase}: `{r}` is missing after the component is \
                                 mapped. A heavy component mapped and never dropped turns the \
                                 sequential peak into a SUM, and the manifest's measured \
                                 vramGbByTier admits cards against the max"
                            )
                        });
                    last_release = last_release.max(at);
                }
                cursor = last_release
                    + body[last_release..].find(RELEASE).unwrap_or_else(|| {
                        panic!(
                            "{signature} / {phase}: no `{RELEASE}` between this phase's last drop \
                             and the end of the function"
                        )
                    })
                    + RELEASE.len();
            }

            // Exhaustive for this function: a phase added without a release must not be able to
            // hide behind the phases that have one.
            let found = body.matches(RELEASE).count();
            assert_eq!(
                found,
                phases.len(),
                "{signature} has {found} `{RELEASE}` call(s) for {} pinned phase(s) — if a phase \
                 was added, add it to the table in this test so it is pinned too",
                phases.len()
            );
        }
    }

    /// A staged snapshot root with all four component dirs — enough for `load` to succeed.
    fn staged_root(tmp: &tempfile::TempDir) -> PathBuf {
        let root = tmp.path().to_path_buf();
        for c in REQUIRED_COMPONENT_DIRS {
            std::fs::create_dir_all(root.join(c)).unwrap();
        }
        root
    }

    /// A `LoadSpec` adapter spec at `kind`, with the two model-specific knobs optional.
    fn adapter_spec(
        kind: candle_gen::gen_core::AdapterKind,
        pass_scales: Option<Vec<f32>>,
        moe_expert: Option<candle_gen::gen_core::MoeExpert>,
    ) -> AdapterSpec {
        AdapterSpec {
            path: PathBuf::from("/turbo.safetensors"),
            scale: 1.0,
            kind,
            pass_scales,
            moe_expert,
        }
    }

    /// **A staged LoRA is CARRIED to the render, not dropped and not refused** (sc-18728).
    ///
    /// The predecessor of this test asserted the opposite — that `spec.adapters` was refused —
    /// because the seam did not exist. Now that it does, the property worth guarding flips: `load`
    /// must accept the spec **and** the loaded provider must still be holding it, because "accepted
    /// then discarded" is exactly the silent failure the old refusal existed to prevent, and it is
    /// indistinguishable from success at the `load` boundary alone.
    #[test]
    fn a_staged_lora_survives_load_rather_than_being_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);

        // Control: without adapters the same root loads and holds none.
        let bare = MiniMaxH3::load(&LoadSpec::new(WeightsSource::Dir(root.clone()))).unwrap();
        assert!(bare.adapters().is_empty());

        let spec = LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![adapter_spec(
            candle_gen::gen_core::AdapterKind::Lora,
            None,
            None,
        )]);
        load(&spec).expect("a LoRA spec must load now that the seam is ported");

        let model = MiniMaxH3::load(&spec).unwrap();
        assert_eq!(
            model.adapters().len(),
            1,
            "the spec's adapters must be RETAINED — a provider that accepted and forgot them would \
             render at the base checkpoint with no error at all"
        );
        assert_eq!(model.adapters()[0].scale, 1.0);
    }

    /// **The three adapter-shaped knobs this lane still cannot honor are refused by name.**
    ///
    /// Each is a field a caller can set on an otherwise-valid LoRA spec, and each is read by nothing
    /// in this crate. They are asserted **individually**, not as a set: a single "some knob is
    /// refused" arm would stay green if two of the three checks were deleted.
    ///
    /// The control at the top is what makes the arms attributable — the same path, the same scale,
    /// everything identical except the knob under test, loads cleanly.
    #[test]
    fn lokr_and_the_two_foreign_adapter_knobs_are_each_refused_individually() {
        use candle_gen::gen_core::{AdapterKind, MoeExpert};
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let with =
            |a: AdapterSpec| LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![a]);

        // Control: a plain LoRA with neither knob set loads.
        load(&with(adapter_spec(AdapterKind::Lora, None, None)))
            .expect("the plain LoRA control must load — otherwise the arms prove nothing");

        for (spec, needle) in [
            (adapter_spec(AdapterKind::Lokr, None, None), "LoKr"),
            (
                adapter_spec(AdapterKind::Lora, Some(vec![1.0, 0.5]), None),
                "pass_scales",
            ),
            (
                adapter_spec(AdapterKind::Lora, None, Some(MoeExpert::High)),
                "MoE expert",
            ),
        ] {
            // `Box<dyn Generator>` is not `Debug`, so `expect_err` is unavailable.
            let err = match load(&with(spec)) {
                Ok(_) => panic!("{needle} must be refused, not silently dropped"),
                Err(e) => e,
            };
            assert!(
                matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
                "must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(needle),
                "the refusal must name the knob it refused; wanted {needle:?}, got {msg}"
            );
        }
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

    /// **Every `LoadSpec` slot this provider does not read is refused, and each is proven alone.**
    ///
    /// `spec.adapters` and `spec.quantize` were the two this crate shipped first, but "declared and
    /// unenforced" is a *class*. `control`, `extra_controls` and `ip_adapter` are explicitly refused
    /// by `candle-gen-mochi`, `candle-gen-wan` and `candle-gen-ltx` — the very sibling idiom this
    /// provider cites — and H3 reads none of them, nor `pid`, `identity`, `text_encoder` or a
    /// non-default `precision`. Staging any of them produced a clean load and a render that behaved
    /// as though the caller had never set it.
    ///
    /// Each pass mutates exactly ONE slot against a control that loads the SAME fixture with the
    /// slot unset, so a refusal cannot be credited to a different slot or to a broken root. The
    /// message assertion reads the generated slot LIST, not the static prose (which names several
    /// slots and would satisfy a substring check even with the guard deleted) — so deleting any one
    /// `if` in `reject_unread_slots` fails exactly one pass.
    #[test]
    fn every_unread_loadspec_slot_is_refused_and_names_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let src = || WeightsSource::Dir(root.clone());

        for slot in [
            "control",
            "extra_controls",
            "ip_adapter",
            "pid",
            "identity",
            "text_encoder",
            "precision",
        ] {
            // The control, re-run every pass: the same root with the slot unset must still load.
            load(&LoadSpec::new(src())).unwrap_or_else(|e| {
                panic!(
                    "{slot}: the bare spec must load — otherwise the refusal proves nothing: {e}"
                )
            });

            let mut spec = LoadSpec::new(src());
            match slot {
                "control" => spec.control = Some(src()),
                "extra_controls" => spec.extra_controls = vec![src()],
                "ip_adapter" => spec.ip_adapter = Some(src()),
                "pid" => {
                    spec.pid = Some(candle_gen::gen_core::PidWeights {
                        checkpoint: src(),
                        gemma: src(),
                    })
                }
                "identity" => {
                    spec.identity = Some(candle_gen::gen_core::IdentityWeights {
                        encoder: Some(src()),
                        eva: None,
                        face_dir: None,
                    })
                }
                "text_encoder" => spec.text_encoder = Some(src()),
                // `Precision::Bf16` is gen-core's "no override" sentinel, so `Fp32` is the only ask.
                "precision" => spec.precision = Precision::Fp32,
                other => unreachable!("unhandled slot `{other}`"),
            }

            let err = match load(&spec) {
                Ok(_) => panic!(
                    "`{slot}` was set and the load SUCCEEDED — nothing in this crate reads it, so \
                     the render would silently ignore the caller's weights"
                ),
                Err(e) => e,
            };
            assert!(
                matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
                "{slot}: must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("slots — {slot}.")),
                "the refusal must NAME the offending slot in its list so a caller knows what to \
                 drop: {msg}"
            );
        }
    }

    /// The three `LoadSpec` slots that are decided but deliberately **not** refused stay that way.
    ///
    /// `offload_policy` and `load_shape` are *overridden and reported*, not ignored — refusing them
    /// would break every caller that never asked for staging — and `components` is refused a layer
    /// down by `reject_unknown_components`. This pins that a bare spec carrying the non-default
    /// residency knobs still loads, so the refusal list above cannot quietly grow to cover them.
    #[test]
    fn the_residency_knobs_are_overridden_rather_than_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let root = staged_root(&tmp);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Resident)
            .with_load_shape(candle_gen::gen_core::LoadShape::DeferredMaterialization);
        load(&spec).unwrap_or_else(|e| {
            panic!("the residency knobs are honored by override, not refused, but load failed: {e}")
        });

        // ...and an unknown named component IS refused, one layer down.
        let unknown = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_component("transformer_ref", WeightsSource::Dir(root));
        assert!(
            load(&unknown).is_err(),
            "an unrecognized named component must be refused by reject_unknown_components"
        );
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
        // **The turbo LoRA seam IS ported** (sc-18728) — and this bit is only an advertisement:
        // gen-core reads `supports_lora` on no path, so what a consumer can rely on is
        // `crate::adapters`, exercised end-to-end in `tests/turbo_lora.rs`.
        assert!(d.capabilities.supports_lora);
        // LoKr stays false and is enforced twice in real code (by kind in `load`, by file content
        // in the installer) precisely because this flag refuses nothing on its own.
        assert!(!d.capabilities.supports_lokr);
        assert!(d.capabilities.supported_quants.is_empty());
        // The residency lifecycle IS wired — and forced — so the discovery signal says so. This is
        // the bit a consumer reads to know whether `OffloadPolicy::Sequential` bounds the peak here
        // or is a silent no-op; `false` would tell the fit-gate to size this render at the SUM of
        // the components. It moves in lockstep with `OFFLOAD_POLICY` and the staging tripwire.
        assert!(d.capabilities.supports_sequential_offload);
        assert_eq!(OFFLOAD_POLICY, OffloadPolicy::Sequential);
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
