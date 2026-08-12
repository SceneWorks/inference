//! The **`minimax_h3` generator** (sc-17147): prompt → real video frames plus a real synchronized
//! stereo soundtrack.
//!
//! `t2va` on the `transformer` partition. The checkpoint is guidance-distilled — no negative prompt,
//! no guidance scale, one transformer forward per step.
//!
//! # Phases, and why nothing is held resident
//!
//! The three heavy components are **66.7 GB** of Qwen3-VL-32B text encoder, **62 GB** of DiT (38.7 GB
//! once the AdaLN projections are evicted — sc-17145) and **10 GB** of video VAE. Holding any two of
//! them at once does not fit a sensible budget, so `MiniMaxH3::generate_impl` builds each phase's
//! component, uses it, drops it and drains MLX's allocator cache before the next — the staged shape
//! the LTX provider uses, applied to a much larger spread. `load` therefore holds **paths**, not
//! tensors: it validates that every partition the render needs is present and defers the reads.
//!
//! # Geometry: `frames` is a lattice point, `duration` is a request
//!
//! Two request fields reach the same quantity and they are deliberately treated differently:
//!
//! * **`frames`** names a point on the model's own `17n + 5` lattice. An off-lattice value is
//!   **rejected** ([`crate::pipeline::resolve_geometry`]) — SceneWorks normalizes dimensions
//!   upstream, so a gate that silently refits is a gate that can never be observed to fire, and at
//!   video scale a refit is three quarters of a second of picture the caller never asked for.
//! * **`duration`** is a continuous seconds value with no lattice of its own, so it is aligned
//!   **upward** to the next legal frame count ([`crate::pipeline::align_frames_for_duration`]) — the
//!   reference ships `align_num_frames` for exactly this — and the alignment is bounded by the same
//!   5-15 s range rather than being a way around it.
//!
//! When both are present `frames` wins, because it is the exact one.

use std::path::{Path, PathBuf};

use mlx_rs::Dtype;

use mlx_gen::gen_core::{
    reject_unknown_components, Capabilities, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, KeyframeRef, LoadSpec, Modality, ModelDescriptor, Precision, Progress, SizeFloor,
    WeightsSource,
};
use mlx_gen::weights::Weights;
use mlx_gen::{default_seed, Error, Result};
use mlx_gen_boogu::vision::VisionTower;

use crate::audio_config::{MiniMaxH3AudioVaeConfig, AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::audio_vae::MiniMaxH3AudioVae;
use crate::denoise::{JointSchedule, MIN_INFERENCE_STEPS};
use crate::dit::adaln::AdaLnResidency;
use crate::dit::model::{JointDit, MiniMaxH3Dit};
use crate::pipeline::{
    fit_audio_to_video, fl2va_layout, frames_to_images, initial_latents, prepend_condition_rows,
    render_latents, resolve_geometry, revert_pixel_normalization, t2va_layout, RequestGeometry,
    MAX_DURATION_SECONDS, MIN_DURATION_SECONDS, SMALLEST_LEGAL_FRAMES, SPATIAL_STRIDE,
};
use crate::text_encoder::{
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, LM_PREFIX, VISION_PREFIX,
};
use crate::vae::MiniMaxH3VideoVae;

/// The published provider id.
pub const MODEL_ID: &str = "minimax_h3";

/// Model evaluations a request runs when it names no step count.
///
/// The reference declares no default; 50 is what the sc-17242 spike rendered at and what the model
/// card's own examples use. **Evaluations**, not `MiniMaxH3Scheduler`'s `num_inference_steps` —
/// that count includes the terminal `σ = 0` the model is never evaluated at, so the schedule is
/// built with `steps + 1` (see [`crate::denoise::schedule`]).
pub const DEFAULT_STEPS: u32 = 50;

/// Upper bound on requested steps. A provider-local guard: every step is a full 33 B forward and the
/// AdaLN cache grows with the distinct-timestep count, so an unbounded value is a resource hazard
/// rather than a slow render.
pub const MAX_STEPS: u32 = 200;

/// Text-encoder shards the layer-50 tap needs. Shards 13-14 hold only the never-executed tail
/// (layers 50-63 and `lm_head`), so mapping the whole 66.7 GB component would read 15 GB for
/// nothing.
const TE_SHARDS: std::ops::RangeInclusive<u32> = 1..=12;

/// The shard holding the Qwen3-VL **vision tower**. All 351 `model.visual.*` tensors live in
/// shard 14 and nowhere else, so [`TE_SHARDS`] excludes it for `t2va` and `fl2va` must add it back.
const VISION_SHARD: u32 = 14;

/// Quantization group size the shared vision tower is built with — the same value every other
/// consumer of `mlx-gen-boogu`'s tower passes.
const VISION_GROUP_SIZE: i32 = 64;

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "minimax_h3",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch exists anywhere in the checkpoint.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // `t2va` and `fl2va` (sc-17148). The omni-reference (`ref2va`, `transformer_ref`) is
            // sc-17149.
            //
            // **One kind covers all three keyframe shapes.** `Keyframe` carries a `frame_idx`, and
            // first-only / last-only / first+last are one, one and two of them — see
            // [`keyframe_anchors`] for why last-frame-only is a payload shape rather than a mode
            // of its own.
            conditioning: vec![ConditioningKind::Keyframe],
            supports_lora: false,
            supports_lokr: false,
            supported_quants: &[],
            component_precision_floors: &[],
            samplers: Vec::new(),
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: SPATIAL_STRIDE,
            max_size: 1344,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_preview: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // The audio surface describes a *selectable* audio request (voice / language / rate),
            // which a video model has none of — the soundtrack rides `GenerationOutput::Video`.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
        },
    }
}

/// The loaded generator — paths and precision, deliberately no tensors. See the module docs.
pub struct MiniMaxH3 {
    descriptor: ModelDescriptor,
    root: PathBuf,
    dtype: Dtype,
}

/// Reject a request this model cannot serve, before any weight is read.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{MODEL_ID}: prompt must not be empty")));
    }
    caps.validate_request(MODEL_ID, req)?;
    if let Some(steps) = req.steps {
        if steps == 0 || steps > MAX_STEPS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: steps must be in 1..={MAX_STEPS}, got {steps}"
            )));
        }
    }
    // The geometry gate itself — the same call `generate` makes, so `validate` and the render agree
    // by construction rather than by two copies of the lattice arithmetic.
    request_geometry(req).map(|_| ())
}

/// Map a request's keyframes onto MiniMax-H3's two anchor slots, in packed order.
///
/// # The last-frame-only decision (sc-17148)
///
/// The story asks whether last-frame-only should be a **new SceneWorks mode** or an **optional
/// field on `first_last_frame`**. It is the latter, and this function is where that is expressed:
/// there is no third mode, only a `Keyframe` payload whose `frame_idx` names which end it anchors.
///
/// Three reasons, in order of weight:
///
/// 1. **Upstream does not model it as a separate task either.** `MiniMaxH3Blocks._workflow_map`
///    has one `fl2va` entry with two accepted signatures (`image`, or `last_image`), and
///    `before_encoder.py` filters a fixed `(("first", image), ("last", last_image))` pair by
///    presence. All four shapes fall out of one ordered tuple. A third mode would model as
///    disjoint something the reference models as one mechanism with an optional slot.
/// 2. **The information is already carried.** `frame_idx` distinguishes the two ends losslessly. A
///    new mode would add a manifest entry (sc-17158), an allow-list entry and routing across six
///    surfaces (sc-17159), and a Video Studio affordance (sc-17161) — all to re-encode one bit
///    that the existing payload already has.
/// 3. **The genuine difference is a prompting one, not a plumbing one.** Upstream's own
///    `VIDEO_PROMPT_WRITING_GUIDE` names I2VA / L2VA / FL2VA as distinct tasks because they take
///    **different prompt preambles**. That is UI copy and preset territory, and it is well served
///    by a preset over `first_last_frame` rather than by a mode the engine has to branch on.
///
/// What follows for the surfaces this story does not own: `first_last_frame` must accept a payload
/// with **only** the last slot filled. sc-17159 owns making that reachable; the engine accepts it
/// today.
///
/// # The index convention
///
/// `frame_idx` 0 is the first frame. The last is `num_frames - 1` **or** `-1`, because a caller
/// that knows only "the end of the clip" should not have to resolve the frame count first. Any
/// other index is **rejected**: the model has exactly two anchor slots, and silently snapping a
/// mid-clip index to the nearest end would condition on something the caller did not ask for.
pub(crate) fn keyframe_anchors(
    keyframes: &[KeyframeRef<'_>],
    num_frames: i32,
) -> Result<Vec<crate::dit::positions::KeyframeAnchor>> {
    use crate::dit::positions::KeyframeAnchor;
    let mut first = None;
    let mut last = None;
    for kf in keyframes {
        let slot = if kf.frame_idx == 0 {
            &mut first
        } else if kf.frame_idx == -1 || kf.frame_idx == num_frames - 1 {
            &mut last
        } else {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a keyframe must anchor the FIRST frame (index 0) or the LAST (index \
                 {} or -1); MiniMax-H3 has exactly two anchor slots and no mid-clip conditioning, \
                 got index {}",
                num_frames - 1,
                kf.frame_idx
            )));
        };
        if slot.is_some() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: two keyframes anchor the same end of the clip (index {})",
                kf.frame_idx
            )));
        }
        *slot = Some(kf.image);
    }
    let mut anchors = Vec::with_capacity(2);
    if first.is_some() {
        anchors.push(KeyframeAnchor::First);
    }
    if last.is_some() {
        anchors.push(KeyframeAnchor::Last);
    }
    Ok(anchors)
}

/// The request's keyframe images in packed order — first, then last.
///
/// Positional with [`keyframe_anchors`]: the caller may supply them in any order, and both
/// functions sort them into the reference's `(first, last)` order so `anchors[i]` always describes
/// `images[i]`.
pub(crate) fn keyframe_images<'a>(
    keyframes: &[KeyframeRef<'a>],
    num_frames: i32,
) -> Result<Vec<&'a mlx_gen::media::Image>> {
    let mut first = None;
    let mut last = None;
    for kf in keyframes {
        if kf.frame_idx == 0 {
            first = Some(kf.image);
        } else if kf.frame_idx == -1 || kf.frame_idx == num_frames - 1 {
            last = Some(kf.image);
        }
    }
    // Re-run the validation so the two functions cannot disagree about what is legal.
    keyframe_anchors(keyframes, num_frames)?;
    Ok(first.into_iter().chain(last).collect())
}

/// Resolve the request's geometry: the canvas, and the frame count from `frames` (exact) or
/// `duration` (aligned).
pub(crate) fn request_geometry(req: &GenerationRequest) -> Result<RequestGeometry> {
    let frames = match (req.frames, req.duration) {
        (Some(frames), _) => i32::try_from(frames)
            .map_err(|_| Error::Msg(format!("{MODEL_ID}: frames {frames} does not fit an i32")))?,
        (None, Some(seconds)) => crate::pipeline::align_frames_for_duration(seconds)?,
        (None, None) => SMALLEST_LEGAL_FRAMES,
    };
    if let Some(fps) = req.fps {
        if f64::from(fps) != crate::denoise::MINIMAX_H3_FPS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the released model generates at {} fps only, got {fps}",
                crate::denoise::MINIMAX_H3_FPS
            )));
        }
    }
    resolve_geometry(req.width, req.height, frames)
}

/// Load the generator from a `MiniMaxAI/MiniMax-H3` snapshot root.
///
/// Validates that every partition the `t2va` render reads is present — `transformer/`,
/// `text_encoder/`, `tokenizer/`, `vae/`, `audio_vae/` — and reads none of them. An absent partition
/// is a **load-time** error with the missing path named; this loader never derives a cache location
/// and never self-fetches (epic 13657).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: expected a MiniMax-H3 snapshot directory (the dir holding \
                 `transformer/`), not a single file"
            )))
        }
    };
    reject_unknown_components(spec, &[], MODEL_ID)?;
    if spec.quantize.is_some() {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: quantized tiers are not wired yet (sc-17150); load the bf16 snapshot"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: adapters are not supported"
        )));
    }
    for (partition, probe) in [
        ("transformer", "config.json"),
        ("vae", "config.json"),
        ("audio_vae", "config.json"),
        ("text_encoder", "config.json"),
        ("tokenizer", "tokenizer.json"),
        // The audio VAE's *constructor* arguments live only in the three FL2VA source documents;
        // the repackaged root config carries none of them (see `MiniMaxH3AudioVaeConfig`).
        ("FL2VA/audio_vae", "config.json"),
        ("FL2VA/audio_vae", "config.yaml"),
        ("FL2VA/audio_vae", "metadata.json"),
    ] {
        let path = root.join(partition).join(probe);
        if !path.is_file() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: missing {} in the snapshot root {}",
                path.strip_prefix(&root).unwrap_or(&path).display(),
                root.display()
            )));
        }
    }
    let dtype = match spec.precision {
        Precision::Fp32 => Dtype::Float32,
        _ => Dtype::Bfloat16,
    };
    Ok(Box::new(MiniMaxH3 {
        descriptor: descriptor(),
        root,
        dtype,
    }))
}

/// Release a component's device memory for real: drop the Rust handle, then drain MLX's allocator
/// cache so the buffers go back to the system rather than migrating active → cache.
fn release<T>(component: T) {
    drop(component);
    mlx_rs::memory::clear_cache();
}

impl MiniMaxH3 {
    /// The snapshot root this generator reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The block stack's precision.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Map the text-encoder shards a presentation needs.
    ///
    /// `t2va` reads shards 1-12 ([`TE_SHARDS`]). `fl2va` additionally needs **shard 14**, which is
    /// where all 351 `model.visual.*` tensors live — the vision tower is not spread across the
    /// component, it sits entirely in the last shard alongside the never-executed decoder tail.
    /// That is why the `t2va` window could stop at 12 and why a keyframe request cannot.
    fn map_te_shards(&self, with_vision: bool) -> Result<Weights> {
        let dir = self.root.join("text_encoder");
        let mut w = Weights::empty();
        let shards: Vec<u32> = if with_vision {
            TE_SHARDS.chain(std::iter::once(VISION_SHARD)).collect()
        } else {
            TE_SHARDS.collect()
        };
        for i in shards {
            let shard = format!("model-{i:05}-of-00014.safetensors");
            let part = Weights::from_file(dir.join(&shard))?;
            let keys: Vec<String> = part.keys().map(str::to_owned).collect();
            for k in keys {
                // Shard 14 also holds layers 50-63 and `lm_head`, which are never executed. Keep
                // only what the tower needs so the fl2va path does not carry 15 GB for nothing.
                if with_vision && i == VISION_SHARD && !k.starts_with(VISION_PREFIX) {
                    continue;
                }
                let t = part.require(&k)?.clone();
                w.insert(k, t);
            }
        }
        Ok(w)
    }

    /// Encode the prompt and immediately release the 66.7 GB text encoder.
    fn encode_prompt(&self, prompt: &str) -> Result<mlx_rs::Array> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) = tok.encode_prompt(prompt)?;

        let w = self.map_te_shards(false)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)?;
        let context = te.forward(&ids, &mask)?;
        // Force it BEFORE the encoder is dropped: under lazy evaluation the context is a graph node
        // holding every weight it was computed from, so dropping first would free nothing and the
        // first denoise step would re-materialize 66.7 GB.
        mlx_rs::transforms::eval([&context])?;
        release((te, w));
        Ok(context)
    }

    /// The `fl2va` presentation: run the Qwen3-VL **vision tower** over the keyframes, splice their
    /// embeddings into the `"<Picture i>: "` + vision-block presentation, and return the context
    /// together with its **per-row modality tags**.
    ///
    /// This is one of the two paths a keyframe takes. The other is the VAE encode in
    /// [`crate::conditioning`]; the sc-17242 spike established that the reference runs **both**,
    /// and neither substitutes for the other — the tower supplies semantic context to the prompt
    /// stream, the VAE supplies the pixel-space anchor the video rows are conditioned on.
    fn encode_prompt_grounded(
        &self,
        prompt: &str,
        keyframes: &[&mlx_gen::media::Image],
    ) -> Result<(mlx_rs::Array, Vec<i32>)> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let mut w = self.map_te_shards(true)?;

        let vision = VisionTower::from_weights(
            &w,
            crate::text_encoder::minimax_h3_vision_config(),
            VISION_PREFIX,
            VISION_GROUP_SIZE,
        )?;
        let grounded = crate::text_encoder::run_vision(&vision, keyframes)?;
        // Force the tower's output BEFORE dropping it, and drop its tensors out of `w` too — the
        // same discipline `encode_prompt` documents. Under lazy evaluation `grounded` is a graph
        // node holding every weight it was computed from, so releasing the tower while `w` still
        // maps `model.visual.*` frees nothing and the splice below would re-materialize it.
        let mut forced: Vec<&mlx_rs::Array> = grounded.embeds.iter().collect();
        forced.extend(grounded.deepstack.iter().flatten());
        mlx_rs::transforms::eval(forced)?;
        release(vision);
        w.remove_prefix(VISION_PREFIX);

        let (ids, mask, tags) = tok.encode_fl2va(prompt, &grounded.counts)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)?;
        let context = te.forward_with_images(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        mlx_rs::transforms::eval([&context])?;
        release((te, w, grounded));

        // The tags describe the presentation row for row; a mismatch would mis-tag every row after
        // the divergence and is silent, because both lengths build a runnable sequence.
        if context.shape()[1] != tags.len() as i32 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the grounded context has {} rows but {} modality tags",
                context.shape()[1],
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
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let geometry = request_geometry(req)?;
        let evaluations = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        // `num_inference_steps` counts the terminal σ = 0, at which the model is never evaluated.
        let schedule = JointSchedule::new(evaluations + 1)?;
        if schedule.num_evals() == 0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a {evaluations}-step schedule collapsed to no model evaluations \
                 (num_inference_steps must be >= {MIN_INFERENCE_STEPS})"
            )));
        }
        let seed = req.seed.unwrap_or_else(default_seed);

        // --- 1. conditioning ----------------------------------------------------------------
        // `t2va` and `fl2va` diverge here and stay diverged through the latent prep — deliberately,
        // as the reference does. Zero keyframes is `t2va` at the WEIGHTS level (one `transformer`
        // partition) and a different block path at every other level.
        let keyframes = req.keyframes();
        let anchors = keyframe_anchors(&keyframes, geometry.joint.num_frames)?;
        // Fitted ONCE, here, and shared by both keyframe paths. The vision tower and the VAE
        // encode must see the same pixels — resizing separately per path would let a resampling
        // difference put the two conditioning signals fractionally out of register, which is
        // exactly the class of divergence nothing downstream can detect.
        let fitted = if anchors.is_empty() {
            Vec::new()
        } else {
            let images = keyframe_images(&keyframes, geometry.joint.num_frames)?;
            crate::keyframe::fit_keyframes(&images, geometry.width, geometry.height)?
        };

        let (context, text_tags) = if anchors.is_empty() {
            let context = self.encode_prompt(&req.prompt)?;
            let tags = vec![crate::denoise::TEXT_TAG; context.shape()[1] as usize];
            (context, tags)
        } else {
            let refs: Vec<&mlx_gen::media::Image> = fitted.iter().collect();
            self.encode_prompt_grounded(&req.prompt, &refs)?
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 1b. keyframe conditioning latents (fl2va) ----------------------------------------
        // The VAE encode, the second of the keyframe's two paths. Done before the DiT is mapped so
        // the ~10 GB VAE and the ~62 GB DiT are never both resident.
        let condition_rows = if anchors.is_empty() {
            None
        } else {
            let pixels: Vec<mlx_rs::Array> = fitted
                .iter()
                .map(crate::keyframe::keyframe_to_vae_pixels)
                .collect::<Result<_>>()?;
            let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
            let rows = crate::conditioning::build_condition_rows(
                &vae,
                &pixels,
                &anchors,
                crate::pipeline::PATCH_SIZE,
                &crate::conditioning::KeyframeNoise::Seeded,
            )?;
            if let Some(r) = &rows {
                // Force before the VAE is dropped, for the same lazy-evaluation reason the context
                // is forced above.
                mlx_rs::transforms::eval([r])?;
            }
            release((vae, pixels));
            rows
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. denoise ---------------------------------------------------------------------
        let dit = MiniMaxH3Dit::load(&self.root, "transformer", self.dtype)?;
        let patch = dit.config().patch_size;
        let layout = if anchors.is_empty() {
            t2va_layout(&geometry, context.shape()[1], patch)?
        } else {
            fl2va_layout(&geometry, &text_tags, &anchors, patch)?
        };
        let (video_rows, audio_rows) = initial_latents(&geometry, patch, seed)?;
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
        release(context);

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
            &req.cancel,
            &mut on_step,
        )?;
        release((model, video_rows, audio_rows));

        // --- 3. decode ----------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let frames = self.decode_video(&rendered.video)?;
        let audio = self.decode_audio(&rendered.audio)?;
        let audio = fit_audio_to_video(audio, &geometry)?;
        if audio.sample_rate != AUDIO_SAMPLE_RATE || audio.channels != AUDIO_OUTPUT_CHANNELS {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the decoded soundtrack is {} Hz / {} channels, expected \
                 {AUDIO_SAMPLE_RATE} / {AUDIO_OUTPUT_CHANNELS}",
                audio.sample_rate, audio.channels
            )));
        }
        if frames.len() != geometry.joint.num_frames as usize {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: decoded {} frames for a {}-frame request",
                frames.len(),
                geometry.joint.num_frames
            )));
        }
        Ok(GenerationOutput::Video {
            frames,
            fps: crate::denoise::MINIMAX_H3_FPS as u32,
            audio: Some(audio),
        })
    }

    fn decode_video(&self, latents: &mlx_rs::Array) -> Result<Vec<mlx_gen::media::Image>> {
        let vae = MiniMaxH3VideoVae::load(&self.root, self.dtype)?;
        let decoded = vae.decode(latents)?;
        let pixels = revert_pixel_normalization(&decoded)?;
        mlx_rs::transforms::eval([&pixels])?;
        release(vae);
        frames_to_images(&pixels)
    }

    fn decode_audio(&self, latents: &mlx_rs::Array) -> Result<mlx_gen::media::AudioTrack> {
        // The geometry comes from the three FL2VA source documents the reference's own
        // `from_pretrained` reads — parsed, not defaulted, so a variant snapshot cannot silently
        // run at this one's numbers. The weights come from the repackaged `audio_vae/`, which ships
        // the same tensors under the same names.
        let source = self.root.join("FL2VA").join("audio_vae");
        let read = |name: &str| -> Result<String> {
            let path = source.join(name);
            std::fs::read_to_string(&path)
                .map_err(|e| Error::Msg(format!("{MODEL_ID}: reading {}: {e}", path.display())))
        };
        let cfg = MiniMaxH3AudioVaeConfig::from_source_files(
            &read("config.json")?,
            &read("config.yaml")?,
            &read("metadata.json")?,
        )?;
        let mut w = Weights::from_dir(self.root.join("audio_vae"))?;
        let vae = MiniMaxH3AudioVae::from_weights(&mut w, &cfg, Dtype::Float32)?;
        let track = vae.decode_audio_track(latents)?;
        release((vae, w));
        Ok(track)
    }
}

/// The video VAE decodes a *whole* clip in one call, so the memory contract is the same one
/// `MiniMaxH3VideoVae::decode_temporal` already owns.
const _: () = {
    assert!(MIN_DURATION_SECONDS < MAX_DURATION_SECONDS);
};

mlx_gen::impl_generator!(MiniMaxH3 {
    validate: |s, req| validate_request(&s.descriptor.capabilities, req),
    generate: generate_impl,
});

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::CancelFlag;

    fn request(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            width,
            height,
            cancel: CancelFlag::default(),
            ..Default::default()
        }
    }

    /// The descriptor's shape: a Mac-only video model with no guidance surface and no conditioning.
    #[test]
    fn the_descriptor_declares_a_guidance_free_video_model() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.backend, "mlx");
        assert!(matches!(d.modality, Modality::Video));
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert_eq!(d.capabilities.max_count, 1);
        assert_eq!(d.capabilities.min_size, SPATIAL_STRIDE);
        assert!(d.capabilities.mac_only);
        assert!(
            d.capabilities.audio_sample_rates.is_empty(),
            "the audio surface describes an audio REQUEST, which a video model has none of"
        );
    }

    /// **The geometry gate fires from `validate`, before any weight is read.** This is the
    /// acceptance criterion's "rejected, not silently refit" at the contract boundary.
    #[test]
    fn validate_rejects_off_lattice_geometry() {
        let caps = descriptor().capabilities;

        // An explicit off-lattice frame count.
        for frames in [123u32, 125, 129, 200] {
            let req = GenerationRequest {
                frames: Some(frames),
                ..request(576, 320)
            };
            let e = validate_request(&caps, &req).unwrap_err().to_string();
            assert!(e.contains("17n + 5"), "{frames}: {e}");
        }
        // A canvas that is 16-aligned but not 32-aligned.
        let e = validate_request(&caps, &request(592, 320))
            .unwrap_err()
            .to_string();
        assert!(e.contains("multiple of 32"), "{e}");

        // ...and the legal request passes.
        validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(124),
                ..request(576, 320)
            },
        )
        .unwrap();
    }

    /// `frames` is exact and `duration` is aligned — the split, pinned in both directions.
    #[test]
    fn frames_are_exact_and_duration_is_aligned() {
        // 5.3 s has no lattice point; it aligns UP to 141 frames.
        let by_duration = GenerationRequest {
            duration: Some(5.3),
            ..request(576, 320)
        };
        assert_eq!(
            request_geometry(&by_duration).unwrap().joint.num_frames,
            141
        );

        // The same clip requested as 127 frames is refused rather than aligned.
        let by_frames = GenerationRequest {
            frames: Some(127),
            ..request(576, 320)
        };
        assert!(request_geometry(&by_frames).is_err());

        // `frames` wins when both are given.
        let both = GenerationRequest {
            frames: Some(124),
            duration: Some(15.0),
            ..request(576, 320)
        };
        assert_eq!(request_geometry(&both).unwrap().joint.num_frames, 124);

        // No frames, no duration: the shortest legal render.
        assert_eq!(
            request_geometry(&request(576, 320))
                .unwrap()
                .joint
                .num_frames,
            SMALLEST_LEGAL_FRAMES
        );
    }

    /// The model generates at 24 fps and says so rather than quietly retiming.
    #[test]
    fn a_foreign_frame_rate_is_rejected() {
        let req = GenerationRequest {
            fps: Some(30),
            ..request(576, 320)
        };
        let e = request_geometry(&req).unwrap_err().to_string();
        assert!(e.contains("24 fps"), "{e}");
    }

    /// An empty prompt and an out-of-range step count are contract errors.
    #[test]
    fn malformed_requests_are_rejected() {
        let caps = descriptor().capabilities;
        let blank = GenerationRequest {
            prompt: "   ".into(),
            ..request(576, 320)
        };
        assert!(validate_request(&caps, &blank).is_err());
        let too_many = GenerationRequest {
            steps: Some(MAX_STEPS + 1),
            ..request(576, 320)
        };
        assert!(validate_request(&caps, &too_many).is_err());
        let none = GenerationRequest {
            steps: Some(0),
            ..request(576, 320)
        };
        assert!(validate_request(&caps, &none).is_err());
    }

    /// A missing partition is a load error naming the path, not a mid-render surprise.
    #[test]
    fn a_snapshot_missing_a_partition_fails_at_load() {
        let dir = tempfile::tempdir().unwrap();
        // `Box<dyn Generator>` is not `Debug`, so the error is destructured rather than unwrapped.
        let message = |spec: &LoadSpec| match load(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("this spec must not load"),
        };
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let e = message(&spec);
        assert!(e.contains("transformer"), "{e}");
        // A single file is the wrong shape entirely.
        let file = LoadSpec::new(WeightsSource::File(dir.path().join("x.safetensors")));
        let e = message(&file);
        assert!(e.contains("directory"), "{e}");
    }

    /// The requested step count is model EVALUATIONS; the schedule adds the terminal σ = 0.
    #[test]
    fn the_requested_step_count_is_evaluations() {
        let s = JointSchedule::new(DEFAULT_STEPS as usize + 1).unwrap();
        assert_eq!(s.num_evals(), DEFAULT_STEPS as usize);
        assert_eq!(DEFAULT_STEPS, 50);
    }
}
