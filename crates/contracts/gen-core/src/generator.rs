//! The `Generator` contract — prompt-conditioned synthesis of image, video, **or** audio
//! (or a mix), including multi-modal models. See `docs/MODEL_ARCHITECTURE.md` §3.1.
//!
//! One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX
//! (text→video+audio), and pure audio synthesis (TTS / music). Modality is a
//! [`ModelDescriptor`] property plus a [`GenerationOutput`] variant — *not* a per-modality
//! trait split (which breaks on multi-modal models).

use crate::media::{AudioTrack, Image};
use crate::runtime::{CancelFlag, Progress, Quant};
use crate::{Error, Result};

/// A prompt-conditioned media generator. `generate` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread); the request carries a cancel flag and
/// `on_progress` streams step/decode progress.
pub trait Generator {
    /// Identity + capabilities + modality (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &ModelDescriptor;

    /// Reject a request this model cannot serve (unsupported conditioning, guidance on a
    /// distilled model, out-of-range size/count, …) before doing expensive work.
    fn validate(&self, req: &GenerationRequest) -> Result<()>;

    /// Run generation to completion (or until `req.cancel` trips).
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput>;
}

/// What a [`Generator`] produced. The `Video` variant's `audio` is `Some` for LTX (always
/// audio) and `None` for Wan — no contract change needed across the two. The `Audio` variant
/// is a **pure** audio synthesis (TTS / music — `Modality::Audio`); audio attached to a video
/// stays on the `Video` variant.
#[derive(Clone, Debug)]
pub enum GenerationOutput {
    Images(Vec<Image>),
    Video {
        frames: Vec<Image>,
        fps: u32,
        audio: Option<AudioTrack>,
    },
    Audio(AudioTrack),
}

/// The request union (lifted from the SceneWorks worker's `ImageRequest`/`VideoRequest`). Most
/// fields are optional; a model reads what it supports and `validate()` rejects the rest. A
/// single `Default`-able struct (no builder): `GenerationRequest { prompt, ..Default::default() }`.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    // --- Core ---
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Number of images to produce (1..=8 for image models).
    pub count: u32,

    // --- Sampling (all optional; model/descriptor supply defaults) ---
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub true_cfg: Option<f32>,
    /// CFG-scheduling start step — the companion to [`true_cfg`](Self::true_cfg): real classifier-free
    /// guidance (and any per-branch conditioning gated with it) engages only once the denoise reaches
    /// this step, leaving earlier steps single-forward. `None` ⇒ each model's own default. Today only
    /// PuLID-FLUX honors it (default 1; its photoreal preset uses 4 to delay CFG a few steps); models
    /// without CFG scheduling ignore it.
    pub timestep_to_start_cfg: Option<u32>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,

    /// Guidance method — how the conditional and unconditional model predictions are combined (epic
    /// 7434, the fourth orthogonal sampling layer). `"cfg"` (plain) | `"cfg_rescale"` (Lin et al.
    /// per-token norm-rescale) | `"apg"` (adaptive-projected guidance) | `"cfg_pp"` (CFG++ — renoise
    /// from the unconditional branch). `None` ⇒ the engine's default guidance path (the N1 no-op).
    /// Gated per-model-per-backend by [`Capabilities::supported_guidance_methods`]; an unadvertised
    /// value is rejected here at the contract boundary, and dropped-to-default with a worker event by
    /// the N3 fallback layer (P5).
    pub guidance_method: Option<String>,
    /// APG projection mix η (`apg` only): recombine as `orthogonal + η·parallel` against the
    /// conditional base. `η = 1` with no momentum and `norm_threshold = 0` reduces APG to plain CFG.
    /// `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_eta: Option<f32>,
    /// APG momentum (`apg` only): `running = diff + momentum·running`, the buffer persisting across
    /// denoise steps (0 ⇒ no momentum). `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_momentum: Option<f32>,
    /// APG norm-threshold (`apg` only): clamp the guidance delta to `‖diff‖ ≤ norm_threshold`
    /// (`0` disables the clamp). `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_norm_threshold: Option<f32>,

    // --- Conditioning ---
    pub conditioning: Vec<Conditioning>,
    /// img2img strength when a single `Reference` is supplied without its own strength.
    pub strength: Option<f32>,
    /// Wan-VACE control strength — the diffusers `conditioning_scale` / per-vace-layer
    /// `control_hidden_states_scale` (`hidden += proj_out(control)·scale`), broadcast to every
    /// `vace_layers` entry. `None` ⇒ the diffusers default `1.0`. Only the `wan_vace` model reads it;
    /// other models ignore it. (sc-3441)
    pub control_scale: Option<f32>,
    /// Krea "text style" gain — reweights the 12 stacked Qwen3-VL select-layer taps before the DiT's
    /// `TextFusionTransformer` aggregates them (the ComfyUI-Conditioning-Rebalance mechanism, sc-8596/
    /// sc-11878). A single scalar `g` maps to the per-layer ramp `w[i] = g + (2−2g)·i/(n−1)`: `g = 1`
    /// (or `None`) is a byte-exact no-op, `g > 1` emphasizes the early (low-level) taps for a
    /// warmer/richer/moodier look, `g < 1` biases toward the late (semantic) taps. GPU-validated safe
    /// over `[0.25, 1.75]` (the engine clamps to that range). **Krea / Qwen-Image-family only** (depends
    /// on the multi-tap text encoder); other models ignore it. It does NOT transfer subject/identity —
    /// it is a stylistic nudge, distinct from the reference-image [`strength`](Self::strength) lever.
    pub text_style_gain: Option<f32>,
    /// Image-guidance (true CFG on the **reference/image** condition) for reference-conditioned edit
    /// models — the identity-strength lever (sc-8273/sc-8278). When `Some(s)` with `s > 1`, the
    /// denoise extrapolates the with-reference velocity against the reference-dropped
    /// (image-unconditional) velocity: `v = v_img0 + s·(v_ref − v_img0)`, pulling output toward the
    /// reference identity *without* pinning composition. `None`/`≤1` ⇒ off (the shipped behavior;
    /// the reference is plain edit conditioning). Today only the FLUX.2 klein/dev **edit** path reads
    /// it (non-kv); other models ignore it. The `FLUX2_IMG_GUIDANCE` env var overrides this (debug).
    pub image_guidance: Option<f32>,

    // --- Video (Option; consumed by video models at the follow-on port) ---
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub duration: Option<f32>,
    pub video_mode: Option<String>,
    /// Generate this many extra leading temporal chunks (each = `vae_stride_t` latent frames) and
    /// discard them after decode, so the first *kept* frame has a full temporal receptive field of
    /// real (non-zero-padded) data — mitigates first-frame VAE/causal-conv artifacts. `None`/0 = off
    /// (the default). Consumed by Wan video models (`generate_wan.py`'s `trim_first_frames`); video
    /// models that don't support it ignore it.
    pub trim_first_frames: Option<u32>,

    // --- SVD image→video micro-conditioning (sc-3523; ignored by other models) ---
    /// SVD `motion_bucket_id` — the motion-strength bucket baked into the `added_time_ids`
    /// micro-conditioning (higher = more motion). `None` ⇒ the model default (127). Only the
    /// `svd_xt` model reads it; other models ignore it.
    pub motion_bucket_id: Option<f32>,
    /// SVD `noise_aug_strength` — Gaussian noise added to the VAE-encoded conditioning image (and
    /// surfaced in `added_time_ids`); higher = less fidelity to the source / more motion. `None` ⇒
    /// the model default (0.02). Only `svd_xt` reads it.
    pub noise_aug_strength: Option<f32>,
    /// Frames decoded per temporal-VAE pass (diffusers `decode_chunk_size`) — a memory/quality knob
    /// for chunked video VAE decode (smaller = less peak memory, changes temporal-boundary
    /// behavior). `None` ⇒ the model default. Only `svd_xt` reads it today.
    pub decode_chunk_size: Option<u32>,
    /// SVD motion **conditioning** fps — the cadence the model was trained on, baked into the
    /// `added_time_ids` micro-conditioning (`fps − 1`); lower ⇒ smoother/slower motion. This is
    /// distinct from [`fps`](Self::fps), which is the output/playback cadence used when muxing the
    /// clip: SVD decouples them (diffusers `StableVideoDiffusionPipeline(fps=…)` vs
    /// `export_to_video(fps=…)`). `None` ⇒ the model default (7). Only `svd_xt` reads it (sc-3764).
    pub conditioning_fps: Option<u32>,

    // --- SeedVR2 super-resolution (sc-4816; ignored by other models) ---
    /// SeedVR2 input **softness** — a pre-blur applied to the bicubic-upscaled low-resolution input
    /// before VAE encode (reference `SeedVR2.generate_image(softness=…)`). Higher = more smoothing of
    /// source compression/noise artifacts before the one-step restoration (trades fine detail for
    /// fewer amplified artifacts on degraded footage). `None`/0.0 ⇒ no pre-blur (the reference
    /// default). Only the `seedvr2` upscaler reads it; other models ignore it.
    pub softness: Option<f32>,

    // --- Prompt enhancement (LTX-2.3 sc-2845 + FLUX.2-dev caption upsampling sc-6030; ignored by
    //     other models) ---
    /// Rewrite `prompt` with an autoregressive LLM before encoding. Default `false` — the diffusion
    /// path is unchanged. On any enhancer failure the model falls back to the original prompt
    /// (reference-faithful). Consumed by: LTX-2.3 (the Gemma-3 `--enhance-prompt`) and FLUX.2-**dev**
    /// (the Mistral3 multimodal `upsample_prompt`, sc-6030 — text-only for T2I, image-conditioned on
    /// the request's reference images for edit; gated like the reference `caption_upsample_temperature`).
    pub enhance_prompt: bool,
    /// Use the separate uncensored 4-bit Gemma enhancer (`--use-uncensored-enhancer`) instead of the
    /// loaded text-encoder backbone. Only consulted when `enhance_prompt` is set. LTX-2.3 only;
    /// FLUX.2-dev ignores it (its upsampler is the loaded Mistral3 tower).
    pub use_uncensored_enhancer: bool,
    /// Max tokens for prompt enhancement (LTX default 512, FLUX.2-dev caption-upsample default 512,
    /// each model's own default when `None`).
    pub enhance_max_tokens: Option<u32>,
    /// Sampling temperature for prompt enhancement (model default when `None`: LTX 0.7, FLUX.2-dev
    /// caption-upsample 0.15 — the reference `caption_upsample_temperature`).
    pub enhance_temperature: Option<f32>,

    // --- Decoder (epic 7840; ignored by models without a PiD backbone) ---
    /// Route this generation's decode through the optional **PiD** super-resolving decoder instead of
    /// the native VAE. Default `false` — the VAE-decode path is unchanged. Only honored when the model
    /// was loaded with [`LoadSpec::pid`](crate::LoadSpec::pid) weights (the PiD-eligible providers,
    /// Qwen-Image / Krea today — sc-7845); a provider with no PiD loaded errors rather than silently
    /// ignoring the request, and PiD-ineligible models ignore the flag. Turning PiD on also changes the
    /// output resolution (native → 4×), so it is not a transparent decoder swap. PiD output is
    /// research/evaluation-only (NSCLv1), surfaced/labeled at the worker/web layer (Phase 3).
    pub use_pid: bool,
    /// PiD **`from_ldm` early-stop** capture σ (epic 7840, sc-7993). Only consulted when
    /// [`use_pid`](Self::use_pid) is set. When `Some(σ)`, stop the denoise as soon as the schedule's
    /// noise level first drops to `≤ σ`, then hand that *partially-denoised* `x_k` to PiD with the
    /// **achieved** degrade σ (`= sigmas[k]`) — the speed optimization that lets the (expensive)
    /// backbone denoise exit early and the 4-step pixel decoder finish the rest. `None`/`≤0` (the
    /// default) = the clean σ=0 path (full denoise, then decode the clean latent). The value is a
    /// noise *ceiling*, schedule-agnostic, so the same σ maps to the right step on an 8-step Turbo and
    /// a 50-step trajectory alike (the policy is [`crate::sampling::flow_capture_plan`]).
    ///
    /// **Frame:** σ is interpreted in the **flow-matching** frame `x_t = (1−σ)x0 + σε` — the path wired
    /// today is the qwenimage latent space (Qwen-Image / Krea / Lightning-Qwen). A latent space whose
    /// PiD student is variance-preserving (SDXL) or whose `from_ldm` wiring is a follow-on errors rather
    /// than silently ignoring the request (see `mlx_gen_pid::resolve_pid_decoder`).
    pub pid_capture_sigma: Option<f32>,

    // --- Audio (Option; consumed by audio models — `Modality::Audio`) ---
    /// The typed audio sub-block (sc-12834). `None` for every image/video request — the top-level
    /// request stays un-bloated, mirroring the planned typed video guider block (§9 known additive
    /// extensions). Audio models read what they support; the shared floor gates the values against
    /// the [`Capabilities`] audio surface. See [`AudioParams`].
    pub audio: Option<AudioParams>,

    // --- Control ---
    pub cancel: CancelFlag,
}

/// The typed audio request sub-block carried by [`GenerationRequest::audio`] (sc-12834). A single
/// `Default`-able struct (no builder), like the request itself: every field is optional so the
/// struct stays **additively extensible** — a later story adds e.g. a multi-speaker script field
/// without breaking `AudioParams { voice: Some(..), ..Default::default() }` construction.
///
/// A model reads what it supports and ignores the rest; the shared validation floor
/// ([`Capabilities::validate_request`] and its size-skipping siblings) rejects values outside the
/// advertised audio surface ([`Capabilities::audio_voices`] / [`audio_languages`](Capabilities::audio_languages)
/// / [`audio_sample_rates`](Capabilities::audio_sample_rates) /
/// [`max_audio_duration_secs`](Capabilities::max_audio_duration_secs)).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioParams {
    /// Voice / speaker id (TTS). Gated by [`Capabilities::audio_voices`] when supplied.
    pub voice: Option<String>,
    /// Language code (e.g. `"en"`). Gated by [`Capabilities::audio_languages`] when supplied.
    pub language: Option<String>,
    /// Requested output duration in seconds. Range-checked against
    /// [`Capabilities::max_audio_duration_secs`] (and the shared duration sanity cap).
    pub target_duration: Option<f32>,
    /// Requested output sample rate in Hz. Gated by [`Capabilities::audio_sample_rates`] when
    /// supplied; `None` ⇒ the model's native rate.
    pub sample_rate: Option<u32>,
    /// Musical tempo in beats per minute (music models). Must be finite and positive.
    pub bpm: Option<f32>,
    /// Musical key (e.g. `"C minor"`; music models). Free-form — each model documents what it
    /// accepts and rejects the rest in its own `validate`.
    pub musical_key: Option<String>,
    /// Lyrics to sing / condition on (music models). Free-form text, distinct from `prompt`.
    pub lyrics: Option<String>,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 1024,
            height: 1024,
            count: 1,
            seed: None,
            steps: None,
            guidance: None,
            true_cfg: None,
            timestep_to_start_cfg: None,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            guidance_method: None,
            guidance_eta: None,
            guidance_momentum: None,
            guidance_norm_threshold: None,
            conditioning: Vec::new(),
            strength: None,
            control_scale: None,
            text_style_gain: None,
            image_guidance: None,
            frames: None,
            fps: None,
            duration: None,
            video_mode: None,
            trim_first_frames: None,
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            use_pid: false,
            pid_capture_sigma: None,
            audio: None,
            cancel: CancelFlag::default(),
        }
    }
}

/// A first_last_frame / multi-keyframe input — a borrowed, normalized view of a
/// [`Conditioning::Keyframe`]. Returned by [`GenerationRequest::keyframes`].
#[derive(Clone, Copy, Debug)]
pub struct KeyframeRef<'a> {
    pub image: &'a Image,
    pub frame_idx: i32,
    pub strength: f32,
}

/// An in-context conditioning clip — a borrowed view of a [`Conditioning::VideoClip`]. Returned by
/// [`GenerationRequest::video_clips`].
#[derive(Clone, Copy, Debug)]
pub struct VideoClipRef<'a> {
    pub frames: &'a [Image],
    pub frame_idx: i32,
    pub strength: f32,
}

/// A replace_person masked control clip — a borrowed view of a [`Conditioning::ControlClip`].
/// Returned by [`GenerationRequest::control_clip`].
#[derive(Clone, Copy, Debug)]
pub struct ControlClipRef<'a> {
    pub frames: &'a [Image],
    pub mask: &'a [Image],
    pub masking_strength: f32,
    pub start_frame: i32,
    pub mode: ReplacementMode,
}

impl GenerationRequest {
    /// The first request-supplied `Option<f32>` knob that is **non-finite** (NaN / ±Inf), returned as
    /// `(field, value)` — or `None` when every present float is finite. This is the single home of the
    /// finiteness floor (F-053 / F-001): a NaN/Inf on *any* float knob flows into the guidance /
    /// scheduler / conditioning math and silently poisons the whole denoise (garbage-as-success, no
    /// error), so it must be rejected at the contract boundary.
    ///
    /// The exhaustive destructuring below (no `..`) is deliberate and load-bearing: adding a field to
    /// [`GenerationRequest`] fails to compile *here*, forcing the author to decide whether the new knob
    /// is a float that must join the guard. New `Option<f32>` fields therefore inherit the finiteness
    /// check **by construction** instead of silently slipping past it — the recurring "the floor lags
    /// the request surface" regression this method exists to close.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            // Non-float fields: explicitly ignored, but named (no `..`) so a newly-added field breaks
            // the build here and the author must classify it.
            prompt: _,
            negative_prompt: _,
            width: _,
            height: _,
            count: _,
            seed: _,
            steps: _,
            timestep_to_start_cfg: _,
            sampler: _,
            scheduler: _,
            guidance_method: _,
            conditioning,
            frames: _,
            fps: _,
            video_mode: _,
            trim_first_frames: _,
            decode_chunk_size: _,
            conditioning_fps: _,
            enhance_prompt: _,
            use_uncensored_enhancer: _,
            enhance_max_tokens: _,
            use_pid: _,
            cancel: _,
            // The audio sub-block carries its own floats — destructured below the flat knobs.
            audio,
            // Every `Option<f32>` knob the floor owns.
            guidance,
            true_cfg,
            scheduler_shift,
            guidance_eta,
            guidance_momentum,
            guidance_norm_threshold,
            strength,
            control_scale,
            image_guidance,
            duration,
            motion_bucket_id,
            noise_aug_strength,
            softness,
            enhance_temperature,
            pid_capture_sigma,
            text_style_gain,
        } = self;
        let floats: [(&'static str, Option<f32>); 16] = [
            ("guidance", *guidance),
            ("true_cfg", *true_cfg),
            ("scheduler_shift", *scheduler_shift),
            ("guidance_eta", *guidance_eta),
            ("guidance_momentum", *guidance_momentum),
            ("guidance_norm_threshold", *guidance_norm_threshold),
            ("strength", *strength),
            ("control_scale", *control_scale),
            ("image_guidance", *image_guidance),
            ("duration", *duration),
            ("motion_bucket_id", *motion_bucket_id),
            ("noise_aug_strength", *noise_aug_strength),
            ("softness", *softness),
            ("enhance_temperature", *enhance_temperature),
            ("pid_capture_sigma", *pid_capture_sigma),
            ("text_style_gain", *text_style_gain),
        ];
        for (name, v) in floats {
            if let Some(x) = v {
                if !x.is_finite() {
                    return Some((name, x));
                }
            }
        }
        // Audio sub-block floats (sc-12834): destructured exhaustively (no `..`) for the same
        // reason as the request itself — a new `AudioParams` float field fails to compile here
        // until it is classified into the floor.
        if let Some(AudioParams {
            voice: _,
            language: _,
            sample_rate: _,
            musical_key: _,
            lyrics: _,
            target_duration,
            bpm,
        }) = audio
        {
            let audio_floats: [(&'static str, Option<f32>); 2] = [
                ("audio.target_duration", *target_duration),
                ("audio.bpm", *bpm),
            ];
            for (name, v) in audio_floats {
                if let Some(x) = v {
                    if !x.is_finite() {
                        return Some((name, x));
                    }
                }
            }
        }
        // Conditioning-carried floats the floor also owns (F-001): the Control-branch scale and the
        // per-Reference img2img strength both flow into the same denoise/scheduler math.
        for c in conditioning {
            match c {
                Conditioning::Control { scale: Some(s), .. } if !s.is_finite() => {
                    return Some(("conditioning.control.scale", *s));
                }
                Conditioning::Reference {
                    strength: Some(s), ..
                } if !s.is_finite() => {
                    return Some(("conditioning.reference.strength", *s));
                }
                Conditioning::ReferenceAudio {
                    strength: Some(s), ..
                } if !s.is_finite() => {
                    return Some(("conditioning.reference_audio.strength", *s));
                }
                _ => {}
            }
        }
        None
    }

    /// Reject the request when any `Option<f32>` knob is non-finite (see
    /// [`first_nonfinite_float`](Self::first_nonfinite_float)). The shared home of the F-053 / F-001
    /// finiteness floor: [`Capabilities::validate_request`] calls it, and providers with a bespoke
    /// `validate` (e.g. flux1's IP-Adapter carve-out) call it directly so they inherit the guard too.
    pub fn ensure_finite_floats(&self) -> Result<()> {
        if let Some((field, value)) = self.first_nonfinite_float() {
            return Err(Error::Msg(format!("{field} must be finite (got {value})")));
        }
        Ok(())
    }

    /// All [`Conditioning::Keyframe`] inputs (first_last_frame / multi-keyframe), in request order.
    pub fn keyframes(&self) -> Vec<KeyframeRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::Keyframe {
                    image,
                    frame_idx,
                    strength,
                } => Some(KeyframeRef {
                    image,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// All [`Conditioning::VideoClip`] in-context clips (extend_clip / video_bridge), in request order.
    pub fn video_clips(&self) -> Vec<VideoClipRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::VideoClip {
                    frames,
                    frame_idx,
                    strength,
                } => Some(VideoClipRef {
                    frames,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// The replace_person masked control clip ([`Conditioning::ControlClip`]), if present. The first
    /// one wins (a request carries at most one person edit per generation).
    pub fn control_clip(&self) -> Option<ControlClipRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode,
            } => Some(ControlClipRef {
                frames,
                mask,
                masking_strength: *masking_strength,
                start_frame: *start_frame,
                mode: *mode,
            }),
            _ => None,
        })
    }
}

/// Seed when a [`GenerationRequest`] omits one: nanos since the epoch (any nonzero value works —
/// this only sets which sample is drawn; a caller wanting reproducibility passes `req.seed`).
/// Shared by every generator (F-006).
pub fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        // Fall back to a nonzero value: 0 is the "no seed" sentinel a caller would pass to mean
        // "pick one", so the default must never itself be 0 (F-089).
        .unwrap_or(1)
}

/// Typed conditioning inputs. Each image family uses the subset its `Capabilities` advertises.
///
/// The video families ([`Conditioning::Keyframe`] / [`Conditioning::VideoClip`] /
/// [`Conditioning::ControlClip`]) are the epic-3040 advanced-mode inputs and map onto the two LTX
/// conditioning mechanisms (see `docs/SPIKE_ADVANCED_VIDEO_3040.md`): a [`Keyframe`](Conditioning::Keyframe)
/// is **replace-latent** (overwrite the target latent at a frame index — first_last_frame); a
/// [`VideoClip`](Conditioning::VideoClip) / [`ControlClip`](Conditioning::ControlClip) is
/// **keyframe-append** (append the clip's VAE latents as extra in-context tokens — extend_clip /
/// video_bridge / replace_person, the IC-LoRA path).
#[derive(Clone, Debug)]
pub enum Conditioning {
    /// img2img / IP-Adapter / identity reference.
    Reference { image: Image, strength: Option<f32> },
    /// A reference **audio** clip — voice cloning / style reference for audio models
    /// (sc-12834; the audio analogue of [`Conditioning::Reference`]). `strength` mirrors the
    /// per-reference img2img strength: `None` ⇒ the model default. Video→audio conditioning
    /// later reuses [`Conditioning::VideoClip`] — this variant is audio-in only.
    ReferenceAudio {
        audio: AudioTrack,
        strength: Option<f32>,
    },
    /// Multiple references with no per-image strength (Qwen-Image-Edit).
    MultiReference { images: Vec<Image> },
    /// FLUX.1-Redux references, each with its own strength.
    ReduxRefs { refs: Vec<(Image, f32)> },
    /// ControlNet / pose conditioning. `scale` mirrors the strength on
    /// [`Conditioning::Reference`]: `None` means "use the
    /// per-model default control scale" and `Some(x)` is an explicit override — including `Some(0.0)`,
    /// a deliberately inert control branch. The `Option` is what distinguishes explicit-inert from
    /// unset (the old bare `f32` could not; F-085).
    Control {
        image: Image,
        kind: ControlKind,
        scale: Option<f32>,
    },
    /// FLUX.1-Depth.
    Depth { image: Image },
    /// FIBO-Edit / inpaint mask.
    Mask { image: Image },
    /// A keyframe pinned at a specific output **latent** frame index (first_last_frame / general
    /// multi-keyframe). VAE-encoded and its tokens **overwrite** the target latent at `frame_idx`
    /// with denoise mask `1 − strength` (the replace-latent mechanism — reference
    /// `VideoConditionByLatentIndex`). `strength = 1.0` fully pins the frame. first_last_frame is two
    /// of these (at `0` and the last latent frame).
    Keyframe {
        image: Image,
        frame_idx: i32,
        strength: f32,
    },
    /// An in-context conditioning **clip** (extend_clip / video_bridge — the LTX IC-LoRA path). The
    /// frames are VAE-encoded and **appended** as extra tokens at `frame_idx` (RoPE-offset on the
    /// frame axis) with denoise mask `1 − strength` (reference `VideoConditionByKeyframeIndex`).
    /// extend_clip = one clip at `frame_idx 0`; video_bridge = a left clip at `0` and a right clip at
    /// the tail.
    VideoClip {
        frames: Vec<Image>,
        frame_idx: i32,
        strength: f32,
    },
    /// A masked control clip for replace_person. `frames` is the (host-built, person-region
    /// neutralized) control clip; `mask` is the per-frame binary person mask (white = regenerate).
    /// Drives the keyframe-append in-context conditioning **plus** mask injection (force the masked
    /// region toward the re-noised source for the first `ceil(steps · masking_strength)` steps —
    /// reference `prepare_mask_injection`). Person detect/track stays in onnx and supplies these.
    ControlClip {
        frames: Vec<Image>,
        mask: Vec<Image>,
        masking_strength: f32,
        /// Output latent-frame the control clip aligns to (reference `masking_source.start_frame`).
        start_frame: i32,
        /// Replacement granularity (reference `replacement_mode`); the LTX mask path is region-driven
        /// so it is carried for the worker contract / WanVACE parity rather than changing the mask math.
        mode: ReplacementMode,
    },
}

impl Conditioning {
    /// The [`ConditioningKind`] discriminant — for capability checks / `validate()`. Centralized here
    /// so adding a [`Conditioning`] variant updates every model's validation in one place.
    pub fn kind(&self) -> ConditioningKind {
        match self {
            Conditioning::Reference { .. } => ConditioningKind::Reference,
            Conditioning::ReferenceAudio { .. } => ConditioningKind::ReferenceAudio,
            Conditioning::MultiReference { .. } => ConditioningKind::MultiReference,
            Conditioning::ReduxRefs { .. } => ConditioningKind::ReduxRefs,
            Conditioning::Control { .. } => ConditioningKind::Control,
            Conditioning::Depth { .. } => ConditioningKind::Depth,
            Conditioning::Mask { .. } => ConditioningKind::Mask,
            Conditioning::Keyframe { .. } => ConditioningKind::Keyframe,
            Conditioning::VideoClip { .. } => ConditioningKind::VideoClip,
            Conditioning::ControlClip { .. } => ConditioningKind::ControlClip,
        }
    }
}

/// Granularity of a replace_person edit (reference `replacement_mode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplacementMode {
    /// Replace the face region only.
    #[default]
    FaceOnly,
    /// Replace the full person but keep the original outfit.
    FullPersonKeepOutfit,
    /// Replace the full person including the outfit.
    FullPersonReplaceOutfit,
}

/// The control signal carried by [`Conditioning::Control`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Pose,
    Canny,
    Depth,
    Other(String),
}

/// Which [`Conditioning`] variants a model accepts — for capability introspection + validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditioningKind {
    Reference,
    /// Voice/style reference audio ([`Conditioning::ReferenceAudio`]).
    ReferenceAudio,
    MultiReference,
    ReduxRefs,
    Control,
    Depth,
    Mask,
    /// first_last_frame / multi-keyframe ([`Conditioning::Keyframe`]).
    Keyframe,
    /// extend_clip / video_bridge ([`Conditioning::VideoClip`]).
    VideoClip,
    /// replace_person ([`Conditioning::ControlClip`]).
    ControlClip,
}

/// What kind of media a model emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Image,
    Video,
    /// Emits both image and video (e.g. the SeedVR2 upscaler over either).
    Both,
    /// Pure audio synthesis — TTS / music (sc-12834). Emits [`GenerationOutput::Audio`];
    /// `width`/`height` are unused, so audio models validate through the size-skipping floor
    /// ([`Capabilities::validate_request_audio`]).
    Audio,
}

/// A model's stable identity + advertised capabilities. Returned by `descriptor()` and also
/// constructible without loading weights (registry introspection).
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    /// `"mlx"` | `"candle"` — the tensor backend whose provider crate registered this engine.
    /// Drives the worker's registry-derived capability advertisement (sc-3723); MLX families set
    /// `"mlx"`.
    pub backend: &'static str,
    pub modality: Modality,
    pub capabilities: Capabilities,
}

/// What a model supports — drives `validate()` and consumer UI. `Default` is "supports
/// nothing"; a model turns on what it offers (`Capabilities { supports_guidance: true,
/// ..Default::default() }`).
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub supports_negative_prompt: bool,
    pub supports_guidance: bool,
    pub supports_true_cfg: bool,
    pub conditioning: Vec<ConditioningKind>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub samplers: Vec<&'static str>,
    pub schedulers: Vec<&'static str>,
    /// The guidance methods this model+backend honors (epic 7434), e.g. `["cfg", "cfg_rescale"]`.
    /// Empty ⇒ only the engine's implicit default path (no selectable guidance axis). Per-model-
    /// per-backend, like [`samplers`](Self::samplers) / [`schedulers`](Self::schedulers).
    pub supported_guidance_methods: Vec<&'static str>,
    pub min_size: u32,
    pub max_size: u32,
    pub max_count: u32,
    pub mac_only: bool,
    // Audio surface (sc-12834) — read by the floor when a request carries
    // [`GenerationRequest::audio`]; all `Default` to the empty/no-audio surface so image/video
    // descriptors are untouched.
    /// Output sample rates (Hz) this model can synthesize. Empty ⇒ no selectable sample-rate
    /// surface: an explicit `audio.sample_rate` is rejected as [`Error::Unsupported`] (the
    /// same convention as [`samplers`](Self::samplers)); `None` on the request always passes
    /// (the model's native rate).
    pub audio_sample_rates: Vec<u32>,
    /// Longest audio clip (seconds) this model synthesizes. `None` ⇒ no advertised cap — an
    /// `audio.target_duration` is then bounded only by the shared duration sanity cap.
    pub max_audio_duration_secs: Option<f32>,
    /// Voice / speaker ids this model offers (TTS). Empty ⇒ no selectable voice surface: an
    /// explicit `audio.voice` is rejected as [`Error::Unsupported`].
    pub audio_voices: Vec<&'static str>,
    /// Language codes this model supports. Empty ⇒ no selectable language surface: an explicit
    /// `audio.language` is rejected as [`Error::Unsupported`].
    pub audio_languages: Vec<&'static str>,
    /// On-the-fly quantization levels this engine offers (empty slice = none). Read by the worker's
    /// capability advertisement (sc-3723) instead of a hardcoded per-row flag. `Default` is `&[]`.
    pub supported_quants: &'static [Quant],
    // Loader hints.
    pub supports_kv_cache: bool,
    pub requires_sigma_shift: bool,
    /// Whether this engine honors [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
    /// (epic 10765, sc-11126). [`crate::OffloadPolicy::Sequential`] is *advisory* — a provider that
    /// has not
    /// wired the load→use→drop residency lifecycle silently treats it as `Resident` (never an error),
    /// which makes the fallback undiscoverable from the outside. This bit is the discovery signal: a
    /// consumer (worker / UI) reads it to know whether requesting `Sequential` will actually bound peak
    /// **footprint** on this engine or be a no-op. "Bound peak footprint" covers both shapes: an engine
    /// that holds several components co-resident (e.g. the Wan A14B MoE) bounds the peak **active** set
    /// by dropping the inactive ones, while an engine that already stages its active set (e.g. the dense
    /// Wan TI2V-5B) bounds the peak **retained cache / RSS** by `clear_cache`-flushing each dead
    /// component off-GPU instead of leaving it warm — the fit-gate models both via the staged
    /// max-single-component estimate. `Default` is `false` so an unwired engine does not over-advertise;
    /// a provider that drives the shared [`crate::runtime`] residency seam sets it `true`.
    pub supports_sequential_offload: bool,
}

/// Generous upper sanity caps for the unbounded counter knobs (F-004). Not model limits — each model
/// layers a tighter, better-messaged bound in its own `validate` (e.g. kolors caps `steps` at its
/// train-timestep count); these sit ABOVE any real model bound so they only reject a pathological
/// value (`u32::MAX` steps/frames) that would otherwise launch an effectively-unbounded, cancel-only
/// run — never preempting a model's own check.
const MAX_STEPS: u32 = 100_000;
const MAX_FRAMES: u32 = 1_000_000;
const MAX_FPS: u32 = 100_000;
const MAX_DURATION_SECS: f32 = 1_000_000.0;

impl Capabilities {
    /// Whether this model accepts the given conditioning kind.
    pub fn accepts(&self, kind: ConditioningKind) -> bool {
        self.conditioning.contains(&kind)
    }

    /// Reject a request that violates the **advertised** capability surface — the model-agnostic
    /// checks every `Generator::validate` shares, so a descriptor cannot promise something
    /// `validate` then silently ignores at runtime:
    ///
    /// - `count` within `1..=max_count`,
    /// - `steps` (when supplied) must be `>= 1` — an explicit `0` would run a 0-step denoise and
    ///   VAE-decode pure noise (F-007); the schedule builders' `.max(1)` clamps document this as the
    ///   real floor,
    /// - `width`/`height` within `min_size..=max_size`,
    /// - `negative_prompt` / `guidance` / `true_cfg` only when the matching `supports_*` flag is set,
    ///   and `guidance` / `true_cfg` must be finite (a NaN would poison the guidance math, F-053),
    /// - `sampler` / `scheduler` / `guidance_method` (when supplied) must name an advertised entry,
    /// - every `conditioning` entry must be an [`accepts`](Self::accepts)ed kind,
    /// - the [`audio`](GenerationRequest::audio) sub-block's supplied values must sit inside the
    ///   advertised audio surface (voice / language / sample-rate membership,
    ///   `target_duration` within `(0, `[`max_audio_duration_secs`](Self::max_audio_duration_secs)`]`,
    ///   positive `bpm` — sc-12834).
    ///
    /// Capability-gap rejections (unsupported negative_prompt / guidance / true_cfg / sampler /
    /// scheduler / guidance_method / conditioning) return the typed [`Error::Unsupported`] so a
    /// consumer (SceneWorks worker / candle gating) can distinguish "this backend can't do that"
    /// from a range violation or generic failure (F-008); malformed-value rejections (count/size/
    /// steps out of range, non-finite guidance) return [`Error::Msg`].
    ///
    /// `id` is the model's descriptor id, used in error messages. Model-specific constraints — an
    /// empty-prompt rejection, size-alignment (multiple-of-N), frame-count divisibility,
    /// sampler→solver mapping — are layered on top by each model's own `validate`; this is the shared
    /// floor, not a replacement for them.
    pub fn validate_request(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        self.validate_request_inner(id, req, true)
    }

    /// The shared floor **minus the size-range check** — for providers with a "match the driving-media
    /// size" convention (`width`/`height == 0` is a resolve-downstream sentinel, e.g. SCAIL-2 sizing
    /// from the driving-video frames), where the size range would wrongly reject the sentinel. Every
    /// other floor check still runs unconditionally: count / steps / frame / fps / duration caps,
    /// negative-prompt / guidance / true_cfg support gating, finiteness (F-053), sampler / scheduler /
    /// guidance_method membership, and the conditioning allowlist. A provider that calls this must
    /// range-check its resolved size itself (F-158).
    pub fn validate_request_skip_size(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        self.validate_request_inner(id, req, false)
    }

    /// The audio-aware floor (sc-12834): the shared floor **minus the width/height range check**,
    /// for pure-audio models (`Modality::Audio`) where the request's `width`/`height` are unused
    /// and the visual size range would wrongly reject every request. Parallel to
    /// [`validate_request_skip_size`](Self::validate_request_skip_size) — every other floor check
    /// still runs unconditionally: count / steps / frame / fps / duration caps, capability gating,
    /// finiteness (F-053, including the [`AudioParams`] floats), sampler / scheduler /
    /// guidance_method membership, the conditioning allowlist, and the audio-surface checks
    /// (voice / language / sample-rate membership, `target_duration` vs
    /// [`max_audio_duration_secs`](Self::max_audio_duration_secs)).
    pub fn validate_request_audio(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        self.validate_request_inner(id, req, false)
    }

    /// Shared implementation of the floor. `check_size` gates only the size-range check so the
    /// auto-size path ([`validate_request_skip_size`](Self::validate_request_skip_size)) still runs
    /// every other check; the public [`validate_request`](Self::validate_request) passes `true`.
    fn validate_request_inner(
        &self,
        id: &str,
        req: &GenerationRequest,
        check_size: bool,
    ) -> Result<()> {
        // Footgun guard (F-084): a descriptor that enables a capability but leaves max_count/max_size
        // at the `Default` 0 would reject EVERY request with a confusing "out of range 0..=0". A real
        // model always sets non-zero bounds, so catch the descriptor mistake in debug/test builds.
        // `max_size` is only asserted when the size check runs: on the size-skipping floors the size
        // bounds are legitimately unused (a pure-audio descriptor leaves them at 0 — sc-12834).
        debug_assert!(
            self.max_count > 0 && (!check_size || self.max_size > 0),
            "{id}: Capabilities max_count={} max_size={} left at Default 0 — descriptor forgot its bounds",
            self.max_count,
            self.max_size
        );
        if req.count == 0 || req.count > self.max_count {
            return Err(Error::Msg(format!(
                "{id}: count {} out of range 1..={}",
                req.count, self.max_count
            )));
        }
        // An explicit `steps: Some(0)` runs a 0-step denoise and VAE-decodes pure scaled noise; the
        // schedule builders' `.max(1)` clamps (sampling.rs) cite this as the real floor, so enforce it
        // here rather than letting it fall through to ad-hoc per-provider guards (F-007). `None` falls
        // back to each model's default; a *derived* 0 from img2img `int(steps·strength)` is a separate,
        // legitimate no-op handled downstream.
        if req.steps == Some(0) {
            return Err(Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
        }
        // Upper sanity caps (F-004): the floor enforced `steps >= 1` but no ceiling, so
        // `steps: Some(u32::MAX)` (and the video frame/counter fields) validated and launched an
        // effectively-unbounded, cancel-only-recoverable run. These are deliberately generous — far
        // above any real request (LTX's frame ceiling is 1025, the priciest image trajectories ~50–100
        // steps) — a footgun guard against a pathological/garbage value, not a model-specific limit
        // (each model layers its own tighter bound in its `validate`).
        if let Some(steps) = req.steps {
            if steps > MAX_STEPS {
                return Err(Error::Msg(format!(
                    "{id}: steps {steps} exceeds the sanity cap {MAX_STEPS}"
                )));
            }
        }
        if let Some(frames) = req.frames {
            if frames > MAX_FRAMES {
                return Err(Error::Msg(format!(
                    "{id}: frames {frames} exceeds the sanity cap {MAX_FRAMES}"
                )));
            }
        }
        if let Some(fps) = req.fps {
            if fps > MAX_FPS {
                return Err(Error::Msg(format!(
                    "{id}: fps {fps} exceeds the sanity cap {MAX_FPS}"
                )));
            }
        }
        if let Some(d) = req.duration {
            // `d` is finiteness-checked below; here only the upper bound (a NaN compares false and is
            // caught by `ensure_finite_floats`).
            if d > MAX_DURATION_SECS {
                return Err(Error::Msg(format!(
                    "{id}: duration {d}s exceeds the sanity cap {MAX_DURATION_SECS}s"
                )));
            }
        }
        // Audio sub-block (sc-12834): gate the supplied values against the advertised audio
        // surface. Membership gaps (voice / language / sample rate) are capability gaps →
        // typed `Error::Unsupported` (F-008); malformed values (non-positive / over-cap
        // duration, non-positive bpm) are range errors → `Error::Msg`. Finiteness of the audio
        // floats is enforced by `ensure_finite_floats` below (a NaN compares false here and
        // falls through to that guard, like `duration`).
        if let Some(audio) = &req.audio {
            if let Some(d) = audio.target_duration {
                if d <= 0.0 {
                    return Err(Error::Msg(format!(
                        "{id}: audio.target_duration must be > 0 (got {d}s)"
                    )));
                }
                if d > MAX_DURATION_SECS {
                    return Err(Error::Msg(format!(
                        "{id}: audio.target_duration {d}s exceeds the sanity cap {MAX_DURATION_SECS}s"
                    )));
                }
                if let Some(cap) = self.max_audio_duration_secs {
                    if d > cap {
                        return Err(Error::Msg(format!(
                            "{id}: audio.target_duration {d}s exceeds the supported maximum {cap}s"
                        )));
                    }
                }
            }
            if let Some(bpm) = audio.bpm {
                if bpm <= 0.0 {
                    return Err(Error::Msg(format!(
                        "{id}: audio.bpm must be > 0 (got {bpm})"
                    )));
                }
            }
            if let Some(sr) = audio.sample_rate {
                if !self.audio_sample_rates.contains(&sr) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio.sample_rate {sr} (supported: {:?})",
                        self.audio_sample_rates
                    )));
                }
            }
            if let Some(v) = &audio.voice {
                if !self.audio_voices.contains(&v.as_str()) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio.voice {v:?} (supported: {:?})",
                        self.audio_voices
                    )));
                }
            }
            if let Some(l) = &audio.language {
                if !self.audio_languages.contains(&l.as_str()) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio.language {l:?} (supported: {:?})",
                        self.audio_languages
                    )));
                }
            }
        }
        if check_size
            && (req.width < self.min_size
                || req.height < self.min_size
                || req.width > self.max_size
                || req.height > self.max_size)
        {
            return Err(Error::Msg(format!(
                "{id}: size {}x{} outside supported range {}..={}",
                req.width, req.height, self.min_size, self.max_size
            )));
        }
        if req.negative_prompt.is_some() && !self.supports_negative_prompt {
            return Err(Error::Unsupported(format!(
                "{id}: negative prompts are not supported"
            )));
        }
        if req.guidance.is_some() && !self.supports_guidance {
            return Err(Error::Unsupported(format!(
                "{id}: guidance is not supported"
            )));
        }
        if req.true_cfg.is_some() && !self.supports_true_cfg {
            return Err(Error::Unsupported(format!(
                "{id}: true_cfg is not supported"
            )));
        }
        // A non-finite guidance / true_cfg / eta / momentum / strength / control_scale / … would flow
        // into the CFG combine, scheduler shift, or conditioning math and NaN-poison the run (a NaN
        // passes `x > 1.0`-style checks silently). The finiteness guard is centralized on the request
        // so every `Option<f32>` knob — including ones added after F-053 — inherits it by construction
        // (F-053 / F-001). `id`-prefixing is dropped from the message here; the field name is enough.
        req.ensure_finite_floats()?;
        if let Some(s) = &req.sampler {
            if !self.samplers.contains(&s.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported sampler {s:?} (supported: {:?})",
                    self.samplers
                )));
            }
        }
        if let Some(s) = &req.scheduler {
            if !self.schedulers.contains(&s.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported scheduler {s:?} (supported: {:?})",
                    self.schedulers
                )));
            }
        }
        if let Some(m) = &req.guidance_method {
            if !self.supported_guidance_methods.contains(&m.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported guidance_method {m:?} (supported: {:?})",
                    self.supported_guidance_methods
                )));
            }
        }
        for c in &req.conditioning {
            let kind = c.kind();
            if !self.accepts(kind) {
                return Err(Error::Unsupported(format!(
                    "{id}: {kind:?} conditioning is not supported"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn keyframes_accessor_collects_in_order() {
        // first_last_frame: two keyframes at 0 and the last latent frame.
        let req = GenerationRequest {
            conditioning: vec![
                Conditioning::Keyframe {
                    image: img(2, 2),
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::Reference {
                    image: img(2, 2),
                    strength: None,
                },
                Conditioning::Keyframe {
                    image: img(4, 4),
                    frame_idx: 8,
                    strength: 0.75,
                },
            ],
            ..Default::default()
        };
        let kf = req.keyframes();
        assert_eq!(kf.len(), 2);
        assert_eq!((kf[0].frame_idx, kf[0].strength), (0, 1.0));
        assert_eq!((kf[1].frame_idx, kf[1].strength), (8, 0.75));
        assert_eq!((kf[1].image.width, kf[1].image.height), (4, 4));
        // Reference is not a keyframe and is not a video clip / control clip.
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
    }

    #[test]
    fn video_clips_accessor_collects_clips() {
        // video_bridge: left clip @0, right clip @tail.
        let req = GenerationRequest {
            conditioning: vec![
                Conditioning::VideoClip {
                    frames: vec![img(2, 2), img(2, 2)],
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::VideoClip {
                    frames: vec![img(2, 2)],
                    frame_idx: 24,
                    strength: 0.9,
                },
            ],
            ..Default::default()
        };
        let clips = req.video_clips();
        assert_eq!(clips.len(), 2);
        assert_eq!((clips[0].frames.len(), clips[0].frame_idx), (2, 0));
        assert_eq!((clips[1].frames.len(), clips[1].frame_idx), (1, 24));
        assert!(req.keyframes().is_empty());
    }

    #[test]
    fn control_clip_accessor_returns_first() {
        let req = GenerationRequest {
            conditioning: vec![Conditioning::ControlClip {
                frames: vec![img(2, 2), img(2, 2)],
                mask: vec![img(2, 2), img(2, 2)],
                masking_strength: 0.8,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            }],
            ..Default::default()
        };
        let cc = req.control_clip().expect("control clip present");
        assert_eq!((cc.frames.len(), cc.mask.len()), (2, 2));
        assert_eq!(cc.masking_strength, 0.8);
        assert_eq!(cc.mode, ReplacementMode::FaceOnly);
    }

    #[test]
    fn accessors_empty_by_default() {
        let req = GenerationRequest::default();
        assert!(req.keyframes().is_empty());
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
    }

    /// A capability surface that turns nothing extra on: a single 256..=1024 image, no
    /// negative/guidance/true_cfg, no samplers/schedulers, only `Reference` conditioning.
    fn caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::Reference],
            samplers: vec!["euler"],
            min_size: 256,
            max_size: 1024,
            max_count: 1,
            ..Default::default()
        }
    }

    fn base_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            ..Default::default()
        }
    }

    #[test]
    fn validate_request_accepts_in_surface() {
        let c = caps();
        assert!(c.validate_request("m", &base_req()).is_ok());
        // An advertised sampler + an accepted conditioning kind are fine.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    sampler: Some("euler".into()),
                    conditioning: vec![Conditioning::Reference {
                        image: img(8, 8),
                        strength: None,
                    }],
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn validate_request_enforces_advertised_surface() {
        let c = caps();
        let cases: Vec<GenerationRequest> = vec![
            // count out of range
            GenerationRequest {
                count: 0,
                ..base_req()
            },
            GenerationRequest {
                count: 2,
                ..base_req()
            },
            // size out of range (below min, above max)
            GenerationRequest {
                width: 128,
                ..base_req()
            },
            GenerationRequest {
                height: 2048,
                ..base_req()
            },
            // capability flags not advertised
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..base_req()
            },
            GenerationRequest {
                true_cfg: Some(4.0),
                ..base_req()
            },
            // sampler / scheduler not advertised
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..base_req()
            },
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..base_req()
            },
            // conditioning kind not accepted
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..base_req()
            },
        ];
        for (i, req) in cases.iter().enumerate() {
            assert!(
                c.validate_request("m", req).is_err(),
                "case {i} should have been rejected"
            );
        }
    }

    #[test]
    fn validate_request_skip_size_runs_every_non_size_check() {
        // F-158: the auto-size floor. `validate_request_skip_size` must still enforce the whole floor
        // *except* the size range — so a sentinel `0x0` size passes, but an out-of-surface count /
        // sampler / conditioning / non-finite knob is still rejected. Used by providers that resolve
        // size downstream (SCAIL-2 sizing from the driving video).
        let c = caps();
        // Auto-size sentinel (0x0) is accepted where the full floor would reject it for being < min.
        let auto = GenerationRequest {
            width: 0,
            height: 0,
            ..base_req()
        };
        assert!(
            c.validate_request("m", &auto).is_err(),
            "size 0x0 is below min for the full floor"
        );
        assert!(
            c.validate_request_skip_size("m", &auto).is_ok(),
            "skip_size must accept the 0x0 auto-size sentinel"
        );
        // Every non-size violation must still fire on the auto-size path.
        let rejected: Vec<GenerationRequest> = vec![
            // oversized count
            GenerationRequest {
                count: 2,
                ..auto.clone()
            },
            // explicit zero steps
            GenerationRequest {
                steps: Some(0),
                ..auto.clone()
            },
            // unadvertised sampler
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..auto.clone()
            },
            // disallowed conditioning kind
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..auto.clone()
            },
            // non-finite knob (not support-gated) would NaN-poison the run — finiteness still fires
            GenerationRequest {
                strength: Some(f32::NAN),
                ..auto.clone()
            },
        ];
        for (i, req) in rejected.iter().enumerate() {
            assert!(
                c.validate_request_skip_size("m", req).is_err(),
                "skip_size case {i} should have been rejected on the auto-size path"
            );
        }
    }

    #[test]
    fn validate_request_rejects_explicit_zero_steps() {
        // F-007: the floor now enforces the steps>=1 claim the schedule builders rely on.
        let c = caps();
        let bad = GenerationRequest {
            steps: Some(0),
            ..base_req()
        };
        let err = c.validate_request("m", &bad).unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "steps=0 is a range error");
        assert!(err.to_string().contains("steps must be >= 1"));
        // `None` and a positive count still pass.
        assert!(c.validate_request("m", &base_req()).is_ok());
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    steps: Some(1),
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn non_finite_extended_float_knobs_are_rejected() {
        // F-001: the finiteness floor now covers every `Option<f32>` knob added after F-053, not just
        // guidance/true_cfg. A guidance-capable caps so the support gate never fires; each knob is
        // exercised with NaN/±Inf and must produce a typed `Msg` naming the field.
        let c = Capabilities {
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Control],
            ..caps()
        };
        type Build = fn(f32) -> GenerationRequest;
        let mk: [(&str, Build); 12] = [
            ("guidance_eta", |v| GenerationRequest {
                guidance_eta: Some(v),
                ..base_req()
            }),
            ("guidance_momentum", |v| GenerationRequest {
                guidance_momentum: Some(v),
                ..base_req()
            }),
            ("guidance_norm_threshold", |v| GenerationRequest {
                guidance_norm_threshold: Some(v),
                ..base_req()
            }),
            ("strength", |v| GenerationRequest {
                strength: Some(v),
                ..base_req()
            }),
            ("control_scale", |v| GenerationRequest {
                control_scale: Some(v),
                ..base_req()
            }),
            ("image_guidance", |v| GenerationRequest {
                image_guidance: Some(v),
                ..base_req()
            }),
            ("scheduler_shift", |v| GenerationRequest {
                scheduler_shift: Some(v),
                ..base_req()
            }),
            ("motion_bucket_id", |v| GenerationRequest {
                motion_bucket_id: Some(v),
                ..base_req()
            }),
            ("noise_aug_strength", |v| GenerationRequest {
                noise_aug_strength: Some(v),
                ..base_req()
            }),
            ("softness", |v| GenerationRequest {
                softness: Some(v),
                ..base_req()
            }),
            ("enhance_temperature", |v| GenerationRequest {
                enhance_temperature: Some(v),
                ..base_req()
            }),
            ("pid_capture_sigma", |v| GenerationRequest {
                pid_capture_sigma: Some(v),
                ..base_req()
            }),
        ];
        for (field, build) in mk {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let req = build(bad);
                let err = c.validate_request("m", &req).unwrap_err();
                assert!(matches!(err, Error::Msg(_)), "{field} {bad} → Msg");
                assert!(
                    err.to_string().contains(field) && err.to_string().contains("must be finite"),
                    "{field} {bad}: got {err}"
                );
            }
        }
        // The Control-branch scale carried inside a conditioning entry is guarded too (F-001).
        let ctrl = GenerationRequest {
            conditioning: vec![Conditioning::Control {
                image: img(8, 8),
                kind: ControlKind::Pose,
                scale: Some(f32::NAN),
            }],
            ..base_req()
        };
        let err = c.validate_request("m", &ctrl).unwrap_err();
        assert!(
            err.to_string().contains("conditioning.control.scale"),
            "got {err}"
        );
        // A fully-finite request across the extended knobs still passes.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    guidance_eta: Some(1.0),
                    guidance_momentum: Some(0.0),
                    strength: Some(0.6),
                    control_scale: Some(1.0),
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn oversized_counters_hit_the_sanity_cap() {
        // F-004: the floor now rejects a pathological `steps` / video-counter value that would launch
        // an effectively-unbounded run. Model-realistic values still pass.
        let c = Capabilities {
            max_size: 4096,
            ..caps()
        };
        let steps = GenerationRequest {
            steps: Some(u32::MAX),
            ..base_req()
        };
        assert!(
            c.validate_request("m", &steps)
                .unwrap_err()
                .to_string()
                .contains("steps"),
            "u32::MAX steps must be capped"
        );
        let frames = GenerationRequest {
            frames: Some(u32::MAX),
            ..base_req()
        };
        assert!(
            c.validate_request("m", &frames).is_err(),
            "u32::MAX frames capped"
        );
        let fps = GenerationRequest {
            fps: Some(u32::MAX),
            ..base_req()
        };
        assert!(
            c.validate_request("m", &fps).is_err(),
            "u32::MAX fps capped"
        );
        // Realistic values pass.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    steps: Some(50),
                    frames: Some(121),
                    fps: Some(24),
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn capability_gaps_return_typed_unsupported() {
        // F-008: capability-gap branches must be `Error::Unsupported`, not `Msg`, so candle gating /
        // the worker can distinguish them. Malformed-value branches (range/finiteness) stay `Msg`.
        let c = caps();
        let gap_cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..base_req()
            },
            GenerationRequest {
                true_cfg: Some(4.0),
                ..base_req()
            },
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..base_req()
            },
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance_method: Some("apg".into()),
                ..base_req()
            },
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..base_req()
            },
        ];
        for (i, req) in gap_cases.iter().enumerate() {
            let err = c.validate_request("m", req).unwrap_err();
            assert!(
                matches!(err, Error::Unsupported(_)),
                "gap case {i} should be typed Unsupported, got {err:?}"
            );
        }
    }

    /// A pure-audio capability surface (sc-12834): no visual size bounds (unused for
    /// `Modality::Audio`), an advertised voice/language/sample-rate surface, a 60 s cap, and
    /// `ReferenceAudio` conditioning.
    fn audio_caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::ReferenceAudio],
            audio_sample_rates: vec![24_000, 48_000],
            max_audio_duration_secs: Some(60.0),
            audio_voices: vec!["nova"],
            audio_languages: vec!["en"],
            max_count: 1,
            ..Default::default()
        }
    }

    fn track() -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; 16],
            sample_rate: 24_000,
            channels: 1,
        }
    }

    /// A TTS-shaped request: prompt + typed audio sub-block, size left at the unused 0x0.
    fn audio_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "read this aloud".into(),
            width: 0,
            height: 0,
            audio: Some(AudioParams {
                voice: Some("nova".into()),
                language: Some("en".into()),
                target_duration: Some(12.5),
                sample_rate: Some(24_000),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn audio_request_validates_against_audio_descriptor() {
        // sc-12834 acceptance: a `GenerationRequest { prompt, audio: Some(..) }` validates against
        // an audio Capabilities descriptor through the size-skipping audio floor.
        let c = audio_caps();
        assert!(c.validate_request_audio("tts", &audio_req()).is_ok());
        // A reference-audio conditioned request (voice cloning) passes when advertised.
        let cloned = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: track(),
                strength: Some(0.8),
            }],
            ..audio_req()
        };
        assert!(c.validate_request_audio("tts", &cloned).is_ok());
        // The music-shaped knobs are free-form/positive-gated, not membership-gated.
        let music = GenerationRequest {
            audio: Some(AudioParams {
                target_duration: Some(30.0),
                bpm: Some(120.0),
                musical_key: Some("C minor".into()),
                lyrics: Some("la la la".into()),
                ..Default::default()
            }),
            ..audio_req()
        };
        assert!(c.validate_request_audio("music", &music).is_ok());
    }

    #[test]
    fn audio_floor_rejects_visual_only_mismatches_with_typed_errors() {
        // sc-12834 acceptance: the audio floor still runs every non-size check — capability gaps
        // are typed `Unsupported`, malformed values are `Msg`.
        let c = audio_caps();
        // Capability gaps (visual-only surface the audio descriptor does not advertise).
        let gap_cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..audio_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..audio_req()
            },
            GenerationRequest {
                sampler: Some("euler".into()),
                ..audio_req()
            },
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..audio_req()
            },
        ];
        for (i, req) in gap_cases.iter().enumerate() {
            let err = c.validate_request_audio("tts", req).unwrap_err();
            assert!(
                matches!(err, Error::Unsupported(_)),
                "gap case {i} should be typed Unsupported, got {err:?}"
            );
        }
        // Malformed values stay `Msg`.
        let msg_cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                count: 0,
                ..audio_req()
            },
            GenerationRequest {
                count: 2,
                ..audio_req()
            },
            GenerationRequest {
                steps: Some(0),
                ..audio_req()
            },
        ];
        for (i, req) in msg_cases.iter().enumerate() {
            let err = c.validate_request_audio("tts", req).unwrap_err();
            assert!(
                matches!(err, Error::Msg(_)),
                "msg case {i} should be a Msg range error, got {err:?}"
            );
        }
    }

    #[test]
    fn audio_surface_membership_and_ranges_are_enforced() {
        let c = audio_caps();
        let with_audio = |a: AudioParams| GenerationRequest {
            audio: Some(a),
            ..audio_req()
        };
        // Membership gaps → typed Unsupported naming the field.
        let gaps: [(&str, AudioParams); 3] = [
            (
                "audio.voice",
                AudioParams {
                    voice: Some("santa".into()),
                    ..Default::default()
                },
            ),
            (
                "audio.language",
                AudioParams {
                    language: Some("xx".into()),
                    ..Default::default()
                },
            ),
            (
                "audio.sample_rate",
                AudioParams {
                    sample_rate: Some(44_100),
                    ..Default::default()
                },
            ),
        ];
        for (field, params) in gaps {
            let err = c.validate_request_audio("tts", &with_audio(params)).unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{field}: got {err:?}");
            assert!(err.to_string().contains(field), "{field}: got {err}");
        }
        // Range violations → Msg.
        let over_cap = with_audio(AudioParams {
            target_duration: Some(61.0),
            ..Default::default()
        });
        let err = c.validate_request_audio("tts", &over_cap).unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "got {err:?}");
        assert!(err.to_string().contains("audio.target_duration"), "got {err}");
        for bad in [0.0, -3.0] {
            let req = with_audio(AudioParams {
                target_duration: Some(bad),
                ..Default::default()
            });
            assert!(
                matches!(c.validate_request_audio("tts", &req).unwrap_err(), Error::Msg(_)),
                "target_duration {bad} must be rejected"
            );
        }
        let bad_bpm = with_audio(AudioParams {
            bpm: Some(0.0),
            ..Default::default()
        });
        assert!(matches!(
            c.validate_request_audio("tts", &bad_bpm).unwrap_err(),
            Error::Msg(_)
        ));
        // No advertised duration cap ⇒ only the sanity cap applies.
        let uncapped = Capabilities {
            max_audio_duration_secs: None,
            ..audio_caps()
        };
        assert!(uncapped
            .validate_request_audio(
                "tts",
                &with_audio(AudioParams {
                    target_duration: Some(3600.0),
                    ..Default::default()
                })
            )
            .is_ok());
        assert!(uncapped
            .validate_request_audio(
                "tts",
                &with_audio(AudioParams {
                    target_duration: Some(MAX_DURATION_SECS + 1.0),
                    ..Default::default()
                })
            )
            .is_err());
    }

    #[test]
    fn audio_floats_join_the_finiteness_floor() {
        // The `AudioParams` floats inherit the F-053/F-001 finiteness floor by construction.
        let c = audio_caps();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (field, params) in [
                (
                    "audio.target_duration",
                    AudioParams {
                        target_duration: Some(bad),
                        ..Default::default()
                    },
                ),
                (
                    "audio.bpm",
                    AudioParams {
                        bpm: Some(bad),
                        ..Default::default()
                    },
                ),
            ] {
                let req = GenerationRequest {
                    audio: Some(params),
                    ..audio_req()
                };
                let err = c.validate_request_audio("tts", &req).unwrap_err();
                assert!(matches!(err, Error::Msg(_)), "{field} {bad} → Msg");
                // ±Inf may trip the range/sanity-cap branch first (same convention as the
                // request-level `duration`); NaN falls through every comparison and must be
                // caught by the finiteness floor naming the field.
                assert!(err.to_string().contains(field), "{field} {bad}: got {err}");
                if bad.is_nan() {
                    assert!(
                        err.to_string().contains("must be finite"),
                        "{field} NaN: got {err}"
                    );
                }
            }
        }
        // The ReferenceAudio conditioning strength is guarded too.
        let cloned = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: track(),
                strength: Some(f32::NAN),
            }],
            ..audio_req()
        };
        let err = c.validate_request_audio("tts", &cloned).unwrap_err();
        assert!(
            err.to_string().contains("conditioning.reference_audio.strength"),
            "got {err}"
        );
    }

    #[test]
    fn reference_audio_is_gated_by_the_conditioning_allowlist() {
        // A visual descriptor that does not advertise ReferenceAudio rejects it, typed.
        let visual = caps();
        let req = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: track(),
                strength: None,
            }],
            ..base_req()
        };
        let err = visual.validate_request("m", &req).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
        assert_eq!(
            Conditioning::ReferenceAudio {
                audio: track(),
                strength: None
            }
            .kind(),
            ConditioningKind::ReferenceAudio
        );
    }

    #[test]
    fn audio_output_and_modality_variants_carry_the_track() {
        // The additive output variant round-trips the host-type track (tensor-free invariant).
        let out = GenerationOutput::Audio(track());
        match out {
            GenerationOutput::Audio(t) => {
                assert_eq!(t.sample_rate, 24_000);
                assert_eq!(t.channels, 1);
                assert_eq!(t.samples.len(), 16);
            }
            other => panic!("expected Audio output, got {other:?}"),
        }
        assert_ne!(Modality::Audio, Modality::Both);
        // A visual request is untouched by the audio block: `Default` carries `audio: None`.
        assert!(GenerationRequest::default().audio.is_none());
    }

    #[test]
    fn non_finite_guidance_and_true_cfg_are_rejected() {
        // F-053: a NaN passes `x > 1.0`-style checks; the floor rejects non-finite explicitly. Uses
        // a caps that advertises guidance/true_cfg so the finiteness branch (not the support gate) runs.
        let c = Capabilities {
            supports_guidance: true,
            supports_true_cfg: true,
            ..caps()
        };
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let g = GenerationRequest {
                guidance: Some(bad),
                ..base_req()
            };
            let err = c.validate_request("m", &g).unwrap_err();
            assert!(
                matches!(err, Error::Msg(_)),
                "guidance {bad} → Msg range error"
            );
            assert!(err.to_string().contains("guidance must be finite"));
            let t = GenerationRequest {
                true_cfg: Some(bad),
                ..base_req()
            };
            assert!(matches!(
                c.validate_request("m", &t).unwrap_err(),
                Error::Msg(_)
            ));
        }
        // Finite guidance/true_cfg still pass.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    guidance: Some(3.5),
                    true_cfg: Some(2.0),
                    ..base_req()
                }
            )
            .is_ok());
    }
}
