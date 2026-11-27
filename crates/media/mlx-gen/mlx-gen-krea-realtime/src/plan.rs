//! Weights-free pre-load Krea Realtime execution, KV residency, and chunk delivery plan
//! (`PIPE-16`, `§6.9`, `DEC-13`).
//!
//! Compiled and validated during [`KreaRealtime::validate`](crate::KreaRealtime) and
//! [`KreaRealtime::run`](crate::KreaRealtime) BEFORE opening or allocating model weights.

use mlx_gen::gen_core::{CausalCodecStatePolicy, ChunkResidencyPolicy, GenerationRequest};
use mlx_gen::{Conditioning, Error, Result};

use crate::config::{KreaRealtimeConfig, KvCacheQuant, MODEL_ID};
use crate::generate::chunk_frame_counts;
use crate::scheduler::FewStepSchedule;
use crate::t2v::{
    bounded_latent_frame_count, causal_codec_policy_for_config, resolve_request_config,
    SPATIAL_STRIDE,
};

/// Default output cadence when a request omits `fps` (the reference realtime-video default).
pub(crate) const DEFAULT_FPS: u32 = 16;

/// Default requested **output** frame count when a request omits `frames`/`duration`. The AR path turns
/// this into `(81 − 1)/4 + 1 = 21` latent frames; the non-causal z16 Wan VAE decode then yields
/// `4 · 21 = 84` output frames (`output = 4 · latent`, not `(latent − 1)·4 + 1`), which the pipeline
/// trims back to this requested 81 (see [`crate::t2v::decode_latents_to_video`]). 81 is the reference
/// realtime-video canonical clip length.
pub(crate) const DEFAULT_FRAMES: u32 = 81;

/// Resolved output cadence and z16 latent frame counts shared by pre-load validation and execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedFrames {
    pub(crate) fps: u32,
    pub(crate) output: u32,
    pub(crate) output_latent: usize,
    pub(crate) generation_latent: usize,
}

/// Request route selected before weight loading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KreaRealtimeRoute {
    T2v,
    I2v {
        reference_latent_frames: usize,
    },
    V2v {
        source_frames: usize,
        source_latent_frames: usize,
        strength: f32,
    },
}

/// Pre-load [`CausalKvCache`](crate::CausalKvCache) residency and sliding-window plan.
#[derive(Clone, Debug, PartialEq)]
pub struct KreaKvResidencyPlan {
    pub local_attn_frames: i64,
    pub max_attention_size: usize,
    pub sink_frames: usize,
    pub sink_tokens: usize,
    pub kv_cache_quant: Option<KvCacheQuant>,
    pub kv_bytes_per_token: usize,
    pub total_sequence_tokens: usize,
    pub steady_retained_kv_tokens: usize,
    pub peak_retained_kv_tokens: usize,
    pub steady_retained_kv_bytes: u64,
    pub peak_retained_kv_bytes: u64,
    pub do_kv_recomp: bool,
    pub context_noise: f32,
    pub commits_per_generated_chunk: usize,
    pub context_commits: usize,
}

/// Complete pre-load execution plan for Krea Realtime 14B (`PIPE-16`, `§6.9`).
#[derive(Clone, Debug, PartialEq)]
pub struct KreaRealtimeRenderPlan {
    pub route: KreaRealtimeRoute,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub output_frames: u32,
    pub output_latent_frames: usize,
    pub context_latent_frames: usize,
    pub generation_latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub frame_seq_length: usize,
    pub seq_length: usize,
    pub frames_per_block: usize,
    pub chunk_frame_counts: Vec<usize>,
    pub context_chunk_frame_counts: Vec<usize>,
    pub num_chunks: usize,
    pub n_steps: usize,
    pub total_outer_steps: usize,
    pub step_timesteps_f64: Vec<f64>,
    pub step_sigmas_f64: Vec<f64>,
    pub execution_sigmas_f64: Vec<f64>,
    pub kv_residency: KreaKvResidencyPlan,
    pub chunk_residency: ChunkResidencyPolicy,
    pub causal_codec_state: CausalCodecStatePolicy,
    pub resolved_config: KreaRealtimeConfig,
}

/// Resolve the output cadence/count and its z16 latent count exactly once for provider validation and
/// execution. Keeping explicit-frames, duration-derived, and default precedence here prevents the
/// preflight and run paths from disagreeing about the model-local allocation cap.
pub(crate) fn resolve_frames(req: &GenerationRequest) -> Result<ResolvedFrames> {
    let fps = req.fps.unwrap_or(DEFAULT_FPS);
    if fps == 0 {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: fps must be >= 1, got 0"
        )));
    }
    let output = match req.frames {
        Some(frames) => frames,
        None => match req.duration {
            Some(duration) => {
                if !duration.is_finite() || duration <= 0.0 {
                    return Err(Error::Unsupported(format!(
                        "{MODEL_ID}: duration must be finite and > 0.0, got {duration}"
                    )));
                }
                ((duration * fps as f32).round() as u32).max(1)
            }
            None => DEFAULT_FRAMES,
        },
    };
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
/// Typed [`Error::Unsupported`], not [`Error::Msg`]: the worker classifies `Unsupported` as a
/// user-facing invalid-payload refusal and `Msg` as an opaque internal engine failure.
pub(crate) fn reject_unimplemented_video_clip_knobs(req: &GenerationRequest) -> Result<()> {
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
pub(crate) fn validate_v2v_contract(
    req: &GenerationRequest,
    resolved: ResolvedFrames,
) -> Result<()> {
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
    if !explicitly_v2v {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: a VideoClip carrier requires video_mode=video_to_video"
        )));
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

impl KreaRealtimeRenderPlan {
    /// Validate and compile a pre-load [`KreaRealtimeRenderPlan`] from `base_cfg` and `req`
    /// without opening or loading any model weights.
    pub fn from_generation_request(
        base_cfg: &KreaRealtimeConfig,
        req: &GenerationRequest,
    ) -> Result<Self> {
        reject_unimplemented_video_clip_knobs(req)?;
        if req.width == 0 || req.height == 0 {
            return Err(Error::Msg("krea realtime: width/height must be > 0".into()));
        }
        let grid_h = base_cfg.wan.patch_size.1 * SPATIAL_STRIDE;
        let grid_w = base_cfg.wan.patch_size.2 * SPATIAL_STRIDE;
        if grid_h == 0
            || grid_w == 0
            || !(req.height as usize).is_multiple_of(grid_h)
            || !(req.width as usize).is_multiple_of(grid_w)
        {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: {}×{} must be positive multiples of {}×{}",
                req.width, req.height, grid_w, grid_h
            )));
        }
        if req.steps == Some(0) {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: steps must be >= 1"
            )));
        }

        let frames = resolve_frames(req)?;
        validate_v2v_contract(req, frames)?;

        let (route, context_latent_frames, generation_latent_frames) =
            if let Some((clip, strength)) = req.conditioning.iter().find_map(|c| match c {
                Conditioning::VideoClip {
                    frames, strength, ..
                } => Some((frames.as_slice(), *strength)),
                _ => None,
            }) {
                (
                    KreaRealtimeRoute::V2v {
                        source_frames: clip.len(),
                        source_latent_frames: frames.generation_latent,
                        strength,
                    },
                    0usize,
                    frames.generation_latent,
                )
            } else if req
                .conditioning
                .iter()
                .any(|c| matches!(c, Conditioning::Reference { .. }))
            {
                const F_REF: usize = 1;
                let num_generate = frames
                    .output_latent
                    .checked_sub(F_REF)
                    .filter(|&n| n >= 1)
                    .ok_or_else(|| {
                        Error::Unsupported(format!(
                        "{MODEL_ID} image_to_video requires at least 2 latent frames (5+ output \
                         frames: a reference frame + a generated frame), got {} output frame(s)",
                        frames.output
                    ))
                    })?;
                (
                    KreaRealtimeRoute::I2v {
                        reference_latent_frames: F_REF,
                    },
                    F_REF,
                    num_generate,
                )
            } else {
                (KreaRealtimeRoute::T2v, 0usize, frames.output_latent)
            };

        let latent_height = req.height as usize / SPATIAL_STRIDE;
        let latent_width = req.width as usize / SPATIAL_STRIDE;
        let total_kv_latent_frames = context_latent_frames + generation_latent_frames;
        let resolved_config = resolve_request_config(
            base_cfg,
            latent_height,
            latent_width,
            total_kv_latent_frames,
        )?;

        let frames_per_block = resolved_config.ar.num_frames_per_block;
        if frames_per_block == 0 {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: num_frames_per_block must be >= 1"
            )));
        }

        let n_steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(resolved_config.ar.denoising_step_list.len());
        if n_steps == 0 {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: denoising_step_list must not be empty"
            )));
        }

        let schedule = match route {
            KreaRealtimeRoute::V2v { strength, .. } => FewStepSchedule::for_strength(
                resolved_config.ar.timestep_shift as f64,
                strength as f64,
                n_steps,
            )?,
            KreaRealtimeRoute::T2v | KreaRealtimeRoute::I2v { .. } => {
                FewStepSchedule::for_strength(
                    resolved_config.ar.timestep_shift as f64,
                    1.0,
                    n_steps,
                )?
            }
        };
        let step_timesteps_f64 = schedule.step_timesteps().to_vec();
        let step_sigmas_f64: Vec<f64> = step_timesteps_f64
            .iter()
            .map(|&t| schedule.sigma_at_timestep(t))
            .collect();
        let execution_sigmas_f64: Vec<f64> = step_sigmas_f64
            .iter()
            .copied()
            .chain(std::iter::once(0.0))
            .collect();

        let context_chunk_frame_counts =
            chunk_frame_counts(context_latent_frames, frames_per_block);
        let chunk_frame_counts = chunk_frame_counts(generation_latent_frames, frames_per_block);
        let num_chunks = chunk_frame_counts.len();
        let total_outer_steps = num_chunks * n_steps;

        let frame_seq_length = resolved_config.ar.frame_seq_length;
        let seq_length = resolved_config.ar.seq_length;
        let max_attention_size = resolved_config.ar.max_attention_size();
        let sink_tokens = resolved_config.ar.sink_tokens();
        let kv_bytes_per_token = resolved_config.kv_bytes_per_token()?;
        let total_sequence_tokens = total_kv_latent_frames * frame_seq_length;
        let window_cap = max_attention_size.max(sink_tokens);
        let steady_retained_kv_tokens = total_sequence_tokens.min(window_cap);
        let peak_retained_kv_tokens = total_sequence_tokens
            .min(steady_retained_kv_tokens + frames_per_block * frame_seq_length);
        let kv_residency = KreaKvResidencyPlan {
            local_attn_frames: resolved_config.ar.local_attn_size,
            max_attention_size,
            sink_frames: resolved_config.ar.sink_size,
            sink_tokens,
            kv_cache_quant: resolved_config.ar.kv_cache_quant,
            kv_bytes_per_token,
            total_sequence_tokens,
            steady_retained_kv_tokens,
            peak_retained_kv_tokens,
            steady_retained_kv_bytes: (steady_retained_kv_tokens as u64)
                .saturating_mul(kv_bytes_per_token as u64),
            peak_retained_kv_bytes: (peak_retained_kv_tokens as u64)
                .saturating_mul(kv_bytes_per_token as u64),
            do_kv_recomp: resolved_config.ar.do_kv_recomp,
            context_noise: resolved_config.ar.context_noise,
            commits_per_generated_chunk: 1,
            context_commits: context_chunk_frame_counts.len(),
        };

        let causal_codec_state = causal_codec_policy_for_config(&resolved_config);
        Ok(Self {
            route,
            width: req.width,
            height: req.height,
            fps: frames.fps,
            output_frames: frames.output,
            output_latent_frames: frames.output_latent,
            context_latent_frames,
            generation_latent_frames,
            latent_height,
            latent_width,
            frame_seq_length,
            seq_length,
            frames_per_block,
            chunk_frame_counts,
            context_chunk_frame_counts,
            num_chunks,
            n_steps,
            total_outer_steps,
            step_timesteps_f64,
            step_sigmas_f64,
            execution_sigmas_f64,
            kv_residency,
            chunk_residency: ChunkResidencyPolicy::TerminalWholeClip,
            causal_codec_state,
            resolved_config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_tracks_kv_residency_with_sink_and_q8_quantization() {
        let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
        cfg.ar.sink_size = 2;
        let req = GenerationRequest {
            width: 832,
            height: 480,
            frames: Some(81),
            count: 1,
            ..Default::default()
        };

        let bf16_plan = KreaRealtimeRenderPlan::from_generation_request(&cfg, &req).unwrap();
        let mut q8_cfg = cfg;
        q8_cfg.ar.kv_cache_quant = Some(KvCacheQuant::Q8);
        let q8_plan = KreaRealtimeRenderPlan::from_generation_request(&q8_cfg, &req).unwrap();

        assert_eq!(bf16_plan.kv_residency.sink_frames, 2);
        assert_eq!(bf16_plan.kv_residency.sink_tokens, 2 * 1560);
        assert_eq!(
            bf16_plan.causal_codec_state,
            CausalCodecStatePolicy::RollingKvWindow {
                max_attention_tokens: 6 * 1560,
                sink_tokens: 2 * 1560,
            }
        );
        assert!(
            q8_plan.kv_residency.steady_retained_kv_bytes
                < bf16_plan.kv_residency.steady_retained_kv_bytes
        );
        assert!(
            q8_plan.kv_residency.steady_retained_kv_bytes
                <= q8_plan.kv_residency.peak_retained_kv_bytes
        );
    }
}
