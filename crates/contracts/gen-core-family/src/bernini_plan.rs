//! Weights-free pre-load execution plan for Bernini (`bernini` full MAR planner+renderer and
//! `bernini_renderer` renderer-only, `DEC-3`, `DEC-13`).
//!
//! Captures the shared host contract across `mlx-gen-bernini` and `candle-gen-bernini`:
//! - explicit stage sequence (`Stage 1 MAR Planner` -> `Drop Planner` -> `Stage 2 Dual
//!   Wan2.2-A14B Experts high->low` -> `VaeDecode`) for full `bernini`, vs Stage 2+3 for
//!   `bernini_renderer`,
//! - model-dependent branches (`t2v`, `i2v`, `v2v`, `t2i`, `i2i`, `r2v`, `rv2v`, `ads2v`),
//! - 4-way ViT/text handoff streams (`wtxt_wvit`, `wtxt_wovit`, `wotxt_wvit`, `wotxt_wovit`),
//! - dual-expert switch at `switch_dit_boundary * num_train_timesteps` with `OMEGA_SCALE = 0.8`
//!   one-time rescale,
//! - exact backend seed schemes (MLX `0x4d_a4` / `0x9e_37 + s` vs Candle MT19937 `numpy_shuffle` /
//!   `torch_step_noise`), and
//! - output contract (`Images` when `frames == 1`, `Video` on `1 + 4*k` grid when `frames > 1`).

use gen_core::{Conditioning, Error, GenerationRequest, ModelDescriptor, Result};
use serde::{Deserialize, Serialize};

pub const BERNINI_SIZE_MULTIPLE: u32 = 16;
pub const BERNINI_MAX_AREA_14B: usize = 921_600;
pub const BERNINI_DEFAULT_FRAMES: usize = 81;
pub const BERNINI_DEFAULT_STEPS: usize = 40;
pub const BERNINI_PLANNING_STEPS: usize = 25;
pub const BERNINI_VIT_DENOISING_STEPS: usize = 5;
pub const BERNINI_VIT_TXT_CFG: f32 = 1.2;
pub const BERNINI_VIT_IMG_CFG: f32 = 1.0;
pub const BERNINI_OMEGA_SCALE: f32 = 0.8;
pub const BERNINI_RENDERER_DEFAULT_SHIFT: f32 = 3.0;
pub const BERNINI_FULL_DEFAULT_SHIFT: f64 = 5.0;
pub const BERNINI_NUM_TRAIN_TIMESTEPS: usize = 1000;
pub const BERNINI_DEFAULT_BOUNDARY: f32 = 0.875;
pub const BERNINI_DEFAULT_FPS: u32 = 16;

pub const BERNINI_FOUR_STREAMS: [&str; 4] =
    ["wtxt_wvit", "wtxt_wovit", "wotxt_wvit", "wotxt_wovit"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BerniniEntrypoint {
    Full,
    Renderer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BerniniBackendKind {
    Mlx,
    Candle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BerniniCanonicalBranch {
    T2i,
    T2v,
    I2i,
    I2v,
    V2v,
    R2v,
    Rv2v,
    Ads2v,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BerniniRendererGuidanceMode {
    T2v,
    T2vApg,
    V2v,
    V2vChain,
    V2vApg,
    R2vApg,
    Rv2v,
}

impl BerniniRendererGuidanceMode {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "t2v" => Self::T2v,
            "t2v_apg" => Self::T2vApg,
            "v2v" => Self::V2v,
            "v2v_chain" => Self::V2vChain,
            "v2v_apg" => Self::V2vApg,
            "r2v_apg" => Self::R2vApg,
            "rv2v" => Self::Rv2v,
            _ => return None,
        })
    }

    pub const fn needs_conditioning(self) -> bool {
        !matches!(self, Self::T2v | Self::T2vApg)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BerniniFullVitMode {
    VaeTxtVit,
    VaeTxtVitWapg,
    Rv2vWapg,
    R2vWapg,
    V2vApg,
}

impl BerniniFullVitMode {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "vae_txt_vit" => Self::VaeTxtVit,
            "vae_txt_vit_wapg" => Self::VaeTxtVitWapg,
            "rv2v_wapg" => Self::Rv2vWapg,
            "r2v_wapg" => Self::R2vWapg,
            "v2v_apg" => Self::V2vApg,
            _ => return None,
        })
    }

    pub const fn needs_conditioning(self) -> bool {
        matches!(self, Self::Rv2vWapg | Self::R2vWapg | Self::V2vApg)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BerniniOutputContract {
    Images { expected_count: usize },
    Video { fps: u32, latent_frames: usize },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum BerniniStageSpec {
    MarPlannerLoop {
        planning_steps: usize,
        vit_denoising_steps: usize,
        vit_txt_cfg: f32,
        vit_img_cfg: f32,
        input_streams: [String; 3],
        handoff_streams: [String; 4],
    },
    DropPlanner {
        flush_device_cache: bool,
    },
    DualWan22A14bExpertsHighToLow {
        boundary_timestep: f64,
        switch_dit_boundary: f32,
        num_train_timesteps: usize,
        omega_scale_once_on_low: f32,
        steps: usize,
        shift: f64,
    },
    VaeDecode {
        output: BerniniOutputContract,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BerniniSeedScheme {
    pub permutation_scheme: String,
    pub planner_step_noise_scheme: String,
    pub renderer_init_noise_scheme: String,
    pub default_seed_when_omitted: bool,
}

impl BerniniSeedScheme {
    pub fn for_backend(backend: BerniniBackendKind, entrypoint: BerniniEntrypoint) -> Self {
        match backend {
            BerniniBackendKind::Mlx => Self {
                permutation_scheme: if entrypoint == BerniniEntrypoint::Full {
                    "mlx-random-key(seed+0x4da4)-argsort".to_owned()
                } else {
                    "none".to_owned()
                },
                planner_step_noise_scheme: if entrypoint == BerniniEntrypoint::Full {
                    "mlx-random-key(seed+0x9e37+step)-normal".to_owned()
                } else {
                    "none".to_owned()
                },
                renderer_init_noise_scheme: "mlx-random-key(seed)-normal".to_owned(),
                default_seed_when_omitted: true,
            },
            BerniniBackendKind::Candle => Self {
                permutation_scheme: if entrypoint == BerniniEntrypoint::Full {
                    "bernini_numpy_mt19937_shuffle_v1".to_owned()
                } else {
                    "none".to_owned()
                },
                planner_step_noise_scheme: if entrypoint == BerniniEntrypoint::Full {
                    "bernini_torch_mt19937_normal_v1".to_owned()
                } else {
                    "none".to_owned()
                },
                renderer_init_noise_scheme: "wan_candle_create_noise_mt19937".to_owned(),
                default_seed_when_omitted: true,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BerniniPlanKnobs {
    pub switch_dit_boundary: f32,
    pub shift: f32,
    pub max_trained_src_id: f64,
    pub interpolate_src_id: bool,
    pub max_sequence_length: usize,
}

impl Default for BerniniPlanKnobs {
    fn default() -> Self {
        Self {
            switch_dit_boundary: BERNINI_DEFAULT_BOUNDARY,
            shift: BERNINI_RENDERER_DEFAULT_SHIFT,
            max_trained_src_id: 5.0,
            interpolate_src_id: true,
            max_sequence_length: 512,
        }
    }
}

/// Pre-load serializable Bernini execution plan built by `validate()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BerniniRenderPlan {
    pub provider_id: String,
    pub entrypoint: BerniniEntrypoint,
    pub backend: BerniniBackendKind,
    pub branch: BerniniCanonicalBranch,
    pub canonical_task: String,
    pub renderer_mode: Option<BerniniRendererGuidanceMode>,
    pub vit_mode: Option<BerniniFullVitMode>,
    pub stages: Vec<BerniniStageSpec>,
    pub planner_streams: [String; 4],
    pub width: u32,
    pub height: u32,
    pub frames: usize,
    pub fps: u32,
    pub steps: usize,
    pub folded_progress_total: usize,
    pub trained_sources: f64,
    pub sampler: String,
    pub scheduler_shift: Option<f32>,
    pub seed: Option<u64>,
    pub omega_txt: f32,
    pub omega_img: f32,
    pub omega_vid: f32,
    pub omega_tgt: Option<f32>,
    pub eta: f32,
    pub switch_dit_boundary: f32,
    pub num_train_timesteps: usize,
    pub boundary_timestep: f64,
    pub omega_scale: f32,
    pub shift: f64,
    pub seed_scheme: BerniniSeedScheme,
    pub output_contract: BerniniOutputContract,
}

pub fn resolve_renderer_mode(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
) -> BerniniRendererGuidanceMode {
    if let Some(s) = video_mode {
        if let Some(m) = BerniniRendererGuidanceMode::from_name(s) {
            return m;
        }
        match s {
            "t2i" | "t2v" => return BerniniRendererGuidanceMode::T2vApg,
            "i2i" => return BerniniRendererGuidanceMode::V2v,
            "v2v" | "mv2v" | "ads2v" => return BerniniRendererGuidanceMode::V2vApg,
            "i2v" | "r2v" => return BerniniRendererGuidanceMode::R2vApg,
            "rv2v" => return BerniniRendererGuidanceMode::Rv2v,
            _ => {}
        }
    }
    match (has_video, has_image) {
        (true, true) => BerniniRendererGuidanceMode::Rv2v,
        (true, false) => BerniniRendererGuidanceMode::V2vApg,
        (false, true) => BerniniRendererGuidanceMode::V2v,
        (false, false) => BerniniRendererGuidanceMode::T2vApg,
    }
}

pub fn resolve_full_vit_mode(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> BerniniFullVitMode {
    if let Some(s) = video_mode {
        if let Some(m) = BerniniFullVitMode::from_name(s) {
            return m;
        }
        match s {
            "t2i" | "t2v" | "i2i" => return BerniniFullVitMode::VaeTxtVitWapg,
            "v2v" | "mv2v" | "ads2v" => return BerniniFullVitMode::V2vApg,
            "i2v" | "r2v" | "rv2v" => return BerniniFullVitMode::Rv2vWapg,
            _ => {}
        }
    }
    match (has_video, has_image, out_video) {
        (true, _, _) => {
            if has_image {
                BerniniFullVitMode::Rv2vWapg
            } else {
                BerniniFullVitMode::V2vApg
            }
        }
        (false, true, true) => BerniniFullVitMode::Rv2vWapg,
        _ => BerniniFullVitMode::VaeTxtVitWapg,
    }
}

pub fn canonical_task_for_full(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> &'static str {
    if let Some(s) = video_mode {
        match s {
            "t2i" => return "t2i",
            "t2v" => return "t2v",
            "i2i" => return "i2i",
            "v2v" => return "v2v",
            "r2v" => return "r2v",
            "rv2v" => return "rv2v",
            _ => {}
        }
    }
    match (has_video, has_image, out_video) {
        (true, true, _) => "rv2v",
        (true, false, _) => "v2v",
        (false, true, true) => "r2v",
        (false, true, false) => "i2i",
        (false, false, true) => "t2v",
        (false, false, false) => "t2i",
    }
}

pub fn resolve_canonical_branch(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> BerniniCanonicalBranch {
    if let Some(s) = video_mode {
        match s {
            "t2i" => return BerniniCanonicalBranch::T2i,
            "t2v" | "t2v_apg" => return BerniniCanonicalBranch::T2v,
            "i2i" => return BerniniCanonicalBranch::I2i,
            "i2v" => return BerniniCanonicalBranch::I2v,
            "v2v" | "v2v_chain" | "v2v_apg" | "mv2v" => return BerniniCanonicalBranch::V2v,
            "r2v" | "r2v_apg" => return BerniniCanonicalBranch::R2v,
            "rv2v" | "rv2v_wapg" => return BerniniCanonicalBranch::Rv2v,
            "ads2v" => return BerniniCanonicalBranch::Ads2v,
            _ => {}
        }
    }
    match (has_video, has_image, out_video) {
        (true, true, _) => BerniniCanonicalBranch::Rv2v,
        (true, false, _) => BerniniCanonicalBranch::V2v,
        (false, true, true) => BerniniCanonicalBranch::R2v,
        (false, true, false) => BerniniCanonicalBranch::I2i,
        (false, false, true) => BerniniCanonicalBranch::T2v,
        (false, false, false) => BerniniCanonicalBranch::T2i,
    }
}

/// Shared request-geometry validation for both Bernini entry points.
pub fn validate_bernini_request_geometry(id: &str, req: &GenerationRequest) -> Result<()> {
    if req.steps == Some(0) {
        return Err(Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
        )));
    }
    if let Some(shift) = req.scheduler_shift {
        if !shift.is_finite() || shift <= 0.0 {
            return Err(Error::Unsupported(
                "scheduler shift must be finite and > 0".into(),
            ));
        }
    }
    if !req.width.is_multiple_of(BERNINI_SIZE_MULTIPLE)
        || !req.height.is_multiple_of(BERNINI_SIZE_MULTIPLE)
    {
        return Err(Error::Msg(format!(
            "{id}: width/height must be multiples of {BERNINI_SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    let is_still_image = req.frames == Some(1);
    let area = req.width as usize * req.height as usize;
    if !is_still_image && area > BERNINI_MAX_AREA_14B {
        return Err(Error::Msg(format!(
            "{id}: width×height ({}×{} = {area} px) exceeds the max area {BERNINI_MAX_AREA_14B} px \
             (1280×720); reduce the resolution",
            req.width, req.height
        )));
    }
    if let Some(f) = req.frames {
        if f == 0 || f % 4 != 1 {
            return Err(Error::Msg(format!(
                "{id}: num_frames must be 1 + 4·k (got {f})"
            )));
        }
    }
    Ok(())
}

/// Reject empty or non-`1+4k` conditioning video clips and unimplemented per-clip knobs (`strength`,
/// `frame_idx`).
pub fn validate_bernini_conditioning(id: &str, req: &GenerationRequest) -> Result<(bool, bool)> {
    let mut has_video = false;
    let mut has_image = false;
    for conditioning in &req.conditioning {
        match conditioning {
            Conditioning::VideoClip {
                frames,
                strength,
                frame_idx,
            } => {
                if frames.is_empty() {
                    return Err(Error::Msg("bernini: empty conditioning video clip".into()));
                }
                if frames.len() % 4 != 1 {
                    return Err(Error::Msg(format!(
                        "bernini: conditioning video-clip frame count must be 1 + 4·k (got {})",
                        frames.len()
                    )));
                }
                if *strength != 1.0 {
                    return Err(Error::Unsupported(format!(
                        "{id} does not implement VideoClip strength (got {strength}); remove it or \
                         leave it at the default 1.0 — Bernini conditions through its guidance mode \
                         and has no per-clip denoise mask to weight"
                    )));
                }
                if *frame_idx != 0 {
                    return Err(Error::Unsupported(format!(
                        "{id} does not implement VideoClip frame_idx (got {frame_idx}); remove it \
                         or leave it at the default 0 — Bernini conditions on the clip as a whole \
                         and renders its own timeline, so there is no position to splice it at"
                    )));
                }
                has_video = true;
            }
            Conditioning::Reference { .. } | Conditioning::MultiReference { .. } => {
                has_image = true;
            }
            other => {
                return Err(Error::Msg(format!(
                    "{id}: unsupported conditioning kind {:?}",
                    other.kind()
                )));
            }
        }
    }
    Ok((has_video, has_image))
}

/// Validate `req` against `descriptor` and build a serializable [`BerniniRenderPlan`].
pub fn build_bernini_render_plan(
    descriptor: &ModelDescriptor,
    entrypoint: BerniniEntrypoint,
    backend: BerniniBackendKind,
    knobs: BerniniPlanKnobs,
    req: &GenerationRequest,
) -> Result<BerniniRenderPlan> {
    let id = descriptor.id;
    descriptor.capabilities.validate_request(id, req)?;
    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    validate_bernini_request_geometry(id, req)?;
    let (has_video, has_image) = validate_bernini_conditioning(id, req)?;

    let frames = req.frames.unwrap_or(BERNINI_DEFAULT_FRAMES as u32).max(1) as usize;
    let out_video = frames > 1;
    let steps = req
        .steps
        .map(|s| s as usize)
        .unwrap_or(BERNINI_DEFAULT_STEPS)
        .max(1);
    let fps = req.fps.unwrap_or(BERNINI_DEFAULT_FPS);

    let (renderer_mode, vit_mode, canonical_task, shift, folded_progress_total, stages) =
        match entrypoint {
            BerniniEntrypoint::Renderer => {
                let mode = resolve_renderer_mode(req.video_mode.as_deref(), has_video, has_image);
                let has_source = has_video || has_image;
                if mode.needs_conditioning() && !has_source {
                    return Err(Error::Msg(format!(
                        "{id}: guidance mode {mode:?} needs source conditioning (a reference image \
                         / video clip) but none was provided — pass conditioning or select a \
                         text-only mode (t2v/t2v_apg)"
                    )));
                }
                if !mode.needs_conditioning() && has_source {
                    return Err(Error::Msg(format!(
                        "{id}: guidance mode {mode:?} is text-only but source conditioning was \
                         provided (it would be silently ignored) — select a conditioning mode or \
                         drop the conditioning"
                    )));
                }
                let shift = req
                    .scheduler_shift
                    .map(|s| s as f64)
                    .unwrap_or(knobs.shift as f64);
                let boundary_timestep =
                    knobs.switch_dit_boundary as f64 * BERNINI_NUM_TRAIN_TIMESTEPS as f64;
                let output = output_contract_for(frames, fps);
                let stages = vec![
                    BerniniStageSpec::DualWan22A14bExpertsHighToLow {
                        boundary_timestep,
                        switch_dit_boundary: knobs.switch_dit_boundary,
                        num_train_timesteps: BERNINI_NUM_TRAIN_TIMESTEPS,
                        omega_scale_once_on_low: BERNINI_OMEGA_SCALE,
                        steps,
                        shift,
                    },
                    BerniniStageSpec::VaeDecode {
                        output: output.clone(),
                    },
                ];
                (
                    Some(mode),
                    None,
                    "renderer".to_owned(),
                    shift,
                    steps,
                    stages,
                )
            }
            BerniniEntrypoint::Full => {
                let vmode = resolve_full_vit_mode(
                    req.video_mode.as_deref(),
                    has_video,
                    has_image,
                    out_video,
                );
                let has_source = has_video || has_image;
                if vmode.needs_conditioning() && !has_source {
                    return Err(Error::Msg(format!(
                        "{id}: guidance mode {vmode:?} needs source conditioning (a reference \
                         image / video clip) but none was provided — pass conditioning or select a \
                         text/edit mode"
                    )));
                }
                let task = canonical_task_for_full(
                    req.video_mode.as_deref(),
                    has_video,
                    has_image,
                    out_video,
                );
                let shift = req
                    .scheduler_shift
                    .map(|s| s as f64)
                    .unwrap_or(BERNINI_FULL_DEFAULT_SHIFT);
                let boundary_timestep =
                    knobs.switch_dit_boundary as f64 * BERNINI_NUM_TRAIN_TIMESTEPS as f64;
                let output = output_contract_for(frames, fps);
                let stages = vec![
                    BerniniStageSpec::MarPlannerLoop {
                        planning_steps: BERNINI_PLANNING_STEPS,
                        vit_denoising_steps: BERNINI_VIT_DENOISING_STEPS,
                        vit_txt_cfg: BERNINI_VIT_TXT_CFG,
                        vit_img_cfg: BERNINI_VIT_IMG_CFG,
                        input_streams: [
                            "cond".to_owned(),
                            "uncond".to_owned(),
                            "imgcond".to_owned(),
                        ],
                        handoff_streams: BERNINI_FOUR_STREAMS.map(str::to_owned),
                    },
                    BerniniStageSpec::DropPlanner {
                        flush_device_cache: matches!(backend, BerniniBackendKind::Mlx),
                    },
                    BerniniStageSpec::DualWan22A14bExpertsHighToLow {
                        boundary_timestep,
                        switch_dit_boundary: knobs.switch_dit_boundary,
                        num_train_timesteps: BERNINI_NUM_TRAIN_TIMESTEPS,
                        omega_scale_once_on_low: BERNINI_OMEGA_SCALE,
                        steps,
                        shift,
                    },
                    BerniniStageSpec::VaeDecode {
                        output: output.clone(),
                    },
                ];
                (
                    None,
                    Some(vmode),
                    task.to_owned(),
                    shift,
                    BERNINI_PLANNING_STEPS + steps,
                    stages,
                )
            }
        };

    let branch =
        resolve_canonical_branch(req.video_mode.as_deref(), has_video, has_image, out_video);
    let trained_sources = if branch == BerniniCanonicalBranch::Ads2v {
        3.0
    } else {
        knobs.max_trained_src_id
    };
    let boundary_timestep = knobs.switch_dit_boundary as f64 * BERNINI_NUM_TRAIN_TIMESTEPS as f64;
    let sampler = req
        .sampler
        .as_deref()
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "uni_pc".to_owned());
    let omega_txt = req.guidance.unwrap_or(4.0);
    let omega_img = 4.5_f32;
    let omega_vid = 1.25_f32;
    let (omega_tgt, eta) = match entrypoint {
        BerniniEntrypoint::Full => (Some(0.5_f32), 1.0_f32),
        BerniniEntrypoint::Renderer => (None, 0.5_f32),
    };

    Ok(BerniniRenderPlan {
        provider_id: id.to_owned(),
        entrypoint,
        backend,
        branch,
        canonical_task,
        renderer_mode,
        vit_mode,
        stages,
        planner_streams: BERNINI_FOUR_STREAMS.map(str::to_owned),
        width: req.width,
        height: req.height,
        frames,
        fps,
        steps,
        folded_progress_total,
        trained_sources,
        sampler,
        scheduler_shift: req.scheduler_shift,
        seed: req.seed,
        omega_txt,
        omega_img,
        omega_vid,
        omega_tgt,
        eta,
        switch_dit_boundary: knobs.switch_dit_boundary,
        num_train_timesteps: BERNINI_NUM_TRAIN_TIMESTEPS,
        boundary_timestep,
        omega_scale: BERNINI_OMEGA_SCALE,
        shift,
        seed_scheme: BerniniSeedScheme::for_backend(backend, entrypoint),
        output_contract: output_contract_for(frames, fps),
    })
}

fn output_contract_for(frames: usize, fps: u32) -> BerniniOutputContract {
    if frames == 1 {
        BerniniOutputContract::Images { expected_count: 1 }
    } else {
        BerniniOutputContract::Video {
            fps,
            latent_frames: (frames - 1) / 4 + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{Capabilities, ConditioningKind, Image, Modality};

    fn bernini_descriptor(id: &'static str) -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: Some(&gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
            control_kinds: None,
            required_components: &[],
            id,
            family: "bernini",
            backend: "test",
            modality: if id == "bernini" {
                Modality::Both
            } else {
                Modality::Video
            },
            capabilities: Capabilities {
                supports_negative_prompt: true,
                supports_guidance: true,
                conditioning: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                    ConditioningKind::VideoClip,
                ],
                samplers: vec!["uni_pc", "unipc"],
                min_size: 16,
                max_size: 1280,
                max_count: 1,
                ..Default::default()
            },
        }
    }

    #[test]
    fn bernini_plan_round_trips_json_and_pins_stages_branches_and_outputs() {
        let full_desc = bernini_descriptor("bernini");
        let still_req = GenerationRequest {
            prompt: "a portrait".into(),
            width: 1024,
            height: 1024,
            frames: Some(1),
            steps: Some(12),
            ..Default::default()
        };
        let still_plan = build_bernini_render_plan(
            &full_desc,
            BerniniEntrypoint::Full,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &still_req,
        )
        .unwrap();
        assert_eq!(still_plan.branch, BerniniCanonicalBranch::T2i);
        assert_eq!(
            still_plan.output_contract,
            BerniniOutputContract::Images { expected_count: 1 }
        );
        assert_eq!(still_plan.stages.len(), 4);
        assert_eq!(still_plan.folded_progress_total, 25 + 12);
        assert_eq!(still_plan.boundary_timestep, 875.0);
        assert_eq!(still_plan.omega_scale, 0.8);
        assert_eq!(still_plan.planner_streams, BERNINI_FOUR_STREAMS);
        assert_eq!(
            still_plan.seed_scheme.permutation_scheme,
            "bernini_numpy_mt19937_shuffle_v1"
        );
        let decoded: BerniniRenderPlan =
            serde_json::from_str(&serde_json::to_string(&still_plan).unwrap()).unwrap();
        assert_eq!(decoded, still_plan);

        let ads2v_req = GenerationRequest {
            prompt: "ads2v product".into(),
            width: 832,
            height: 480,
            frames: Some(17),
            video_mode: Some("ads2v".into()),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 4,
                    height: 4,
                    pixels: vec![0; 48],
                },
                strength: None,
            }],
            ..Default::default()
        };
        let vid_plan = build_bernini_render_plan(
            &full_desc,
            BerniniEntrypoint::Full,
            BerniniBackendKind::Mlx,
            BerniniPlanKnobs::default(),
            &ads2v_req,
        )
        .unwrap();
        assert_eq!(vid_plan.branch, BerniniCanonicalBranch::Ads2v);
        assert_eq!(vid_plan.trained_sources, 3.0);
        assert_eq!(
            vid_plan.output_contract,
            BerniniOutputContract::Video {
                fps: 16,
                latent_frames: 5
            }
        );
        assert_eq!(
            vid_plan.seed_scheme.permutation_scheme,
            "mlx-random-key(seed+0x4da4)-argsort"
        );
        assert_eq!(
            vid_plan.seed_scheme.planner_step_noise_scheme,
            "mlx-random-key(seed+0x9e37+step)-normal"
        );
    }

    #[test]
    fn bernini_plan_refuses_invalid_geometry_conditioning_and_mode_mismatches() {
        let full_desc = bernini_descriptor("bernini");
        let rend_desc = bernini_descriptor("bernini_renderer");

        // Video over MAX_AREA_14B (1024x1024 at 81 frames) fails, while 1-frame still succeeds.
        let over_area = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            frames: Some(81),
            ..Default::default()
        };
        assert!(build_bernini_render_plan(
            &full_desc,
            BerniniEntrypoint::Full,
            BerniniBackendKind::Mlx,
            BerniniPlanKnobs::default(),
            &over_area,
        )
        .is_err());

        // Bad frame grid (16 != 1 + 4*k).
        let bad_frames = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            frames: Some(16),
            ..Default::default()
        };
        assert!(build_bernini_render_plan(
            &rend_desc,
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &bad_frames,
        )
        .is_err());

        // Renderer text-only mode with conditioning rejected; conditioning mode without conditioning rejected.
        let img = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 48],
        };
        let t2v_with_img = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            frames: Some(5),
            video_mode: Some("t2v_apg".into()),
            conditioning: vec![Conditioning::Reference {
                image: img,
                strength: None,
            }],
            ..Default::default()
        };
        assert!(build_bernini_render_plan(
            &rend_desc,
            BerniniEntrypoint::Renderer,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &t2v_with_img,
        )
        .is_err());

        let v2v_no_cond = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            frames: Some(5),
            video_mode: Some("v2v".into()),
            ..Default::default()
        };
        assert!(build_bernini_render_plan(
            &full_desc,
            BerniniEntrypoint::Full,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &v2v_no_cond,
        )
        .is_err());
    }
}
