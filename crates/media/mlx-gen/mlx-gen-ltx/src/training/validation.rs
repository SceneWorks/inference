use super::*;

/// Executable upstream LTX-2.5 validation settings.  The individual guidance terms deliberately
/// stay separate: CFG, STG and modality isolation are different transformer evaluations in the dev
/// sampler and collapsing them to one generic image-preview scale changes the algorithm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25ValidationDefaults {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub video_cfg_scale: f32,
    pub audio_cfg_scale: f32,
    pub video_stg_scale: f32,
    pub audio_stg_scale: f32,
    pub stg_blocks: &'static [usize],
    pub guidance_rescale: f32,
    pub video_modality_guidance_scale: f32,
    pub audio_modality_guidance_scale: f32,
    pub generate_video: bool,
    pub generate_audio: bool,
}

pub const LTX25_VALIDATION_DEFAULTS: Ltx25ValidationDefaults = Ltx25ValidationDefaults {
    width: 960,
    height: 544,
    frames: 89,
    fps: 24,
    steps: 30,
    video_cfg_scale: 3.0,
    audio_cfg_scale: 7.0,
    video_stg_scale: 1.0,
    audio_stg_scale: 1.0,
    stg_blocks: &[28],
    guidance_rescale: 0.7,
    video_modality_guidance_scale: 3.0,
    audio_modality_guidance_scale: 3.0,
    generate_video: true,
    generate_audio: true,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25ValidationConfig {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub video_cfg_scale: f32,
    pub audio_cfg_scale: f32,
    pub video_stg_scale: f32,
    pub audio_stg_scale: f32,
    pub stg_blocks: Vec<usize>,
    pub guidance_rescale: f32,
    pub video_modality_guidance_scale: f32,
    pub audio_modality_guidance_scale: f32,
    pub generate_video: bool,
    pub generate_audio: bool,
}

impl From<Ltx25ValidationDefaults> for Ltx25ValidationConfig {
    fn from(value: Ltx25ValidationDefaults) -> Self {
        Self {
            width: value.width,
            height: value.height,
            frames: value.frames,
            fps: value.fps,
            steps: value.steps,
            video_cfg_scale: value.video_cfg_scale,
            audio_cfg_scale: value.audio_cfg_scale,
            video_stg_scale: value.video_stg_scale,
            audio_stg_scale: value.audio_stg_scale,
            stg_blocks: value.stg_blocks.to_vec(),
            guidance_rescale: value.guidance_rescale,
            video_modality_guidance_scale: value.video_modality_guidance_scale,
            audio_modality_guidance_scale: value.audio_modality_guidance_scale,
            generate_video: value.generate_video,
            generate_audio: value.generate_audio,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ltx25GuidancePlan {
    pub cfg_scale: f32,
    pub stg_scale: f32,
    pub rescale_scale: f32,
    pub modality_scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ltx25ValidationRenderPlan {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub sigmas: Vec<f32>,
    pub stg_blocks: Vec<usize>,
    pub video_guidance: Ltx25GuidancePlan,
    pub audio_guidance: Ltx25GuidancePlan,
    pub generate_audio: bool,
}

pub fn build_ltx25_validation_render_plan(
    config: &Ltx25ValidationConfig,
) -> Result<Ltx25ValidationRenderPlan> {
    if !config.generate_video {
        return Err("ltx_2_5 trainer: validation must generate video".into());
    }
    if config.width < 32
        || config.width > 4096
        || config.height < 32
        || config.height > 4096
        || !config.width.is_multiple_of(32)
        || !config.height.is_multiple_of(32)
    {
        return Err(
            "ltx_2_5 trainer: validation width/height must be 32..=4096 and 32px-aligned".into(),
        );
    }
    if config.frames == 0 || config.frames > 257 || config.frames % 8 != 1 {
        return Err(
            "ltx_2_5 trainer: validation frames must be 1..=257 with frames % 8 == 1".into(),
        );
    }
    if !(1..=120).contains(&config.fps) || !(1..=100).contains(&config.steps) {
        return Err(
            "ltx_2_5 trainer: validation fps must be 1..=120 and steps must be 1..=100".into(),
        );
    }
    if config.stg_blocks.is_empty()
        || config.stg_blocks.len() > 8
        || config.stg_blocks.iter().any(|&block| block >= 48)
    {
        return Err("ltx_2_5 trainer: stgBlocks must contain 1..=8 unique indices in 0..48".into());
    }
    let mut unique = config.stg_blocks.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != config.stg_blocks.len() {
        return Err("ltx_2_5 trainer: stgBlocks must not contain duplicates".into());
    }
    for (name, value, max) in [
        ("videoCfgScale", config.video_cfg_scale, 20.0),
        ("audioCfgScale", config.audio_cfg_scale, 20.0),
        ("videoStgScale", config.video_stg_scale, 20.0),
        ("audioStgScale", config.audio_stg_scale, 20.0),
        (
            "videoModalityGuidanceScale",
            config.video_modality_guidance_scale,
            20.0,
        ),
        (
            "audioModalityGuidanceScale",
            config.audio_modality_guidance_scale,
            20.0,
        ),
        ("guidanceRescale", config.guidance_rescale, 1.0),
    ] {
        if !value.is_finite() || value < 0.0 || value > max {
            return Err(format!("ltx_2_5 trainer: {name} must be finite and in [0,{max}]").into());
        }
    }
    Ok(Ltx25ValidationRenderPlan {
        width: config.width,
        height: config.height,
        frames: config.frames,
        fps: config.fps,
        sigmas: (0..=config.steps)
            .map(|index| 1.0 - index as f32 / config.steps as f32)
            .collect(),
        stg_blocks: config.stg_blocks.clone(),
        video_guidance: Ltx25GuidancePlan {
            cfg_scale: config.video_cfg_scale,
            stg_scale: config.video_stg_scale,
            rescale_scale: config.guidance_rescale,
            modality_scale: config.video_modality_guidance_scale,
        },
        audio_guidance: Ltx25GuidancePlan {
            cfg_scale: config.audio_cfg_scale,
            stg_scale: config.audio_stg_scale,
            rescale_scale: config.guidance_rescale,
            modality_scale: config.audio_modality_guidance_scale,
        },
        generate_audio: config.generate_audio,
    })
}

pub(super) fn parse_validation(value: Option<&serde_json::Value>) -> Result<Ltx25ValidationConfig> {
    let Some(value) = value else {
        return Ok(LTX25_VALIDATION_DEFAULTS.into());
    };
    let object = value.as_object().ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_5 trainer: `ltxValidation` must be an object".into())
    })?;
    let u32_or =
        |key: &str, default: u32| -> Result<u32> { Ok(json_u32(object, key)?.unwrap_or(default)) };
    let f32_or = |key: &str, default: f32| -> Result<f32> {
        let value = object
            .get(key)
            .map(|value| {
                value.as_f64().map(|value| value as f32).ok_or_else(|| {
                    mlx_gen::Error::Msg(format!("ltx_2_5 trainer: `{key}` must be numeric"))
                })
            })
            .transpose()?
            .unwrap_or(default);
        if !value.is_finite() {
            return Err(format!("ltx_2_5 trainer: `{key}` must be finite").into());
        }
        Ok(value)
    };
    let blocks: Vec<usize> = object
        .get("stgBlocks")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| {
                    mlx_gen::Error::Msg("ltx_2_5 trainer: `stgBlocks` must be an array".into())
                })?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(
                                "ltx_2_5 trainer: `stgBlocks` values must be u32".into(),
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![28]);
    let generate_audio = object
        .get("generateAudio")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: `generateAudio` must be boolean".into())
            })
        })
        .transpose()?
        .unwrap_or(true);
    let parsed = Ltx25ValidationConfig {
        width: u32_or("width", 960)?,
        height: u32_or("height", 544)?,
        frames: u32_or("frames", 89)?,
        fps: u32_or("fps", 24)?,
        steps: u32_or("steps", 30)?,
        video_cfg_scale: f32_or("videoCfgScale", 3.0)?,
        audio_cfg_scale: f32_or("audioCfgScale", 7.0)?,
        video_stg_scale: f32_or("videoStgScale", 1.0)?,
        audio_stg_scale: f32_or("audioStgScale", 1.0)?,
        stg_blocks: blocks,
        guidance_rescale: f32_or("guidanceRescale", 0.7)?,
        video_modality_guidance_scale: f32_or("videoModalityGuidanceScale", 3.0)?,
        audio_modality_guidance_scale: f32_or("audioModalityGuidanceScale", 3.0)?,
        generate_video: true,
        generate_audio,
    };
    build_ltx25_validation_render_plan(&parsed)?;
    Ok(parsed)
}

pub(super) fn validation_guider(plan: Ltx25GuidancePlan) -> crate::params::GuiderParams {
    crate::params::GuiderParams {
        cfg_scale: plan.cfg_scale,
        stg_scale: plan.stg_scale,
        // Block selection is executed by `AvPerturbation`; guidance combination consumes scales.
        stg_blocks: &[],
        rescale_scale: plan.rescale_scale,
        modality_scale: plan.modality_scale,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn denoise_ltx25_validation_av(
    dit: &AvDiT,
    video: &crate::conditioning::VideoTokenState,
    audio: &Array,
    video_context: &Array,
    audio_context: &Array,
    negative_video_context: &Array,
    negative_audio_context: &Array,
    audio_positions: &Array,
    plan: &Ltx25ValidationRenderPlan,
    cancel: &gen_core::CancelFlag,
) -> Result<crate::conditioning::VideoTokenState> {
    let dtype = video.latent.dtype();
    let shape = audio.shape();
    let (batch, channels, frames, bins) = (shape[0], shape[1], shape[2], shape[3]);
    let mut video_tokens = video.latent.clone();
    let mut audio_latents = audio.clone();
    let rope_epoch = Some(dit.next_rope_epoch());
    let stg = AvPerturbation::stg_blocks(&plan.stg_blocks)?;
    let video_guider = validation_guider(plan.video_guidance);
    let audio_guider = validation_guider(plan.audio_guidance);
    for (step, sigmas) in plan.sigmas.windows(2).enumerate() {
        if cancel.is_cancelled() {
            return Err(mlx_gen::Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[0], sigmas[1]);
        let audio_tokens = audio_latents.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            frames,
            channels * bins,
        ])?;
        let video_timestep =
            crate::conditioning::token_timesteps(&video.denoise_mask, video_tokens.dtype(), sigma)?;
        let sigma_array = Array::from_slice(&[sigma], &[1]).as_dtype(dtype)?;
        let audio_timestep = broadcast_to(&sigma_array, &[batch, frames])?;
        let forward = |vctx: &Array, actx: &Array, perturbation: AvPerturbation| {
            dit.forward_controlled(
                &video_tokens,
                &video_timestep,
                vctx,
                None,
                &video.positions,
                &audio_tokens,
                &audio_timestep,
                actx,
                None,
                audio_positions,
                video.keyframes_mask.as_ref(),
                rope_epoch,
                perturbation,
            )
        };
        let (conditional_video, conditional_audio) =
            forward(video_context, audio_context, AvPerturbation::NONE)?;
        let (unconditional_video, unconditional_audio) = forward(
            negative_video_context,
            negative_audio_context,
            AvPerturbation::NONE,
        )?;
        let (stg_video, stg_audio) = forward(video_context, audio_context, stg)?;
        let (isolated_video, isolated_audio) = forward(
            video_context,
            audio_context,
            AvPerturbation::modality_isolated(),
        )?;
        let denoised = |video_velocity: Array, audio_velocity: Array| -> Result<(Array, Array)> {
            let audio_velocity = audio_velocity
                .reshape(&[batch, frames, channels, bins])?
                .transpose_axes(&[0, 2, 1, 3])?;
            Ok((
                crate::transformer::to_denoised(&video_tokens, &video_velocity, &sigma_array)?,
                crate::transformer::to_denoised(&audio_latents, &audio_velocity, &sigma_array)?,
            ))
        };
        let (conditional_video, conditional_audio) =
            denoised(conditional_video, conditional_audio)?;
        let (unconditional_video, unconditional_audio) =
            denoised(unconditional_video, unconditional_audio)?;
        let (stg_video, stg_audio) = denoised(stg_video, stg_audio)?;
        let (isolated_video, isolated_audio) = denoised(isolated_video, isolated_audio)?;
        let guided_video = crate::dev_sampler::combine_guidance(
            &conditional_video,
            &unconditional_video,
            &stg_video,
            &isolated_video,
            video_guider,
        )?;
        let guided_video = crate::conditioning::apply_denoise_mask(
            &guided_video,
            &video.clean_latent,
            &video.denoise_mask,
        )?;
        let guided_audio = crate::dev_sampler::combine_guidance(
            &conditional_audio,
            &unconditional_audio,
            &stg_audio,
            &isolated_audio,
            audio_guider,
        )?;
        video_tokens =
            crate::pipeline::euler_step(&video_tokens, &guided_video, sigma, sigma_next)?;
        audio_latents =
            crate::pipeline::euler_step(&audio_latents, &guided_audio, sigma, sigma_next)?;
        eval([&video_tokens, &audio_latents])?;
        let _ = step;
    }
    Ok(crate::conditioning::VideoTokenState {
        latent: video_tokens,
        clean_latent: video.clean_latent.clone(),
        denoise_mask: video.denoise_mask.clone(),
        positions: video.positions.clone(),
        target_tokens: video.target_tokens,
        keyframes_mask: video.keyframes_mask.clone(),
        generated_keyframe_layout: video.generated_keyframe_layout.clone(),
    })
}

pub(super) fn denoise_ltx25_validation_video(
    dit: &AvDiT,
    video: &crate::conditioning::VideoTokenState,
    video_context: &Array,
    negative_video_context: &Array,
    plan: &Ltx25ValidationRenderPlan,
    cancel: &gen_core::CancelFlag,
) -> Result<crate::conditioning::VideoTokenState> {
    let mut tokens = video.latent.clone();
    let rope_epoch = Some(dit.next_rope_epoch());
    let stg = AvPerturbation::stg_blocks(&plan.stg_blocks)?;
    let guider = validation_guider(plan.video_guidance);
    for sigmas in plan.sigmas.windows(2) {
        if cancel.is_cancelled() {
            return Err(mlx_gen::Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[0], sigmas[1]);
        let timestep =
            crate::conditioning::token_timesteps(&video.denoise_mask, tokens.dtype(), sigma)?;
        let forward = |context: &Array, perturbation: AvPerturbation| {
            dit.forward_video_only_controlled(
                &tokens,
                &timestep,
                context,
                None,
                &video.positions,
                video.keyframes_mask.as_ref(),
                rope_epoch,
                perturbation,
            )
        };
        let conditional = forward(video_context, AvPerturbation::NONE)?;
        let unconditional = forward(negative_video_context, AvPerturbation::NONE)?;
        let perturbed = forward(video_context, stg)?;
        let sigma_array = Array::from_slice(&[sigma], &[1]).as_dtype(tokens.dtype())?;
        let conditional = crate::transformer::to_denoised(&tokens, &conditional, &sigma_array)?;
        let unconditional = crate::transformer::to_denoised(&tokens, &unconditional, &sigma_array)?;
        let perturbed = crate::transformer::to_denoised(&tokens, &perturbed, &sigma_array)?;
        // With no audio modality, the isolated-modality branch is the conditional branch and its
        // configured guidance term is therefore exactly zero.
        let guided = crate::dev_sampler::combine_guidance(
            &conditional,
            &unconditional,
            &perturbed,
            &conditional,
            guider,
        )?;
        let guided = crate::conditioning::apply_denoise_mask(
            &guided,
            &video.clean_latent,
            &video.denoise_mask,
        )?;
        tokens = crate::pipeline::euler_step(&tokens, &guided, sigma, sigma_next)?;
        eval([&tokens])?;
    }
    Ok(crate::conditioning::VideoTokenState {
        latent: tokens,
        clean_latent: video.clean_latent.clone(),
        denoise_mask: video.denoise_mask.clone(),
        positions: video.positions.clone(),
        target_tokens: video.target_tokens,
        keyframes_mask: video.keyframes_mask.clone(),
        generated_keyframe_layout: video.generated_keyframe_layout.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_ltx25_validation_sample(
    dit: &AvDiT,
    vae: &LtxVideoVae,
    video_context: &Array,
    audio_context: &Array,
    negative_video_context: &Array,
    negative_audio_context: &Array,
    config: &Ltx25ValidationConfig,
    seed: u64,
    cancel: &gen_core::CancelFlag,
) -> Result<Image> {
    let plan = build_ltx25_validation_render_plan(config)?;
    let latent_frames = ((plan.frames - 1) / crate::positions::TEMPORAL_SCALE as u32 + 1) as usize;
    let latent_height = (plan.height / SPATIAL_SCALE as u32) as usize;
    let latent_width = (plan.width / SPATIAL_SCALE as u32) as usize;
    if latent_height == 0
        || latent_width == 0
        || !plan.height.is_multiple_of(SPATIAL_SCALE as u32)
        || !plan.width.is_multiple_of(SPATIAL_SCALE as u32)
    {
        return Err("ltx_2_5 trainer: validation dimensions must be VAE aligned".into());
    }
    let video_noise = random::normal::<f32>(
        &[
            1,
            128,
            latent_frames as i32,
            latent_height as i32,
            latent_width as i32,
        ],
        None,
        None,
        Some(&random::key(seed)?),
    )?;
    let video_positions = create_position_grid_with(
        1,
        latent_frames,
        latent_height,
        latent_width,
        crate::positions::TEMPORAL_SCALE,
        SPATIAL_SCALE,
        plan.fps as f32,
        true,
    );
    let video_state = crate::conditioning::VideoTokenState::base(&video_noise, &video_positions)?;
    let video = if plan.generate_audio {
        let audio_frames =
            crate::positions::compute_audio_frames(plan.frames as usize, plan.fps as f64);
        let audio_noise = random::normal::<f32>(
            &[
                1,
                crate::positions::AUDIO_LATENT_CHANNELS as i32,
                audio_frames as i32,
                crate::positions::AUDIO_MEL_BINS as i32,
            ],
            None,
            None,
            Some(&random::key(seed.wrapping_add(1))?),
        )?;
        let audio_positions = create_audio_position_grid(1, audio_frames);
        denoise_ltx25_validation_av(
            dit,
            &video_state,
            &audio_noise,
            video_context,
            audio_context,
            negative_video_context,
            negative_audio_context,
            &audio_positions,
            &plan,
            cancel,
        )?
    } else {
        denoise_ltx25_validation_video(
            dit,
            &video_state,
            video_context,
            negative_video_context,
            &plan,
            cancel,
        )?
    };
    let latents = crate::conditioning::unpatchify_grid(
        &video.latent,
        128,
        latent_frames as i32,
        latent_height as i32,
        latent_width as i32,
    )?;
    let frames = crate::pipeline::decode_to_frames(vae, &latents, cancel)?;
    let shape = frames.shape();
    if shape.len() != 4 || shape[0] == 0 || shape[3] != 3 {
        return Err(
            format!("ltx_2_5 trainer: validation decode returned invalid shape {shape:?}").into(),
        );
    }
    let (height, width) = (shape[1] as usize, shape[2] as usize);
    let count = height * width * 3;
    let first = frames.reshape(&[-1])?;
    eval([&first])?;
    Ok(Image {
        width: width as u32,
        height: height as u32,
        pixels: first.as_slice::<u8>()[..count].to_vec(),
    })
}
