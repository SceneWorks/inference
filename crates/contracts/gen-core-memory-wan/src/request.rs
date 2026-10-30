//! Request modes, carriers and request validation.

use super::*;

pub(super) fn reference(request: &GenerationRequest) -> gen_core::Result<&gen_core::Image> {
    if request.conditioning.len() != 1 {
        return Err(gen_core::Error::Unsupported(
            "Wan public I2V requires exactly one conditioning carrier".to_owned(),
        ));
    }
    match &request.conditioning[0] {
        Conditioning::Reference { image, strength }
            if strength.is_none_or(|value| value.to_bits() == 1.0_f32.to_bits()) =>
        {
            Ok(image)
        }
        _ => Err(gen_core::Error::Unsupported(
            "Wan public I2V requires one full-strength Reference".to_owned(),
        )),
    }
}

pub(super) fn valid_rgb8_image(image: &gen_core::Image) -> bool {
    image.width > 0
        && image.height > 0
        && image
            .width
            .checked_mul(image.height)
            .and_then(|pixels| pixels.checked_mul(3))
            .is_some_and(|bytes| bytes as usize == image.pixels.len())
}

pub(super) fn first_last_keyframes(
    request: &GenerationRequest,
) -> gen_core::Result<[gen_core::KeyframeRef<'_>; 2]> {
    let [Conditioning::Keyframe {
        image: first_image,
        frame_idx: first_index,
        strength: first_strength,
    }, Conditioning::Keyframe {
        image: last_image,
        frame_idx: last_index,
        strength: last_strength,
    }] = request.conditioning.as_slice()
    else {
        return Err(gen_core::Error::Unsupported(
            "Wan public first_last_frame requires exactly two ordered Keyframes".to_owned(),
        ));
    };
    if *first_index != 0
        || *last_index != -1
        || !first_strength.is_finite()
        || !last_strength.is_finite()
        || !(0.0..=1.0).contains(first_strength)
        || !(0.0..=1.0).contains(last_strength)
        || !valid_rgb8_image(first_image)
        || !valid_rgb8_image(last_image)
        || first_image.width != request.width
        || first_image.height != request.height
        || last_image.width != request.width
        || last_image.height != request.height
    {
        return Err(gen_core::Error::Unsupported(
            "Wan public first_last_frame requires ordered fitted RGB8 first@0/last@-1 Keyframes"
                .to_owned(),
        ));
    }
    Ok([
        gen_core::KeyframeRef {
            image: first_image,
            frame_idx: *first_index,
            strength: *first_strength,
        },
        gen_core::KeyframeRef {
            image: last_image,
            frame_idx: *last_index,
            strength: *last_strength,
        },
    ])
}

pub(super) fn extend_keyframe(
    request: &GenerationRequest,
) -> gen_core::Result<gen_core::KeyframeRef<'_>> {
    let [Conditioning::Keyframe {
        image,
        frame_idx,
        strength,
    }] = request.conditioning.as_slice()
    else {
        return Err(gen_core::Error::Unsupported(
            "Wan public extend_clip fallback requires exactly one boundary Keyframe".to_owned(),
        ));
    };
    if *frame_idx != 0
        || !strength.is_finite()
        || !(0.0..=1.0).contains(strength)
        || !valid_rgb8_image(image)
        || image.width != request.width
        || image.height != request.height
    {
        return Err(gen_core::Error::Unsupported(
            "Wan public extend_clip fallback requires one fitted RGB8 Keyframe at frame 0"
                .to_owned(),
        ));
    }
    Ok(gen_core::KeyframeRef {
        image,
        frame_idx: *frame_idx,
        strength: *strength,
    })
}

pub(super) fn bridge_keyframes(
    request: &GenerationRequest,
) -> gen_core::Result<[gen_core::KeyframeRef<'_>; 2]> {
    first_last_keyframes(request).map_err(|_| {
        gen_core::Error::Unsupported(
            "Wan video_bridge fallback requires exactly two ordered fitted RGB8 Keyframes at 0/-1"
                .to_owned(),
        )
    })
}

pub(super) fn extend_control_clip(
    request: &GenerationRequest,
) -> gen_core::Result<gen_core::ControlClipRef<'_>> {
    if request.conditioning.len() != 1 {
        return Err(gen_core::Error::Unsupported(
            "Wan VACE extend_clip requires exactly one ControlClip".to_owned(),
        ));
    }
    let clip = request.control_clip().ok_or_else(|| {
        gen_core::Error::Unsupported(
            "Wan VACE extend_clip requires exactly one ControlClip".to_owned(),
        )
    })?;
    let exact_image = |image: &gen_core::Image| {
        valid_rgb8_image(image) && image.width == request.width && image.height == request.height
    };
    if clip.frames.len() != request.frames.unwrap_or_default() as usize
        || clip.frames.len() != clip.mask.len()
        || clip.frames.is_empty()
        || !clip.frames.iter().all(exact_image)
        || !clip.mask.iter().all(exact_image)
        || !clip.masking_strength.is_finite()
        || clip.masking_strength.to_bits() != 1.0_f32.to_bits()
        || clip.start_frame != 0
        || clip.mode != gen_core::ReplacementMode::default()
    {
        return Err(gen_core::Error::Unsupported(
            "Wan VACE extend_clip ControlClip left the exact output-sized carrier envelope"
                .to_owned(),
        ));
    }
    Ok(clip)
}

/// The raw RGB value the Wan-VACE builder writes into a white-mask bridge gap.  The engine maps
/// this mid-gray value to approximately zero before denoising; black is an anchor-mask value, not
/// neutral generated content.
pub(super) const WAN_VACE_BRIDGE_NEUTRAL_RGB: u8 = 128;

pub(super) fn bridge_control_clip(
    request: &GenerationRequest,
) -> gen_core::Result<gen_core::ControlClipRef<'_>> {
    let clip = extend_control_clip(request).map_err(|_| {
        gen_core::Error::Unsupported(
            "Wan VACE video_bridge requires one full output ControlClip with a fitted mask"
                .to_owned(),
        )
    })?;
    // A bridge owns two immutable source boundaries and generates only the middle.  ControlClip
    // follows the VACE replacement convention: black keeps an anchor; white regenerates.  Require
    // a non-empty left tail and right head (which may be asymmetric), plus an all-neutral generated
    // gap, so an arbitrary edit control cannot masquerade as a bridge carrier.
    let is_uniform =
        |image: &gen_core::Image, value| image.pixels.iter().all(|&pixel| pixel == value);
    let left_anchor_len = clip
        .mask
        .iter()
        .take_while(|mask| is_uniform(mask, 0))
        .count();
    let right_anchor_len = clip
        .mask
        .iter()
        .rev()
        .take_while(|mask| is_uniform(mask, 0))
        .count();
    let gap_end = clip.frames.len().saturating_sub(right_anchor_len);
    if left_anchor_len == 0
        || right_anchor_len == 0
        || left_anchor_len >= gap_end
        || !clip.mask[left_anchor_len..gap_end]
            .iter()
            .all(|mask| is_uniform(mask, 255))
        || !clip.frames[left_anchor_len..gap_end]
            .iter()
            .all(|frame| is_uniform(frame, WAN_VACE_BRIDGE_NEUTRAL_RGB))
    {
        return Err(gen_core::Error::Unsupported(
            "Wan VACE video_bridge requires black left-tail/right-head anchors and a white-mask neutral generated gap"
                .to_owned(),
        ));
    }
    Ok(clip)
}

pub(super) fn replace_person_carrier(
    request: &GenerationRequest,
) -> gen_core::Result<(gen_core::ControlClipRef<'_>, Vec<&gen_core::Image>)> {
    let Some((Conditioning::ControlClip { .. }, references)) = request.conditioning.split_first()
    else {
        return Err(gen_core::Error::Unsupported(
            "Wan replace_person requires ControlClip first, followed by ordered References"
                .to_owned(),
        ));
    };
    if !(1..=4).contains(&references.len()) {
        return Err(gen_core::Error::Unsupported(
            "Wan replace_person requires one to four ordered References".to_owned(),
        ));
    }
    let clip = request.control_clip().ok_or_else(|| {
        gen_core::Error::Unsupported(
            "Wan replace_person requires exactly one ControlClip".to_owned(),
        )
    })?;
    let exact_output = |image: &gen_core::Image| {
        valid_rgb8_image(image) && image.width == request.width && image.height == request.height
    };
    if clip.frames.len() != request.frames.unwrap_or_default() as usize
        || clip.frames.len() != clip.mask.len()
        || clip.frames.is_empty()
        || !clip.frames.iter().all(exact_output)
        || !clip.mask.iter().all(exact_output)
        || !clip.masking_strength.is_finite()
        || !(0.0..=1.0).contains(&clip.masking_strength)
        || clip.start_frame != 0
    {
        return Err(gen_core::Error::Unsupported(
            "Wan replace_person ControlClip must be RGB8 output geometry for every output frame with an equal RGB8 mask and finite masking strength"
                .to_owned(),
        ));
    }
    let references = references
        .iter()
        .map(|conditioning| match conditioning {
            Conditioning::Reference { image, strength: None } if valid_rgb8_image(image) => {
                Ok(image)
            }
            _ => Err(gen_core::Error::Unsupported(
                "Wan replace_person references must be ordered approved RGB8 References without per-reference strength"
                    .to_owned(),
            )),
        })
        .collect::<gen_core::Result<Vec<_>>>()?;
    Ok((clip, references))
}

/// The text-to-video carrier: a prompt and **nothing else**.
///
/// `T2v14b` is the only route that admits it, and it admits no other carrier — so this is where an
/// image sneaked onto a text-only route is refused, rather than silently priced against a resident
/// set that never materializes a VAE encoder.
pub(super) fn text_to_video_carrier(request: &GenerationRequest) -> gen_core::Result<()> {
    if !request.conditioning.is_empty() || request.control_clip().is_some() {
        return Err(gen_core::Error::Unsupported(
            "Wan public text_to_video admits no conditioning carrier".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn request_mode(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<WanPublicMode> {
    match request.video_mode.as_deref() {
        // A text-only route publishes its mode either way: the worker may spell it, or leave
        // `video_mode` unset the way the T2V generator's own request path does.
        None | Some("text_to_video") if prepared.route == WanI2vRoute::T2v14b => {
            text_to_video_carrier(request)?;
            Ok(WanPublicMode::TextToVideo)
        }
        Some("image_to_video")
            if prepared.route != WanI2vRoute::Vace && prepared.route != WanI2vRoute::T2v14b =>
        {
            reference(request)?;
            Ok(WanPublicMode::ImageToVideo)
        }
        None if prepared.route == WanI2vRoute::Ti2v5b => {
            first_last_keyframes(request)?;
            Ok(WanPublicMode::FirstLastFrame)
        }
        Some("extend_clip")
            if prepared.route == WanI2vRoute::Ti2v5b && prepared.backend == WanI2vBackend::Mlx =>
        {
            extend_keyframe(request)?;
            Ok(WanPublicMode::ExtendClip)
        }
        Some("extend_clip") if prepared.route == WanI2vRoute::Vace => {
            extend_control_clip(request)?;
            Ok(WanPublicMode::ExtendClip)
        }
        Some("video_bridge")
            if prepared.route == WanI2vRoute::Ti2v5b && prepared.backend == WanI2vBackend::Mlx =>
        {
            bridge_keyframes(request)?;
            Ok(WanPublicMode::VideoBridge)
        }
        Some("video_bridge") if prepared.route == WanI2vRoute::Vace => {
            bridge_control_clip(request)?;
            Ok(WanPublicMode::VideoBridge)
        }
        Some(mode)
            if mode
                .strip_prefix("replace_person@")
                .is_some_and(|identity| {
                    identity.len() == 64 && identity.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
                && matches!(prepared.route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) =>
        {
            replace_person_carrier(request)?;
            Ok(WanPublicMode::ReplacePerson)
        }
        _ => Err(gen_core::Error::Unsupported(format!(
            "{}: request matches no exact public Wan carrier — this route admits \
             text_to_video/no carrier (T2V-A14B only), \
             image_to_video/one Reference (non-VACE, non-T2V), first_last_frame/two ordered Keyframes \
             (Ti2V-5B, no video_mode), extend_clip and video_bridge (Ti2V-5B MLX keyframes or \
             VACE ControlClip), and replace_person@<64-hex identity> (VACE / VACE-Fun)",
            prepared.route.provider_id()
        ))),
    }
}

pub(super) fn request_contract_for_mode(
    prepared: &PreparedWanI2vMemory,
    mode: WanPublicMode,
) -> MemoryProviderContract {
    let mut contract = prepared.contract.clone();
    if (mode == WanPublicMode::FirstLastFrame && prepared.backend == WanI2vBackend::Mlx)
        || mode == WanPublicMode::ExtendClip
        || mode == WanPublicMode::VideoBridge
        || (mode == WanPublicMode::ReplacePerson && prepared.route == WanI2vRoute::Vace)
    {
        let staged = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("Wan strategy table is complete");
        staged.support = MemoryStrategySupport::Missing;
        staged.parameters = MemoryParameterRanges::default();
    }
    contract
}

pub fn contract_for_mode_key(
    prepared: &PreparedWanI2vMemory,
    mode: &str,
) -> gen_core::Result<MemoryProviderContract> {
    let mode = match mode {
        "text_to_video" if prepared.route == WanI2vRoute::T2v14b => WanPublicMode::TextToVideo,
        "image_to_video"
            if prepared.route != WanI2vRoute::Vace && prepared.route != WanI2vRoute::T2v14b =>
        {
            WanPublicMode::ImageToVideo
        }
        "first_last_frame" if prepared.route == WanI2vRoute::Ti2v5b => {
            WanPublicMode::FirstLastFrame
        }
        "extend_clip"
            if prepared.route == WanI2vRoute::Vace
                || (prepared.route == WanI2vRoute::Ti2v5b
                    && prepared.backend == WanI2vBackend::Mlx) =>
        {
            WanPublicMode::ExtendClip
        }
        "video_bridge"
            if prepared.route == WanI2vRoute::Vace
                || (prepared.route == WanI2vRoute::Ti2v5b
                    && prepared.backend == WanI2vBackend::Mlx) =>
        {
            WanPublicMode::VideoBridge
        }
        "replace_person" if matches!(prepared.route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) => {
            WanPublicMode::ReplacePerson
        }
        _ => {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: unsupported Wan memory mode {mode}",
                prepared.route.provider_id()
            )))
        }
    };
    Ok(request_contract_for_mode(prepared, mode))
}

/// The request-specific ladder. Extend-clip exposes only Resident and BoundedDecode on both
/// backends. MLX first/last-frame does the same; Candle first/last-frame additionally retains
/// StagedResidency because its request-selected Sequential renderer is operationally distinct from
/// Resident.
pub fn request_contract(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<MemoryProviderContract> {
    let mode = validate_request(prepared, request)?;
    Ok(request_contract_for_mode(prepared, mode))
}

pub fn validate_request(
    prepared: &PreparedWanI2vMemory,
    request: &GenerationRequest,
) -> gen_core::Result<WanPublicMode> {
    let mode = request_mode(prepared, request)?;
    let frames = request.frames.unwrap_or_default();
    let fps = request.fps.unwrap_or_default();
    let steps = request.steps.unwrap_or_default();
    let unsupported_svd = request.motion_bucket_id.is_some()
        || request.noise_aug_strength.is_some()
        || request.decode_chunk_size.is_some()
        || request.conditioning_fps.is_some();
    let has_lightning = prepared.adapters.first().is_some_and(|receipt| {
        receipt.kind == "lightning_lora"
            && receipt.expert == "high"
            && prepared
                .adapters
                .get(1)
                .is_some_and(|low| low.kind == "lightning_lora" && low.expert == "low")
    });
    let lightning_sampling = if lightning_subdir(prepared.route).is_some()
        && steps == 4
        && request
            .guidance
            .is_some_and(|value| value.to_bits() == 1.0_f32.to_bits())
    {
        has_lightning
    } else {
        !has_lightning
    };
    let exact_rate = match mode {
        WanPublicMode::TextToVideo | WanPublicMode::ImageToVideo => {
            prepared.route.accepts_rate(fps, frames)
        }
        WanPublicMode::FirstLastFrame => prepared.route.accepts_first_last_rate(fps, frames),
        WanPublicMode::ExtendClip => match prepared.route {
            WanI2vRoute::Ti2v5b => prepared.route.accepts_first_last_rate(fps, frames),
            WanI2vRoute::Vace => prepared.route.accepts_rate(fps, frames),
            WanI2vRoute::T2v14b | WanI2vRoute::I2v14b | WanI2vRoute::VaceFun => false,
        },
        WanPublicMode::VideoBridge => match prepared.route {
            WanI2vRoute::Ti2v5b => prepared.route.accepts_first_last_rate(fps, frames),
            WanI2vRoute::Vace => prepared.route.accepts_rate(fps, frames),
            WanI2vRoute::T2v14b | WanI2vRoute::I2v14b | WanI2vRoute::VaceFun => false,
        },
        WanPublicMode::ReplacePerson => {
            matches!(prepared.route, WanI2vRoute::Vace | WanI2vRoute::VaceFun)
                && prepared.route.accepts_rate(fps, frames)
        }
    };
    let sampler_ok = match mode {
        WanPublicMode::TextToVideo | WanPublicMode::ImageToVideo => true,
        WanPublicMode::FirstLastFrame
        | WanPublicMode::ExtendClip
        | WanPublicMode::VideoBridge
        | WanPublicMode::ReplacePerson => match request.sampler.as_deref() {
            None | Some("uni_pc" | "euler") => true,
            Some("dpmpp_2m") => prepared.backend == WanI2vBackend::Mlx,
            Some(_) => false,
        },
    };
    let boundary_axes_ok = !matches!(
        mode,
        WanPublicMode::FirstLastFrame
            | WanPublicMode::ExtendClip
            | WanPublicMode::VideoBridge
            | WanPublicMode::ReplacePerson
    ) || (request.strength.is_none()
        && request.true_cfg.is_none()
        && request.timestep_to_start_cfg.is_none()
        && request.guidance_method.is_none()
        && request.guidance_eta.is_none()
        && request.guidance_momentum.is_none()
        && request.guidance_norm_threshold.is_none()
        && (request.control_scale.is_none()
            || (matches!(prepared.route, WanI2vRoute::Vace | WanI2vRoute::VaceFun)
                && request
                    .control_scale
                    .is_some_and(|value| value.is_finite() && value >= 0.0)))
        && request.text_style_gain.is_none()
        && request.image_guidance.is_none()
        && request.duration.is_none()
        && request.trim_first_frames.is_none()
        && request.softness.is_none()
        && request.pid_capture_sigma.is_none()
        && request.approximation.is_none());
    if request.count != 1
        || !prepared
            .route
            .public_geometries()
            .contains(&(request.width, request.height))
        || !exact_rate
        || !(1..=100).contains(&steps)
        || request.seed.is_none()
        || request.audio.is_some()
        || request.phases.is_some()
        || request.scheduler.is_some()
        || request.scheduler_shift.is_some()
        || request.use_pid
        || request.enhance_prompt
        || request.use_uncensored_enhancer
        || unsupported_svd
        || !lightning_sampling
        || !sampler_ok
        || !boundary_axes_ok
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request left the exact public {} envelope",
            prepared.route.provider_id(),
            mode.key()
        )));
    }
    Ok(mode)
}
