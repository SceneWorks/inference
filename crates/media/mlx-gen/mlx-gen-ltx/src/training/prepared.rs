use super::*;

#[derive(Clone)]
pub(super) struct CachedLtx25Modality {
    pub(super) clean: Array,
    pub(super) context: Array,
    pub(super) positions: Array,
    pub(super) reference: Option<Array>,
    pub(super) reference_positions: Option<Array>,
    pub(super) token_plan: Ltx25TokenPlan,
}

#[derive(Clone, Default)]
pub(super) struct CachedLtx25Example {
    pub(super) video: Option<CachedLtx25Modality>,
    pub(super) audio: Option<CachedLtx25Modality>,
    /// The generated video clip's depth-anchoring frame selection (sc-24830); `None` when depth
    /// anchoring is off (or the example generates no video).
    pub(super) depth: Option<Ltx25DepthFrames>,
}

/// Which latent frames of one example's video clip the depth loss decodes, and the clip's
/// unpatchified grid (sc-24830).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Ltx25DepthFrames {
    /// `(C, F, H, W)` of the patchified target video tokens.
    pub(super) grid: [i32; 4],
    /// Selected latent frame indices (ascending).
    pub(super) frames: Vec<i32>,
}

impl Ltx25DepthFrames {
    /// The latent frames [`Ltx25DepthDecoder`] decodes for the selected (ascending) frames: frame
    /// `0` alone (a `T = 1` clip — it decodes to one pixel frame), every frame `k > 0` preceded by
    /// `k − 1` (a `T = 2` clip, so TAEHV's MemBlocks see the frame its 8-pixel-frame group
    /// follows). Frame 0 can only lead, so the layout is `[0]? ++ [k−1, k]*`.
    pub(super) fn decode_frames(&self) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(2 * self.frames.len());
        for (i, &k) in self.frames.iter().enumerate() {
            match k {
                0 if i == 0 => out.push(0),
                k if k > 0 => out.extend([k - 1, k]),
                _ => {
                    return Err(format!(
                        "ltx_2_5 trainer: depth frames must be ascending and non-negative, got \
                         {:?}",
                        self.frames
                    )
                    .into())
                }
            }
        }
        Ok(out)
    }

    /// The selected frames of the patchified target tokens `(1, F·H·W, C)` as the decoder's NCHW
    /// batch — the model-space (normalized) latent TAELTX2.3 decodes, in the
    /// [`decode_frames`](Self::decode_frames) layout [`Ltx25DepthDecoder`] reads back as clips (one
    /// decoded frame per selected frame).
    pub(super) fn nchw(&self, tokens: &Array) -> Result<Array> {
        let [c, f, h, w] = self.grid;
        let grid = crate::conditioning::unpatchify_grid(tokens, c, f, h, w)?;
        let decode = self.decode_frames()?;
        let n = decode.len() as i32;
        let index = Array::from_slice(&decode, &[n]);
        Ok(grid
            .take_axis(&index, 2)?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[n, c, h, w])?)
    }
}

/// LTX-2.5's TAELTX2.3 depth decoder (sc-24830). A video latent frame `k > 0` encodes a group of
/// 8 pixel frames that TAEHV decodes with temporal memory of frame `k − 1`; decoding it as a lone
/// `T = 1` clip would be out of distribution. So each selected frame `k > 0` decodes together with
/// its predecessor as a `T = 2` clip and keeps only the clip's last output frame (latent `k`'s last
/// grown frame — the same frame a `T = 1` decode keeps), while frame 0 stays a `T = 1` clip. Input:
/// the [`Ltx25DepthFrames::nchw`] batch `[0]? ++ [k−1, k]*` (odd length ⇔ a leading lone frame 0);
/// output: NHWC, one frame per selected frame, in order. Applied identically to the live x0 and to
/// the clean reference, so their shapes match.
pub(super) struct Ltx25DepthDecoder(pub(super) TaehvDecoder);

impl X0Decoder for Ltx25DepthDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        use mlx_rs::ops::indexing::IndexOp;
        let sh = latents.shape().to_vec();
        if sh.len() != 4 {
            return Err(format!(
                "ltx_2_5 trainer: the depth decoder expects NCHW latents, got shape {sh:?}"
            )
            .into());
        }
        let (n, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let lone = n % 2;
        let pairs = n / 2;
        let mut parts = Vec::with_capacity(2);
        if lone == 1 {
            parts.push(self.0.decode_frames(&latents.index(..1))?);
        }
        if pairs > 0 {
            let clips = latents.index(lone..).reshape(&[pairs, 2, c, h, w])?;
            parts.push(self.0.decode_clip_last_frames(&clips)?);
        }
        let parts: Vec<&Array> = parts.iter().collect();
        Ok(concatenate_axis(&parts, 0)?)
    }
}

/// The [`Ltx25DepthDecoder`] for the shared builder: TAELTX2.3 from `perceptual_decoder_dir`, and a
/// per-selected-frame working set of one `T = 2` clip decode (an upper bound: frame 0 decodes as
/// `T = 1`).
pub(super) struct Ltx25DepthDecoderSpec;

impl mlx_gen_perceptual::CustomDecoder for Ltx25DepthDecoderSpec {
    fn name(&self) -> &'static str {
        "TAELTX2.3"
    }

    fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint {
        let cfg = TaehvConfig::taeltx2_3();
        AuxModelFootprint {
            working_set_bytes: cfg.clip_training_working_set_bytes(h, w, 2),
            ..cfg.footprint(h, w)
        }
    }

    fn load(&self, dir: Option<&Path>) -> Result<Box<dyn X0Decoder>> {
        let dir = dir.ok_or_else(|| {
            mlx_gen::Error::Msg(
                "the perceptual losses need the TAELTX2.3 decoder (perceptual_decoder_dir)".into(),
            )
        })?;
        let dec = TaehvDecoder::from_path(dir, TaehvConfig::taeltx2_3())
            .map_err(|e| mlx_gen::Error::Msg(format!("{}: {e}", dir.display())))?;
        Ok(Box::new(Ltx25DepthDecoder(dec)))
    }
}

/// LTX-2.5's decoder for the shared aux-loss builder (sc-24830): TAELTX2.3 through
/// [`Ltx25DepthDecoder`].
pub(super) fn ltx25_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Custom(Box::new(Ltx25DepthDecoderSpec))
}

pub(super) fn parse_shape_metadata(
    weights: &Weights,
    key: &str,
    tensor: &Array,
) -> Result<Vec<i32>> {
    let raw = weights.metadata(key).ok_or_else(|| {
        mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: prepared bundle is missing `{key}` metadata"
        ))
    })?;
    let values: Vec<i32> = serde_json::from_str(raw).map_err(|error| {
        mlx_gen::Error::Msg(format!(
            "ltx_2_5 trainer: prepared bundle `{key}` is not a JSON shape: {error}"
        ))
    })?;
    if values != tensor.shape() {
        return Err(format!(
            "ltx_2_5 trainer: prepared bundle `{key}` {values:?} does not match tensor shape {:?}",
            tensor.shape()
        )
        .into());
    }
    Ok(values)
}

pub(super) fn prepared_fps(weights: &Weights) -> Result<f64> {
    let fps = weights
        .metadata("fps")
        .ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: video bundle is missing `fps` metadata".into())
        })?
        .parse::<f64>()
        .map_err(|_| mlx_gen::Error::Msg("ltx_2_5 trainer: `fps` must be numeric".into()))?;
    if !fps.is_finite() || fps <= 0.0 || fps > f32::MAX as f64 {
        return Err("ltx_2_5 trainer: `fps` must be finite, > 0, and representable as f32".into());
    }
    Ok(fps)
}

pub(super) fn validate_prepared_conditions(
    weights: &Weights,
    modality: &Ltx25ResolvedModality,
    main_shape: &[i32],
    target_tokens: usize,
    is_video: bool,
) -> Result<()> {
    for condition in &modality.conditions {
        let Some(key) = condition.tensor_key.as_deref() else {
            continue;
        };
        let tensor = weights.require(key)?;
        match condition.spec.kind {
            Ltx25ConditionKind::Mask => {
                if tensor.size() != target_tokens {
                    return Err(format!(
                        "ltx_2_5 trainer: mask tensor `{key}` has {} values, expected {target_tokens}",
                        tensor.size()
                    )
                    .into());
                }
            }
            Ltx25ConditionKind::Reference if is_video => {
                let shape = tensor.shape();
                if shape.len() != 5 || shape[0] != main_shape[0] || shape[1] != main_shape[1] {
                    return Err(format!(
                        "ltx_2_5 trainer: video reference `{key}` must be [B,C,F,H,W] with B/C matching video_latents, got {shape:?}"
                    )
                    .into());
                }
            }
            Ltx25ConditionKind::Reference => {
                let shape = tensor.shape();
                if shape.len() != 4
                    || shape[0] != main_shape[0]
                    || shape[1] != main_shape[1]
                    || shape[3] != main_shape[3]
                {
                    return Err(format!(
                        "ltx_2_5 trainer: audio reference `{key}` must be [B,C,T,F] with B/C/F matching audio_latents, got {shape:?}"
                    )
                    .into());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn validate_prepared_bundle_contract(
    weights: &Weights,
    path: &Path,
    plan: &Ltx25TrainingPlan,
) -> Result<()> {
    if weights.metadata("schemaVersion") != Some("ltx-prepared-v1") {
        return Err(format!(
            "ltx_2_5 trainer: prepared bundle {} must declare schemaVersion=ltx-prepared-v1",
            path.display()
        )
        .into());
    }
    let video_timing = if let Some(modality) = &plan.video {
        let raw = weights.require("video_latents")?;
        let shape = parse_shape_metadata(weights, "videoShape", raw)?;
        let [batch, channels, frames, height, width]: [i32; 5] =
            shape.clone().try_into().map_err(|_| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: videoShape must be [B,C,F,H,W]".into())
            })?;
        if batch != 1 || channels <= 0 || frames <= 0 || height <= 0 || width <= 0 {
            return Err(
                "ltx_2_5 trainer: videoShape must be positive with batch dimension 1".into(),
            );
        }
        if channels != LTX25_VIDEO_LATENT_CHANNELS {
            return Err(format!(
                "ltx_2_5 trainer: video_latents has {channels} channels, transformer expects {}",
                LTX25_VIDEO_LATENT_CHANNELS
            )
            .into());
        }
        let fps = prepared_fps(weights)?;
        validate_prepared_conditions(
            weights,
            modality,
            &shape,
            frames as usize * height as usize * width as usize,
            true,
        )?;
        Some((frames as usize, fps))
    } else {
        None
    };
    let audio_frames = if let Some(modality) = &plan.audio {
        let raw = weights.require("audio_latents")?;
        let shape = parse_shape_metadata(weights, "audioShape", raw)?;
        let [batch, channels, frames, bins]: [i32; 4] = shape.clone().try_into().map_err(|_| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: audioShape must be [B,C,T,F]".into())
        })?;
        if batch != 1 || channels <= 0 || frames <= 0 || bins <= 0 {
            return Err(
                "ltx_2_5 trainer: audioShape must be positive with batch dimension 1".into(),
            );
        }
        if channels * bins != LTX25_AUDIO_FLAT_CHANNELS {
            return Err(format!(
                "ltx_2_5 trainer: flattened audio_latents has {} channels, transformer expects {}",
                channels * bins,
                LTX25_AUDIO_FLAT_CHANNELS
            )
            .into());
        }
        validate_prepared_conditions(weights, modality, &shape, frames as usize, false)?;
        Some(frames as usize)
    } else {
        None
    };
    if let (Some((video_latent_frames, fps)), Some(audio_frames)) = (video_timing, audio_frames) {
        let output_frames = (video_latent_frames - 1)
            .checked_mul(crate::positions::TEMPORAL_SCALE as usize)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: videoShape duration overflows frame mapping".into(),
                )
            })?;
        let expected = crate::positions::compute_audio_frames(output_frames, fps);
        if audio_frames != expected {
            return Err(format!(
                "ltx_2_5 trainer: audioShape duration has {audio_frames} latent frames, expected {expected} for videoShape {video_latent_frames} latent frames at {fps} fps"
            )
            .into());
        }
    }
    Ok(())
}

pub(super) fn prepared_bundle_path(item: &mlx_gen::TrainingItem) -> Result<&Path> {
    item.model_options
        .get("ltxPreparedBundlePath")
        .and_then(serde_json::Value::as_str)
        .filter(|path| !path.is_empty())
        .map(Path::new)
        .ok_or_else(|| {
            mlx_gen::Error::Msg(
                "ltx_2_5 trainer: every item requires `ltxPreparedBundlePath`".into(),
            )
        })
}

pub(super) fn reference_condition(
    modality: &Ltx25ResolvedModality,
) -> Option<&Ltx25ResolvedCondition> {
    modality
        .conditions
        .iter()
        .find(|condition| condition.spec.kind == Ltx25ConditionKind::Reference)
}

pub(super) fn mask_condition(modality: &Ltx25ResolvedModality) -> Option<&Ltx25ResolvedCondition> {
    modality
        .conditions
        .iter()
        .find(|condition| condition.spec.kind == Ltx25ConditionKind::Mask)
}

pub(super) fn condition_mask_values(
    weights: &Weights,
    condition: Option<&Ltx25ResolvedCondition>,
    target_tokens: usize,
) -> Result<Option<Vec<bool>>> {
    let Some(condition) = condition else {
        return Ok(None);
    };
    let key = condition.tensor_key.as_deref().ok_or_else(|| {
        mlx_gen::Error::Msg("ltx_2_5 trainer: mask condition has no tensorKey".into())
    })?;
    let values = to_dtype(weights.require(key)?, Dtype::Float32)?;
    eval([&values])?;
    if values.size() != target_tokens {
        return Err(format!(
            "ltx_2_5 trainer: mask tensor `{key}` has {} values, expected {target_tokens}",
            values.size()
        )
        .into());
    }
    Ok(Some(
        values
            .as_slice::<f32>()
            .iter()
            .map(|&value| value > 0.5)
            .collect(),
    ))
}

pub(super) fn prepared_video_modality(
    weights: &Weights,
    modality: &Ltx25ResolvedModality,
    context: &Array,
    fps: f32,
    seed: u64,
    sample_index: u64,
) -> Result<CachedLtx25Modality> {
    let raw = to_dtype(weights.require("video_latents")?, Dtype::Float32)?;
    let shape = parse_shape_metadata(weights, "videoShape", &raw)?;
    let [batch, _channels, frames, height, width]: [i32; 5] = shape.try_into().map_err(|_| {
        mlx_gen::Error::Msg(
            "ltx_2_5 trainer: videoShape must be [B,C,F,H,W] for patchification".into(),
        )
    })?;
    if batch != 1 || frames <= 0 || height <= 0 || width <= 0 {
        return Err("ltx_2_5 trainer: prepared video must have positive shape and B=1".into());
    }
    let clean = crate::conditioning::patchify_grid(&raw)?;
    let geometry = Ltx25TokenGeometry {
        frames: frames as usize,
        height: height as usize,
        width: width as usize,
        spatial_scale: SPATIAL_SCALE as usize,
    };
    let mask = condition_mask_values(weights, mask_condition(modality), geometry.tokens())?;
    let reference_condition = reference_condition(modality);
    let reference_active = reference_condition.is_some_and(|condition| {
        condition_draw(seed, sample_index, 0) < condition.spec.probability
    });
    let (reference, reference_positions, reference_tokens) = if reference_active {
        let condition = reference_condition.expect("checked");
        let key = condition.tensor_key.as_deref().ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: reference condition has no tensorKey".into())
        })?;
        let raw_reference = to_dtype(weights.require(key)?, Dtype::Float32)?;
        let shape = raw_reference.shape();
        if shape.len() != 5 || shape[0] != batch || shape[1] != raw.shape()[1] {
            return Err(format!(
                "ltx_2_5 trainer: video reference `{key}` must be [B,C,F,H,W], got {shape:?}"
            )
            .into());
        }
        let tokens = crate::conditioning::patchify_grid(&raw_reference)?;
        let positions = create_position_grid_with(
            batch as usize,
            shape[2] as usize,
            shape[3] as usize,
            shape[4] as usize,
            condition
                .spec
                .temporal_scale_factor
                .unwrap_or(crate::positions::TEMPORAL_SCALE as usize) as i64,
            condition
                .spec
                .spatial_scale_factor
                .unwrap_or(SPATIAL_SCALE as usize) as i64,
            fps,
            true,
        );
        let count = tokens.shape()[1] as usize;
        (Some(tokens), Some(positions), count)
    } else {
        (None, None, 0)
    };
    let specs: Vec<Ltx25ConditionSpec> = modality
        .conditions
        .iter()
        .map(|condition| condition.spec)
        .collect();
    let token_plan = build_ltx25_token_plan_from_specs(
        modality.is_generated,
        &specs,
        geometry,
        Ltx25ConditionInputs {
            conditioning_mask: mask.as_deref(),
            reference_tokens,
            seed,
            sample_index,
        },
    )?;
    Ok(CachedLtx25Modality {
        clean,
        context: context.clone(),
        positions: create_position_grid_with(
            batch as usize,
            frames as usize,
            height as usize,
            width as usize,
            crate::positions::TEMPORAL_SCALE,
            SPATIAL_SCALE,
            fps,
            true,
        ),
        reference,
        reference_positions,
        token_plan,
    })
}

pub(super) fn prepared_audio_modality(
    weights: &Weights,
    modality: &Ltx25ResolvedModality,
    context: &Array,
    seed: u64,
    sample_index: u64,
) -> Result<CachedLtx25Modality> {
    let raw = to_dtype(weights.require("audio_latents")?, Dtype::Float32)?;
    let shape = parse_shape_metadata(weights, "audioShape", &raw)?;
    let [batch, channels, frames, bins]: [i32; 4] = shape
        .try_into()
        .map_err(|_| mlx_gen::Error::Msg("ltx_2_5 trainer: audioShape must be [B,C,T,F]".into()))?;
    if batch != 1 || channels <= 0 || frames <= 0 || bins <= 0 {
        return Err("ltx_2_5 trainer: prepared audio must have positive shape and B=1".into());
    }
    let clean = raw
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, frames, channels * bins])?;
    let geometry = Ltx25TokenGeometry {
        frames: frames as usize,
        height: 1,
        width: 1,
        spatial_scale: 1,
    };
    let mask = condition_mask_values(weights, mask_condition(modality), geometry.tokens())?;
    let reference_condition = reference_condition(modality);
    let reference_active = reference_condition.is_some_and(|condition| {
        condition_draw(seed, sample_index, 0) < condition.spec.probability
    });
    let (reference, reference_positions, reference_tokens) = if reference_active {
        let condition = reference_condition.expect("checked");
        let key = condition.tensor_key.as_deref().ok_or_else(|| {
            mlx_gen::Error::Msg("ltx_2_5 trainer: reference condition has no tensorKey".into())
        })?;
        let raw_reference = to_dtype(weights.require(key)?, Dtype::Float32)?;
        let shape = raw_reference.shape();
        if shape.len() != 4 || shape[0] != batch || shape[1] != channels || shape[3] != bins {
            return Err(format!(
                "ltx_2_5 trainer: audio reference `{key}` must be [B,C,T,F], got {shape:?}"
            )
            .into());
        }
        let tokens = raw_reference.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            shape[2],
            channels * bins,
        ])?;
        let positions = create_audio_position_grid_with(
            batch as usize,
            shape[2] as usize,
            crate::positions::AUDIO_LATENT_SAMPLE_RATE,
            crate::positions::AUDIO_HOP_LENGTH,
            condition
                .spec
                .temporal_scale_factor
                .unwrap_or(crate::positions::AUDIO_LATENT_DOWNSAMPLE_FACTOR as usize)
                as i64,
            true,
        );
        let count = tokens.shape()[1] as usize;
        (Some(tokens), Some(positions), count)
    } else {
        (None, None, 0)
    };
    let specs: Vec<Ltx25ConditionSpec> = modality
        .conditions
        .iter()
        .map(|condition| condition.spec)
        .collect();
    let token_plan = build_ltx25_token_plan_from_specs(
        modality.is_generated,
        &specs,
        geometry,
        Ltx25ConditionInputs {
            conditioning_mask: mask.as_deref(),
            reference_tokens,
            seed,
            sample_index,
        },
    )?;
    Ok(CachedLtx25Modality {
        clean,
        context: context.clone(),
        positions: create_audio_position_grid(batch as usize, frames as usize),
        reference,
        reference_positions,
        token_plan,
    })
}

pub(super) fn load_prepared_example(
    item: &mlx_gen::TrainingItem,
    plan: &Ltx25TrainingPlan,
    video_context: &Array,
    audio_context: Option<&Array>,
    seed: u64,
    sample_index: u64,
    depth_frames: Option<usize>,
) -> Result<CachedLtx25Example> {
    let path = prepared_bundle_path(item)?;
    let weights = Weights::from_file(path)?;
    validate_prepared_bundle_contract(&weights, path, plan)?;
    let fps = if plan.video.is_some() {
        prepared_fps(&weights)? as f32
    } else {
        crate::positions::DEFAULT_FPS
    };
    let video = plan
        .video
        .as_ref()
        .map(|modality| {
            prepared_video_modality(&weights, modality, video_context, fps, seed, sample_index)
        })
        .transpose()?;
    let audio = plan
        .audio
        .as_ref()
        .map(|modality| {
            let context = audio_context.ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "ltx_2_5 trainer: audio workflow has no audio text context".into(),
                )
            })?;
            prepared_audio_modality(&weights, modality, context, seed, sample_index)
        })
        .transpose()?;
    // sc-24830: the depth frames come from the generated video's own token plan (frames with loss
    // tokens), fixed per example.
    let depth = match (depth_frames, video.as_ref()) {
        (Some(k), Some(video)) if plan.video.as_ref().is_some_and(|v| v.is_generated) => {
            let shape = parse_shape_metadata(
                &weights,
                "videoShape",
                &weights.require("video_latents")?.clone(),
            )?;
            let (c, f, h, w) = (shape[1], shape[2], shape[3], shape[4]);
            let frames = select_depth_frames(
                &video.token_plan.loss_mask[..video.token_plan.target_tokens],
                f as usize,
                (h * w) as usize,
                k,
            );
            if frames.is_empty() {
                return Err(format!(
                    "ltx_2_5 trainer: depth anchoring found no generated video frame in example \
                     {sample_index}"
                )
                .into());
            }
            Some(Ltx25DepthFrames {
                grid: [c, f, h, w],
                frames,
            })
        }
        _ => None,
    };
    let example = CachedLtx25Example {
        video,
        audio,
        depth,
    };
    let mut arrays: Vec<&Array> = Vec::new();
    for modality in [&example.video, &example.audio].into_iter().flatten() {
        arrays.extend([&modality.clean, &modality.context, &modality.positions]);
        if let Some(reference) = &modality.reference {
            arrays.push(reference);
        }
        if let Some(positions) = &modality.reference_positions {
            arrays.push(positions);
        }
    }
    eval(arrays)?;
    Ok(example)
}

pub(super) fn prepare_cached_ltx25(
    cached: &CachedLtx25Example,
    sigma: f32,
    seed: u64,
) -> Result<Ltx25PreparedBatch> {
    let prepare = |modality: &CachedLtx25Modality, salt: u64| -> Result<Ltx25PreparedModality> {
        let noise = random::normal::<f32>(
            modality.clean.shape(),
            None,
            None,
            Some(&random::key(seed.wrapping_add(salt))?),
        )?;
        prepare_ltx25_modality(
            &modality.clean,
            &noise,
            &modality.context,
            &modality.positions,
            modality.reference.as_ref(),
            modality.reference_positions.as_ref(),
            sigma,
            &modality.token_plan,
        )
    };
    Ok(Ltx25PreparedBatch {
        video: cached
            .video
            .as_ref()
            .map(|value| prepare(value, 1))
            .transpose()?,
        audio: cached
            .audio
            .as_ref()
            .map(|value| prepare(value, 2))
            .transpose()?,
    })
}
