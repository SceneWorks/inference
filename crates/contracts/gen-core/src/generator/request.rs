//! The typed core request (`GenerationRequest` / `CoreRequest`), its defaults, and request-level
//! helpers (seed, reference-image short edge, exhaustive float-finiteness floor).

use super::*;

/// Canonical alias for the family-free core request contract (`REG-4`).
pub type CoreRequest = GenerationRequest;

/// The core generation request union (`REG-4`).
///
/// Core fields are modality-neutral or cross-family. Cross-family optional feature parameters live
/// in [`features`](Self::features) ([`FeatureParams`]), and family-specific knobs live in typed
/// extension blocks under [`extensions`](Self::extensions) ([`FamilyExtensions`]). Boundary
/// selectors (`sampler`, `scheduler`, `guidance_method`, `video_mode`) are typed enums that
/// round-trip losslessly with canonical string names.
///
/// A single `Default`-able struct (no builder): `GenerationRequest { prompt, ..Default::default() }`.
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
    /// Typed sampler/solver selector (`REG-4`).
    pub sampler: Option<SolverId>,
    /// Typed sigma-schedule selector (`REG-4`).
    pub scheduler: Option<ScheduleId>,
    pub scheduler_shift: Option<f32>,
    /// Typed guidance-method selector (`REG-4`): `"cfg"` | `"cfg_rescale"` | `"apg"` | `"cfg_pp"`.
    pub guidance_method: Option<GuidanceMethodId>,
    /// APG projection mix η (`apg` only).
    pub guidance_eta: Option<f32>,
    /// APG momentum (`apg` only).
    pub guidance_momentum: Option<f32>,
    /// APG norm-threshold (`apg` only).
    pub guidance_norm_threshold: Option<f32>,
    /// Classifier-free-guidance interval `(lo, hi)` gated by [`Capabilities::supports_cfg_interval`].
    pub cfg_interval: Option<(f32, f32)>,
    /// Batched prompts rendered together as one batched denoise (`Capabilities::supports_prompt_batch`).
    pub prompt_batch: Vec<String>,
    /// Caption-overflow policy (`Capabilities::supports_caption_overflow_policy`).
    pub caption_overflow: Option<CaptionOverflowPolicy>,

    // --- Conditioning ---
    pub conditioning: Vec<Conditioning>,
    /// Ordinary single-source img2img input image (`§6.13`, `FEAT-1`), distinct from pass-0 edit,
    /// reference/identity, and control conditioning.
    pub source_image: Option<Image>,
    /// img2img strength when a single `Reference` or `source_image` is supplied without its own strength.
    pub strength: Option<f32>,
    /// ControlNet / VACE conditioning strength (`None` ⇒ model default `1.0`).
    pub control_scale: Option<f32>,
    /// Image-guidance (true CFG on reference/image conditioning) for reference-conditioned edit models.
    pub image_guidance: Option<f32>,

    // --- Video (Option; consumed by video models) ---
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub duration: Option<f32>,
    /// Typed video-mode selector (`REG-4`).
    pub video_mode: Option<VideoMode>,

    // --- Prompt enhancement (cross-family: LTX, FLUX.2-dev, FLUX.3 PUP) ---
    pub enhance_prompt: bool,
    pub enhance_max_tokens: Option<u32>,
    pub enhance_temperature: Option<f32>,
    pub prompt_enhancement: PromptEnhancementSink,

    // --- Execution / memory / approximation ---
    pub memory: Option<GenerationMemory>,
    pub approximation: Option<ApproximationRequest>,

    // --- Audio (Option; consumed by `Modality::Audio` models) ---
    pub audio: Option<AudioParams>,

    // --- Features & family extensions (`REG-4`) ---
    /// Typed cross-family feature parameters (`pag`, `phases`, `low_res_start`, `pid`, `extra`).
    pub features: FeatureParams,
    /// Typed per-family extension blocks (zero family-specific top-level fields on `CoreRequest`).
    pub extensions: FamilyExtensions,

    // --- Output surface & control ---
    pub cancel: CancelFlag,
    pub preview: PreviewSink,
    /// Opt in to the HDR lane (`None` is SDR, the default that must never drift).
    pub hdr: Option<HdrRequest>,
    /// RGB (default) vs RGBA with straight alpha (`Capabilities::supports_alpha_output`).
    pub output_channels: OutputChannels,
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
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            guidance_method: None,
            guidance_eta: None,
            guidance_momentum: None,
            guidance_norm_threshold: None,
            cfg_interval: None,
            prompt_batch: Vec::new(),
            caption_overflow: None,
            conditioning: Vec::new(),
            source_image: None,
            strength: None,
            control_scale: None,
            image_guidance: None,
            frames: None,
            fps: None,
            duration: None,
            video_mode: None,
            enhance_prompt: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            prompt_enhancement: PromptEnhancementSink::default(),
            memory: None,
            approximation: None,
            audio: None,
            features: FeatureParams::default(),
            extensions: FamilyExtensions::default(),
            cancel: CancelFlag::default(),
            preview: PreviewSink::default(),
            // SDR. The one default that must never drift — see `GenerationRequest::hdr`.
            hdr: None,
            // RGB. Like `hdr`, a default that must never drift.
            output_channels: OutputChannels::Rgb,
        }
    }
}

impl GenerationRequest {
    /// Attach a typed [`FamilyExtension`] block to this request.
    pub fn with_extension(mut self, extension: FamilyExtension) -> Self {
        self.extensions.set(extension);
        self
    }

    /// Whether the cross-family PiD super-resolving decoder feature is active on this request.
    pub fn use_pid(&self) -> bool {
        self.features.pid.is_some()
    }

    /// Early-stop capture σ for the cross-family PiD decoder feature, if active and specified.
    pub fn pid_capture_sigma(&self) -> Option<f32> {
        self.features.pid.and_then(|pid| pid.capture_sigma)
    }

    /// Number of image-conditioning inputs represented by this request for memory-evidence
    /// geometry. Multi-image carriers contribute their flattened image count; reference (opaque
    /// **or** transparent), control/depth/mask carriers each contribute one. A keyframe is a
    /// distinct image input even though its placement is temporal; clips remain represented by the
    /// frame axis.
    pub fn image_reference_count(&self) -> u32 {
        self.conditioning.iter().fold(0_u32, |count, conditioning| {
            // **This match is deliberately wildcard-free** (sc-24111), for the same reason
            // `first_nonfinite_float` is.
            let increment = match conditioning {
                Conditioning::Reference { .. }
                | Conditioning::ReferenceRgba { .. }
                | Conditioning::Keyframe { .. }
                | Conditioning::Control { .. }
                | Conditioning::Depth { .. }
                | Conditioning::Mask { .. } => 1,
                Conditioning::MultiReference { images } => {
                    u32::try_from(images.len()).unwrap_or(u32::MAX)
                }
                Conditioning::ReduxRefs { refs } => u32::try_from(refs.len()).unwrap_or(u32::MAX),
                Conditioning::ReferenceAudio { .. }
                | Conditioning::ReferenceVideo { .. }
                | Conditioning::AudioEdit { .. }
                | Conditioning::AudioEditRegions { .. }
                | Conditioning::VoiceEmbedding { .. }
                | Conditioning::VideoClip { .. }
                | Conditioning::ControlClip { .. }
                | Conditioning::VideoSync { .. }
                | Conditioning::ConversationHistory { .. } => 0,
            };
            count.saturating_add(increment)
        })
    }

    /// Conceptual reference count used by memory evidence and request scopes.
    pub fn memory_reference_count(&self) -> u32 {
        match self.video_mode.as_deref() {
            Some("animation" | "replacement") if self.scail2_animation_conditioning().is_ok() => 1,
            Some("animation") if self.control_clip().is_some() && self.conditioning.len() > 3 => {
                Scail2CharacterReferences::from_request("scail2", self)
                    .ok()
                    .filter(|characters| {
                        self.conditioning.len() == 2 * characters.reference_count() as usize + 1
                    })
                    .map_or_else(
                        || self.image_reference_count(),
                        |characters| characters.reference_count(),
                    )
            }
            Some("video_bridge") => 0,
            Some("extend_clip")
                if matches!(
                    self.conditioning.as_slice(),
                    [Conditioning::ControlClip { .. }]
                ) =>
            {
                1
            }
            _ => self.image_reference_count(),
        }
    }

    /// The first request-supplied `Option<f32>` knob that is **non-finite** (NaN / ±Inf), returned
    /// as `(field, value)` — or `None` when every present float is finite.
    ///
    /// The exhaustive destructuring below (no `..`) is deliberate and load-bearing: adding a field
    /// to [`GenerationRequest`] fails to compile *here*, forcing the author to classify it.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            // Non-float fields: explicitly named (no `..`) so new fields force classification.
            prompt: _,
            negative_prompt: _,
            width: _,
            height: _,
            count: _,
            seed: _,
            steps: _,
            sampler: _,
            scheduler: _,
            guidance_method: _,
            prompt_batch: _,
            caption_overflow: _,
            cfg_interval,
            conditioning,
            source_image: _,
            frames: _,
            fps: _,
            video_mode: _,
            enhance_prompt: _,
            enhance_max_tokens: _,
            prompt_enhancement: _,
            memory: _,
            approximation: _,
            audio,
            features,
            extensions,
            cancel: _,
            preview: _,
            hdr: _,
            output_channels: _,
            // Every core `Option<f32>` knob the floor owns.
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
            enhance_temperature,
        } = self;

        if let Some(nonfinite) = extensions.first_nonfinite_float() {
            return Some(nonfinite);
        }
        if let Some(nonfinite) = features.first_nonfinite_float() {
            return Some(nonfinite);
        }

        let floats: [(&'static str, Option<f32>); 11] = [
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
            ("enhance_temperature", *enhance_temperature),
        ];
        let interval: [(&'static str, Option<f32>); 2] = [
            ("cfg_interval.lo", cfg_interval.map(|(lo, _)| lo)),
            ("cfg_interval.hi", cfg_interval.map(|(_, hi)| hi)),
        ];
        for (name, v) in floats.into_iter().chain(interval) {
            if let Some(x) = v {
                if !x.is_finite() {
                    return Some((name, x));
                }
            }
        }

        // Audio sub-block floats (sc-12834): destructured exhaustively (no `..`).
        if let Some(AudioParams {
            voice: _,
            language: _,
            sample_rate: _,
            musical_key: _,
            lyrics: _,
            script: _,
            segments: _,
            max_new_tokens_per_segment: _,
            output_limiter: _,
            artifacts: _,
            target_duration,
            bpm,
            repetition_penalty,
            reference_region,
            song,
        }) = audio
        {
            if let Some(SongParams {
                planning: _,
                score: _,
                plan: _,
                cached_latents: _,
                decoder: _,
                plan_only: _,
                cover: _,
                score_sampling,
                semantic_sampling,
            }) = song
            {
                for (phase, sampling) in
                    [("score", score_sampling), ("semantic", semantic_sampling)]
                {
                    if let Some(TokenSampling {
                        temperature,
                        top_p,
                        repetition_penalty,
                        top_k: _,
                        penalty_window: _,
                        min_tokens: _,
                        max_tokens: _,
                    }) = sampling
                    {
                        for (field, v) in [
                            ("temperature", temperature),
                            ("top_p", top_p),
                            ("repetition_penalty", repetition_penalty),
                        ] {
                            if let Some(x) = v {
                                if !x.is_finite() {
                                    let name = match (phase, field) {
                                        ("score", "temperature") => {
                                            "audio.song.score_sampling.temperature"
                                        }
                                        ("score", "top_p") => "audio.song.score_sampling.top_p",
                                        ("score", _) => {
                                            "audio.song.score_sampling.repetition_penalty"
                                        }
                                        (_, "temperature") => {
                                            "audio.song.semantic_sampling.temperature"
                                        }
                                        (_, "top_p") => "audio.song.semantic_sampling.top_p",
                                        _ => "audio.song.semantic_sampling.repetition_penalty",
                                    };
                                    return Some((name, *x as f32));
                                }
                            }
                        }
                    }
                }
            }
            let audio_floats: [(&'static str, Option<f32>); 5] = [
                ("audio.target_duration", *target_duration),
                ("audio.bpm", *bpm),
                ("audio.repetition_penalty", *repetition_penalty),
                (
                    "audio.reference_region.start_secs",
                    reference_region.map(|r| r.start_secs),
                ),
                (
                    "audio.reference_region.end_secs",
                    reference_region.and_then(|r| r.end_secs),
                ),
            ];
            for (name, v) in audio_floats {
                if let Some(x) = v {
                    if !x.is_finite() {
                        return Some((name, x));
                    }
                }
            }
        }

        // Conditioning-carried floats (wildcard-free and guard-free).
        for c in conditioning {
            match c {
                Conditioning::Control { scale, .. } => {
                    if let Some(s) = scale {
                        if !s.is_finite() {
                            return Some(("conditioning.control.scale", *s));
                        }
                    }
                }
                Conditioning::Reference { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference.strength", *s));
                        }
                    }
                }
                Conditioning::ReferenceRgba { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference_rgba.strength", *s));
                        }
                    }
                }
                Conditioning::ReferenceAudio { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference_audio.strength", *s));
                        }
                    }
                }
                Conditioning::ReferenceVideo { fps, .. } => {
                    if !fps.is_finite() {
                        return Some(("conditioning.reference_video.fps", *fps));
                    }
                }
                Conditioning::AudioEdit {
                    strength, region, ..
                } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.audio_edit.strength", *s));
                        }
                    }
                    if let Some(r) = region {
                        if !r.start_secs.is_finite() {
                            return Some((
                                "conditioning.audio_edit.region.start_secs",
                                r.start_secs,
                            ));
                        }
                        if let Some(end) = r.end_secs {
                            if !end.is_finite() {
                                return Some(("conditioning.audio_edit.region.end_secs", end));
                            }
                        }
                    }
                }
                Conditioning::AudioEditRegions {
                    regions, strength, ..
                } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.audio_edit_regions.strength", *s));
                        }
                    }
                    for region in regions {
                        if !region.start_secs.is_finite() {
                            return Some((
                                "conditioning.audio_edit_regions.regions.start_secs",
                                region.start_secs,
                            ));
                        }
                        if let Some(end) = region.end_secs {
                            if !end.is_finite() {
                                return Some((
                                    "conditioning.audio_edit_regions.regions.end_secs",
                                    end,
                                ));
                            }
                        }
                    }
                }
                Conditioning::VoiceEmbedding { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.voice_embedding.strength", *s));
                        }
                    }
                }
                Conditioning::Keyframe { strength, .. } => {
                    if !strength.is_finite() {
                        return Some(("conditioning.keyframe.strength", *strength));
                    }
                }
                Conditioning::VideoClip { strength, .. } => {
                    if !strength.is_finite() {
                        return Some(("conditioning.video_clip.strength", *strength));
                    }
                }
                Conditioning::ControlClip {
                    masking_strength, ..
                } => {
                    if !masking_strength.is_finite() {
                        return Some((
                            "conditioning.control_clip.masking_strength",
                            *masking_strength,
                        ));
                    }
                }
                Conditioning::ReduxRefs { refs } => {
                    for (_, strength) in refs {
                        if !strength.is_finite() {
                            return Some(("conditioning.redux_refs.strength", *strength));
                        }
                    }
                }
                Conditioning::MultiReference { images: _ } => {}
                Conditioning::Depth { image: _ } => {}
                Conditioning::Mask { image: _ } => {}
                Conditioning::VideoSync { frames: _ } => {}
                Conditioning::ConversationHistory { turns: _ } => {}
            }
        }
        None
    }

    /// Reject the request when any `Option<f32>` knob is non-finite.
    pub fn ensure_finite_floats(&self) -> Result<()> {
        if let Some((field, value)) = self.first_nonfinite_float() {
            return Err(Error::Msg(format!("{field} must be finite (got {value})")));
        }
        Ok(())
    }

    /// Return the single effective [`PagConfig`](crate::PagConfig) across [`FeatureParams::pag`]
    /// and [`MossImageOptions::pag`](crate::moss_image::MossImageOptions::pag), rejecting
    /// conflicting specifications when both are present and unequal (SW-7173).
    pub fn effective_pag(&self) -> Result<Option<&crate::PagConfig>> {
        let moss_pag = self
            .extensions
            .moss_image
            .as_ref()
            .and_then(|m| m.options.as_ref())
            .and_then(|m| m.pag.as_ref());
        match (self.features.pag.as_ref(), moss_pag) {
            (None, None) => Ok(None),
            (Some(p), None) | (None, Some(p)) => Ok(Some(p)),
            (Some(a), Some(b)) if a == b => Ok(Some(a)),
            (Some(_), Some(_)) => Err(Error::Msg(
                "conflicting top-level pag and moss_image.pag controls".into(),
            )),
        }
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

    /// The replace_person masked control clip ([`Conditioning::ControlClip`]), if present.
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

    /// Parse the exact ordered SCAIL-2 animation/replacement carrier.
    pub fn scail2_animation_conditioning(
        &self,
    ) -> crate::Result<Scail2AnimationConditioningRef<'_>> {
        let [Conditioning::Reference {
            image: character,
            strength: None,
        }, Conditioning::Mask {
            image: character_mask,
        }, Conditioning::ControlClip {
            frames,
            mask: driving_masks,
            masking_strength,
            start_frame,
            mode,
        }] = self.conditioning.as_slice()
        else {
            return Err(Error::Unsupported(
                "scail2 requires exactly ordered Reference(strength unset), Mask, and ControlClip conditioning"
                    .to_owned(),
            ));
        };
        if frames.is_empty() || driving_masks.len() != frames.len() {
            return Err(Error::Unsupported(format!(
                "scail2 requires one driving mask per non-empty driving frame ({} frames, {} masks)",
                frames.len(),
                driving_masks.len()
            )));
        }
        if *masking_strength != 1.0 || *start_frame != 0 || *mode != ReplacementMode::default() {
            return Err(Error::Unsupported(
                "scail2 ControlClip requires masking_strength=1, start_frame=0, and full-person mode"
                    .to_owned(),
            ));
        }
        if character.width == 0
            || character.height == 0
            || (character_mask.width, character_mask.height) != (character.width, character.height)
        {
            return Err(Error::Unsupported(
                "scail2 requires a non-empty character image and an exact-shape character mask"
                    .to_owned(),
            ));
        }
        let control_shape = (frames[0].width, frames[0].height);
        if control_shape.0 == 0
            || control_shape.1 == 0
            || frames.iter().zip(driving_masks).any(|(frame, mask)| {
                (frame.width, frame.height) != control_shape
                    || (mask.width, mask.height) != control_shape
            })
        {
            return Err(Error::Unsupported(
                "scail2 requires uniform non-empty driving frames with one exact-shape mask each"
                    .to_owned(),
            ));
        }
        Ok(Scail2AnimationConditioningRef {
            character,
            character_mask,
            driving_frames: frames,
            driving_masks,
        })
    }

    /// The prompted audio-edit conditioning ([`Conditioning::AudioEdit`]), if present.
    pub fn audio_edit(&self) -> Option<AudioEditRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::AudioEdit {
                audio,
                mode,
                region,
                strength,
            } => Some(AudioEditRef {
                audio,
                mode: *mode,
                region: *region,
                strength: *strength,
            }),
            _ => None,
        })
    }

    /// The multi-region audio-edit conditioning ([`Conditioning::AudioEditRegions`]), if present.
    pub fn audio_edit_regions(&self) -> Option<AudioEditRegionsRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::AudioEditRegions {
                audio,
                mode,
                regions,
                strength,
            } => Some(AudioEditRegionsRef {
                audio,
                mode: *mode,
                regions,
                strength: *strength,
            }),
            _ => None,
        })
    }
}

/// The last value [`default_seed`] handed out in this process.
pub(super) static LAST_DEFAULT_SEED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Seed when a [`GenerationRequest`] omits one: strictly increasing across calls and threads.
pub fn default_seed() -> u64 {
    use std::sync::atomic::Ordering;
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let mut last = LAST_DEFAULT_SEED.load(Ordering::Relaxed);
    loop {
        let candidate = now.max(last.wrapping_add(1)).max(1);
        match LAST_DEFAULT_SEED.compare_exchange_weak(
            last,
            candidate,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return candidate,
            Err(observed) => last = observed,
        }
    }
}

/// Smallest admitted [`H3Extension::reference_image_short_edge`] (sc-23402).
pub const REFERENCE_IMAGE_SHORT_EDGE_MIN: u32 = 1024;

/// Largest admitted [`H3Extension::reference_image_short_edge`] (sc-23402).
pub const REFERENCE_IMAGE_SHORT_EDGE_MAX: u32 = 2048;

/// Default [`H3Extension::reference_image_short_edge`] when omitted (sc-23402).
pub const REFERENCE_IMAGE_SHORT_EDGE_DEFAULT: u32 = 2048;

/// The effective reference-image short edge for one request.
pub fn effective_reference_image_short_edge(req: &GenerationRequest) -> u32 {
    req.extensions
        .h3
        .as_ref()
        .and_then(|h3| h3.reference_image_short_edge)
        .unwrap_or(REFERENCE_IMAGE_SHORT_EDGE_DEFAULT)
}

/// Refuse an out-of-range [`H3Extension::reference_image_short_edge`] (sc-23402).
pub fn validate_reference_image_short_edge(id: &str, req: &GenerationRequest) -> Result<()> {
    if let Some(edge) = req
        .extensions
        .h3
        .as_ref()
        .and_then(|h3| h3.reference_image_short_edge)
    {
        if !(REFERENCE_IMAGE_SHORT_EDGE_MIN..=REFERENCE_IMAGE_SHORT_EDGE_MAX).contains(&edge) {
            return Err(Error::Msg(format!(
                "{id}: reference_image_short_edge must be in \
                 {REFERENCE_IMAGE_SHORT_EDGE_MIN}..={REFERENCE_IMAGE_SHORT_EDGE_MAX}, got {edge} \
                 (it sizes the REFERENCE, not the render — it is refused rather than clamped \
                 because clamping would change the reference token budget silently)"
            )));
        }
    }
    Ok(())
}
