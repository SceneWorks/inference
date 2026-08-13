//! **`ref2va` multi-modal references** (sc-17149): the ordered, heterogeneous condition list that
//! selects the `transformer_ref` checkpoint.
//!
//! ```text
//! references (ordered) ─┬─ image ─► video VAE ─► condition rows          ─┐
//!                       ├─ video ─► video VAE ─► condition rows           ├─► packed sequence
//!                       │        └► audio VAE ─► reference audio rows    ─┤
//!                       └─ audio ──► audio VAE ─► reference audio rows   ─┘
//! ```
//!
//! # `ref2va` is the task that serves image-edit, and `fl2va` is not
//!
//! The single most consequential fact, established by the sc-17242 spike against the reference
//! implementation and recorded here so it is not re-derived per backend:
//!
//! > **References do not bind the generated geometry.** They are encoded at their *own* resolution
//! > and the target canvas defaults to MiniMax-H3's 16:9.
//!
//! That is the exact opposite of [`crate::keyframe`], where a keyframe is stretched onto the canvas,
//! **binds** the canvas to its own aspect ratio, and becomes literal frame 0. So the same image
//! passed as a keyframe and as a reference produces two different canvases *and* two different
//! tasks, and there is no parameter anywhere that reconciles them. `ref2va` is the subject / edit
//! task; `fl2va` is the "start from this exact frame" task.
//!
//! # Order is semantic twice over — a reference list is not a set
//!
//! The order fixes the `"<Picture i>"` / `"<Audio j>"` / `"<Video k>"` labels of the prompt
//! presentation **and** advances the shared audio/video rotary clock
//! ([`crate::dit::positions::reference_block_position_ids`]). **Reordering the same references is a
//! different request.** Any surface that models the reference list as an unordered set is wrong.
//!
//! # A waveform never reaches the conditioner
//!
//! An audio reference contributes only its `"<Audio j>: "` **text label** to the text encoder; its
//! content goes through the audio VAE alone. That asymmetry is why [`Ref2VaReference::Audio`]
//! carries no vision-token count and why [`ReferencePresentation`] has a variant with no vision
//! block: an implementation that "helpfully" ran a tower over a waveform would be conditioning on
//! something the released model never saw.
//!
//! A **video** reference that carries a soundtrack is labelled `"<Audio j>: "` *before*
//! `"<Video k>: "`, mirroring the order its rows are packed in, and its soundtrack is conditioned on
//! as *its own* rather than as a reference of its own — so it consumes an `<Audio j>` label but not
//! an audio-reference cap slot.
//!
//! # Rates are load-bearing and are not defaulted silently
//!
//! MiniMax-H3 resamples every reference onto its own 24 fps and onto the audio VAE's sample rate.
//! A reference built from frames whose real rate was lost is conditioned on **at the wrong speed,
//! and nothing raises**. So [`VideoReference`] carries its `fps` and [`AudioReference`] its
//! `sample_rate` as required data rather than as an optional hint with a plausible default.
//!
//! The **request surface must carry the rate too**, or the guarantee stops at this type's boundary.
//! It does: `gen_core::Conditioning::ReferenceVideo` has its own `fps`, distinct from the
//! request-level `GenerationRequest::fps` (the *output* rate, which this model pins to 24). An
//! earlier revision routed references through `Conditioning::VideoClip`, which has no rate — every
//! reference therefore arrived declaring exactly 24 fps and a 30 fps clip was conditioned on 25%
//! fast. See `crate::model::request_references` for that decision in full.

use mlx_gen::media::{AudioTrack, Image};
use mlx_gen::{Error, Result};

/// Image references a `ref2va` request may carry.
pub const MAX_IMAGE_REFERENCES: usize = 9;

/// Video-clip references a `ref2va` request may carry.
pub const MAX_VIDEO_REFERENCES: usize = 3;

/// Audio references a `ref2va` request may carry.
///
/// A **video** reference's own soundtrack does *not* consume one of these: it is conditioned on as
/// that reference's own, which is why [`Ref2VaReferences::audio_count`] counts only standalone
/// audio references.
pub const MAX_AUDIO_REFERENCES: usize = 3;

/// References of **any** modality a `ref2va` request may carry in total.
///
/// Deliberately smaller than `9 + 3 + 3 = 15`: the per-modality caps and this one are independent
/// gates, and a request may saturate at most one of them.
pub const MAX_TOTAL_REFERENCES: usize = 12;

/// The short edge an **image** reference is encoded at — its own, not the canvas's.
///
/// Images are encoded at high detail, upscaling included and with **no** area cap, unlike video
/// references and the generated canvas which share the one canvas rule. This is the concrete form
/// of "references do not bind the generated geometry": a 2048-short-edge image reference conditions
/// a 768-short-edge render.
pub const REFERENCE_IMAGE_SHORT_EDGE: i32 = 2048;

/// The rate the **conditioner** reads a video reference at — every `24 / 2 = 12`th frame of the
/// normalized clip.
///
/// This is not the rate the *VAE* sees. A video reference takes two paths, exactly as a keyframe
/// does: the vision tower reads it sparsely at 2 fps for semantic context, and the video VAE encodes
/// it densely at the full 24 fps for the pixel-space anchor. Conflating the two would either flood
/// the conditioner or starve the anchor.
pub const VIDEO_SAMPLE_FPS: f64 = 2.0;

/// Qwen3-VL's temporal patch: the conditioner merges sampled frames in groups of this many, and one
/// merged group is one vision block with one `"<t seconds>"` label.
pub const VISION_TEMPORAL_PATCH: usize = 2;

/// Which frames of a normalized video reference the conditioner reads, and the timestamp label of
/// every vision block they merge into.
///
/// Returns `(frame_indices, block_timestamps)`. `frame_indices` indexes the *normalized* clip (the
/// one already resampled onto MiniMax-H3's 24 fps); `block_timestamps` has one entry per merged
/// group of [`VISION_TEMPORAL_PATCH`] sampled frames, which is one `<|video_pad|>` block.
///
/// # Three details that are each individually easy to get wrong
///
/// * **The cursor is fractional and deduplicated.** `stride = fps / sample_fps` is stepped in
///   floating point and each rounded index is kept only if it advanced, so a clip whose rate is
///   below the sample rate contributes each frame once rather than repeating it.
/// * **Rounding is half-to-even**, Python's `round`, not half-away-from-zero. At the shipped
///   24 / 2 the stride is an exact 12 and the two agree; they diverge the moment a fine-tune moves
///   either rate, which is precisely when nothing would notice.
/// * **The trailing partial group repeats the last timestamp** rather than being dropped or
///   averaged short, and a block's label is the mean of its group's first and last timestamps.
///   That is why the first block of a 2 fps pair renders `"<0.2 seconds>"`: `(0.0 + 0.5) / 2 =
///   0.25`, which `{:.1}` rounds half to even down to `0.2`.
///
/// A clip too short to fill one group is **rejected**, because a merged group repeating a single
/// frame is not something the released model was conditioned on.
pub fn sample_video_condition_frames(
    num_frames: usize,
    fps: f64,
    sample_fps: f64,
    temporal_patch: usize,
) -> Result<(Vec<usize>, Vec<f64>)> {
    if num_frames == 0 {
        return Err(Error::Msg(
            "minimax-h3 ref2va: cannot sample a clip with no frames".into(),
        ));
    }
    if !fps.is_finite() || fps <= 0.0 || !sample_fps.is_finite() || sample_fps <= 0.0 {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: sampling needs positive finite rates, got {fps} and {sample_fps}"
        )));
    }
    if temporal_patch == 0 {
        return Err(Error::Msg(
            "minimax-h3 ref2va: the vision temporal patch must be positive".into(),
        ));
    }

    let stride = fps / sample_fps;
    let mut indices: Vec<usize> = Vec::new();
    let mut cursor = 0.0f64;
    loop {
        let at = crate::keyframe::round_half_to_even(cursor);
        if at >= num_frames as f64 {
            break;
        }
        let at = at as usize;
        if indices.last().is_none_or(|&prev| at > prev) {
            indices.push(at);
        }
        cursor += stride;
    }
    if indices.len() < temporal_patch {
        let minimum =
            crate::keyframe::round_half_to_even((temporal_patch - 1) as f64 * stride) + 1.0;
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: a reference clip is read at {sample_fps} fps and its sampled \
             frames merge in groups of {temporal_patch}, so it must run at least {minimum} frames \
             at {fps} fps, got {num_frames}"
        )));
    }

    // One timestamp per sampled frame, then the trailing partial group is padded by repeating the
    // last one, so every group is full before the means are taken.
    let mut timestamps: Vec<f64> = (0..indices.len()).map(|i| i as f64 / sample_fps).collect();
    let pad = (temporal_patch - timestamps.len() % temporal_patch) % temporal_patch;
    if let Some(&last) = timestamps.last() {
        timestamps.extend(std::iter::repeat_n(last, pad));
    }
    let block_timestamps = timestamps
        .chunks(temporal_patch)
        .map(|c| (c[0] + c[c.len() - 1]) / 2.0)
        .collect();

    Ok((indices, block_timestamps))
}

/// Resize an **image** reference onto its own [`REFERENCE_IMAGE_SHORT_EDGE`], on the stride-32
/// lattice.
///
/// **No area cap and upscaling included** — an image reference is encoded at high detail, unlike a
/// video reference and the generated canvas which share the one canvas rule. This is the concrete
/// mechanism behind "references do not bind the generated geometry": the returned size depends only
/// on the image, never on the request's canvas.
pub fn normalize_reference_image(image: &Image, canvas_multiple: i32) -> Result<Image> {
    if image.width == 0 || image.height == 0 {
        return Err(Error::Msg(
            "minimax-h3 ref2va: cannot normalize a zero-extent reference image".into(),
        ));
    }
    if canvas_multiple <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: canvas multiple must be positive, got {canvas_multiple}"
        )));
    }
    let (w, h) = (f64::from(image.width), f64::from(image.height));
    let scale = f64::from(REFERENCE_IMAGE_SHORT_EDGE) / w.min(h);
    let m = f64::from(canvas_multiple);
    let round_to =
        |v: f64| (crate::keyframe::round_half_to_even(v * scale / m) * m).max(m) as i32 as u32;
    let (tw, th) = (round_to(w), round_to(h));
    if (tw, th) == (image.width, image.height) {
        return Ok(image.clone());
    }
    // `Stretch` — the target size was derived from this image's own aspect ratio, so there is
    // nothing to crop and a cover-crop would discard the edges the reference is meant to supply.
    crate::keyframe::fit_keyframe(image, tw, th, crate::keyframe::KeyframeFit::Stretch)
}

/// Resample a **video** reference's frames onto MiniMax-H3's own 24 fps, truncate to the generated
/// frame count, and put them on the canvas their own aspect ratio resolves to.
///
/// # Resampling is whole frames held into slots, not interpolation
///
/// `slot[i] = floor(i · target/source + 0.5)`, and frame `i` is repeated for `slot[i+1] - slot[i]`
/// output frames — `ffmpeg`'s `fps` filter, which is what the released model was conditioned
/// through. Nothing is blended: a reference at 30 fps has frames **dropped**, one at 12 fps has
/// frames **duplicated**. Interpolating instead would invent motion the model never saw.
///
/// The canvas here is the *shared* canvas rule (short edge and area cap), unlike
/// [`normalize_reference_image`] — a clip is put on a canvas, a still is not.
pub fn normalize_reference_clip(
    frames: &[Image],
    fps: f64,
    num_frames: usize,
    canvas_multiple: i32,
    canvas_short_edge: i32,
    canvas_max_pixels: i64,
    target_fps: f64,
) -> Result<Vec<Image>> {
    if frames.is_empty() {
        return Err(Error::Msg(
            "minimax-h3 ref2va: cannot normalize a clip with no frames".into(),
        ));
    }
    if !fps.is_finite() || fps <= 0.0 {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: a reference clip must carry a positive frame rate, got {fps}"
        )));
    }
    // 1. onto the 24 fps grid.
    let resampled: Vec<&Image> = if (fps - target_fps).abs() < f64::EPSILON {
        frames.iter().collect()
    } else {
        let scale = target_fps / fps;
        let slot = |i: usize| (i as f64 * scale + 0.5).floor() as i64;
        let end = (frames.len() as f64 * scale + 0.5).floor() as i64;
        let mut out: Vec<&Image> = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            let next = if i + 1 < frames.len() {
                slot(i + 1)
            } else {
                end
            };
            let repeat = (next - slot(i)).max(0);
            out.extend(std::iter::repeat_n(f, repeat as usize));
        }
        out
    };
    // 2. truncate to the generated duration.
    let kept: Vec<&Image> = resampled.into_iter().take(num_frames).collect();
    if kept.is_empty() {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: resampling a {}-frame clip from {fps} onto {target_fps} fps left \
             no frames",
            frames.len()
        )));
    }
    // 3. onto the canvas its OWN aspect ratio resolves to.
    let (h, w) = crate::keyframe::resolve_canvas_size(
        f64::from(kept[0].width),
        f64::from(kept[0].height),
        canvas_multiple,
        canvas_short_edge,
        canvas_max_pixels,
    )?;
    kept.into_iter()
        .map(|f| {
            if f.width == w as u32 && f.height == h as u32 {
                Ok(f.clone())
            } else {
                crate::keyframe::fit_keyframe(
                    f,
                    w as u32,
                    h as u32,
                    crate::keyframe::KeyframeFit::Stretch,
                )
            }
        })
        .collect()
}

/// The kind discriminant of a [`Ref2VaReference`] — what the caps count and what the presentation
/// labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// A subject / style / scene reference.
    Image,
    /// A motion and camera reference, conditioned on together with its own soundtrack.
    Video,
    /// A voice or music reference. Never reaches the conditioner.
    Audio,
}

impl ReferenceKind {
    /// The per-modality cap this kind is counted against.
    pub fn cap(self) -> usize {
        match self {
            Self::Image => MAX_IMAGE_REFERENCES,
            Self::Video => MAX_VIDEO_REFERENCES,
            Self::Audio => MAX_AUDIO_REFERENCES,
        }
    }

    /// The noun used in a rejection message.
    pub fn noun(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

/// A motion / camera reference: its frames, the rate they carry, and optionally its soundtrack.
#[derive(Debug, Clone)]
pub struct VideoReference {
    /// The reference frames, in order.
    pub frames: Vec<Image>,
    /// **The rate `frames` actually carry.** Not a hint: MiniMax-H3 resamples onto its own 24 fps
    /// by dropping and duplicating whole frames, so a wrong value here is a reference conditioned
    /// on at the wrong speed with nothing to raise about it.
    pub fps: f64,
    /// This video's own soundtrack, conditioned on as the reference's own rather than as a
    /// reference of its own. `None` conditions on motion alone.
    pub audio: Option<AudioTrack>,
}

/// A voice or music reference. It never reaches the conditioner and is encoded by the audio VAE
/// alone — see the module docs.
#[derive(Debug, Clone)]
pub struct AudioReference {
    /// The reference waveform.
    pub audio: AudioTrack,
}

/// One `ref2va` reference. The list of these is **ordered**, and the order is semantic — see the
/// module docs.
#[derive(Debug, Clone)]
pub enum Ref2VaReference {
    /// A subject, style or scene reference, encoded at [`REFERENCE_IMAGE_SHORT_EDGE`].
    Image(Image),
    /// A motion and camera reference, with its own frame rate and optional soundtrack.
    Video(VideoReference),
    /// A voice or music reference.
    Audio(AudioReference),
}

impl Ref2VaReference {
    /// Which modality this reference is counted as.
    pub fn kind(&self) -> ReferenceKind {
        match self {
            Self::Image(_) => ReferenceKind::Image,
            Self::Video(_) => ReferenceKind::Video,
            Self::Audio(_) => ReferenceKind::Audio,
        }
    }

    /// Whether this reference contributes **audio** rows — a standalone audio reference always, a
    /// video reference only when it carries a soundtrack.
    ///
    /// This is what orders the presentation: a reference with audio emits its `"<Audio j>: "` label
    /// **before** its visual label, mirroring the order its rows are packed in.
    pub fn has_audio(&self) -> bool {
        match self {
            Self::Image(_) => false,
            Self::Video(v) => v.audio.is_some(),
            Self::Audio(_) => true,
        }
    }

    /// This reference's soundtrack, whichever variant carries it.
    pub fn audio(&self) -> Option<&AudioTrack> {
        match self {
            Self::Image(_) => None,
            Self::Video(v) => v.audio.as_ref(),
            Self::Audio(a) => Some(&a.audio),
        }
    }
}

/// A **validated**, ordered `ref2va` reference list.
///
/// The only constructor is [`Ref2VaReferences::new`], so a value of this type has already passed
/// every cap and every structural rule. That is deliberate: it makes "the caps were checked" a
/// property of the type rather than a discipline every call site has to remember, and it is why the
/// engine boundary cannot be bypassed by a caller that assembles rows directly.
#[derive(Debug, Clone)]
pub struct Ref2VaReferences {
    references: Vec<Ref2VaReference>,
}

impl Ref2VaReferences {
    /// Validate an ordered reference list against every `ref2va` rule, at the **engine boundary**.
    ///
    /// This is the gate the story requires to exist here and not only at the API: a request that
    /// reaches the engine over-cap must be refused with a message that names the cap, rather than
    /// rendering a plausible clip that silently ignored the twelfth reference.
    ///
    /// The rules, in the order they are checked:
    ///
    /// 1. **At least one reference.** An empty list is `t2va`, a different task on a different
    ///    checkpoint — routing it here would load the wrong 66 GB partition.
    /// 2. **Each per-modality cap** — [`MAX_IMAGE_REFERENCES`], [`MAX_VIDEO_REFERENCES`],
    ///    [`MAX_AUDIO_REFERENCES`].
    /// 3. **The combined cap** — [`MAX_TOTAL_REFERENCES`]. Checked separately because
    ///    `9 + 3 + 3 = 15 > 12`: a request can satisfy all three per-modality caps and still be
    ///    over budget, which is exactly the 15-file case the epic's R2 calls out.
    /// 4. **Audio references are never alone.** An audio reference has to be paired with at least
    ///    one image or video reference. A waveform never reaches the conditioner, so an
    ///    audio-only request would leave the text encoder with nothing but labels and the visual
    ///    stream wholly unconditioned — the model was not trained on that shape.
    /// 5. **Non-degenerate media** — an empty video reference, an empty waveform, a zero-extent
    ///    image and a non-finite or non-positive frame rate are all refused rather than
    ///    propagated into a shape error 50 layers deep.
    pub fn new(references: Vec<Ref2VaReference>) -> Result<Self> {
        if references.is_empty() {
            return Err(Error::Msg(
                "minimax-h3 ref2va: needs at least one reference; a text-only request is t2va, \
                 which runs a different checkpoint"
                    .into(),
            ));
        }

        for kind in [
            ReferenceKind::Image,
            ReferenceKind::Video,
            ReferenceKind::Audio,
        ] {
            let n = references.iter().filter(|r| r.kind() == kind).count();
            if n > kind.cap() {
                return Err(Error::Msg(format!(
                    "minimax-h3 ref2va: at most {} {} references, got {n}",
                    kind.cap(),
                    kind.noun()
                )));
            }
        }

        if references.len() > MAX_TOTAL_REFERENCES {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: at most {MAX_TOTAL_REFERENCES} references in total, got {}",
                references.len()
            )));
        }

        if references.iter().all(|r| r.kind() == ReferenceKind::Audio) {
            return Err(Error::Msg(
                "minimax-h3 ref2va: an audio reference must be paired with at least one image or \
                 video reference and cannot be used on its own — a waveform never reaches the \
                 conditioner, so an audio-only request leaves the visual stream unconditioned"
                    .into(),
            ));
        }

        for (i, r) in references.iter().enumerate() {
            match r {
                Ref2VaReference::Image(img) => Self::check_image(i, img)?,
                Ref2VaReference::Video(v) => {
                    if v.frames.is_empty() {
                        return Err(Error::Msg(format!(
                            "minimax-h3 ref2va: reference {i} is a video with no frames"
                        )));
                    }
                    if !v.fps.is_finite() || v.fps <= 0.0 {
                        return Err(Error::Msg(format!(
                            "minimax-h3 ref2va: reference {i} declares a frame rate of {}; a \
                             reference is resampled onto 24 fps from the rate it carries, so the \
                             rate must be a positive finite number",
                            v.fps
                        )));
                    }
                    for f in &v.frames {
                        Self::check_image(i, f)?;
                    }
                    if let Some(a) = &v.audio {
                        Self::check_audio(i, a)?;
                    }
                }
                Ref2VaReference::Audio(a) => Self::check_audio(i, &a.audio)?,
            }
        }

        Ok(Self { references })
    }

    fn check_image(index: usize, image: &Image) -> Result<()> {
        if image.width == 0 || image.height == 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: reference {index} has a zero extent ({}x{})",
                image.width, image.height
            )));
        }
        let (w, h) = (f64::from(image.width), f64::from(image.height));
        if w > crate::keyframe::MAX_ASPECT_RATIO * h || h > crate::keyframe::MAX_ASPECT_RATIO * w {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: reference {index} is {}x{}, outside the 1:4 to 4:1 the model \
                 accepts",
                image.width, image.height
            )));
        }
        Ok(())
    }

    fn check_audio(index: usize, audio: &AudioTrack) -> Result<()> {
        if audio.samples.is_empty() {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: reference {index} carries an empty waveform"
            )));
        }
        if audio.channels == 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: reference {index} declares zero audio channels"
            )));
        }
        if audio.sample_rate == 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: reference {index} declares a zero sample rate; a soundtrack is \
                 resampled onto the audio VAE's rate from the rate it carries"
            )));
        }
        Ok(())
    }

    /// The references, in packed order.
    pub fn as_slice(&self) -> &[Ref2VaReference] {
        &self.references
    }

    /// How many references of every kind, for the presentation's per-modality numbering.
    pub fn counts(&self) -> ReferenceCounts {
        ReferenceCounts {
            images: self.kind_count(ReferenceKind::Image),
            videos: self.kind_count(ReferenceKind::Video),
            audios: self.kind_count(ReferenceKind::Audio),
        }
    }

    fn kind_count(&self, kind: ReferenceKind) -> usize {
        self.references.iter().filter(|r| r.kind() == kind).count()
    }

    /// Standalone audio references — **not** the number of `<Audio j>` labels, which also counts
    /// every video reference that carries a soundtrack.
    pub fn audio_count(&self) -> usize {
        self.kind_count(ReferenceKind::Audio)
    }

    /// How many `"<Audio j>: "` labels the presentation emits: every reference that contributes
    /// audio rows, standalone or a video's own soundtrack.
    pub fn audio_label_count(&self) -> usize {
        self.references.iter().filter(|r| r.has_audio()).count()
    }

    /// Total references.
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// Whether the list is empty. Never true for a constructed value — [`Self::new`] refuses an
    /// empty list — but required by the lint that pairs it with [`Self::len`].
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }
}

/// Per-modality reference counts, for the presentation's `1`-based per-modality numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceCounts {
    /// Image references.
    pub images: usize,
    /// Video-clip references.
    pub videos: usize,
    /// Standalone audio references.
    pub audios: usize,
}

/// What one reference contributes to the **prompt presentation** — the labels and the vision block,
/// with the vision-token count the tower produced for it.
///
/// Separate from [`Ref2VaReference`] because the presentation is built from the *encoded* reference
/// (the tower's merged token count is not knowable from the pixels alone), and because an audio
/// reference contributes a label and **no** vision block at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ReferencePresentation {
    /// `"<Picture i>: "` plus one `<|image_pad|>` vision block.
    Image {
        /// Merged vision-token count of this image's block.
        num_tokens: usize,
    },
    /// `"<Audio j>: "` alone — a waveform never reaches the conditioner.
    Audio,
    /// `"<Video k>: "` plus one timestamped `<|video_pad|>` vision block per merged frame pair,
    /// optionally preceded by its own `"<Audio j>: "` label.
    Video {
        /// Merged vision-token count **per block** of this video reference.
        num_tokens: usize,
        /// The timestamp of every vision block, in seconds.
        timestamps: Vec<f64>,
        /// Whether this reference carries a soundtrack, i.e. whether an `"<Audio j>: "` label
        /// precedes its `"<Video k>: "` label.
        has_audio: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    fn track() -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; 128],
            sample_rate: 24_000,
            channels: 1,
            stems: Vec::new(),
        }
    }

    fn img_ref() -> Ref2VaReference {
        Ref2VaReference::Image(image(64, 64))
    }

    /// A solid-colour frame tagged with `marker`. Solid is the point: the canvas step may stretch
    /// these, and a stretch of a constant image is that same constant, so the tag survives resizing
    /// and the test can assert *which* source frames came out rather than only how many.
    fn tagged(marker: u8) -> Image {
        Image {
            width: 64,
            height: 64,
            pixels: vec![marker; 64 * 64 * 3],
        }
    }

    fn markers(frames: &[Image]) -> Vec<u8> {
        frames.iter().map(|f| f.pixels[0]).collect()
    }

    fn normalize_at(fps: f64, frames: &[Image]) -> Vec<Image> {
        normalize_reference_clip(
            frames,
            fps,
            64, // well above every output length here, so truncation never confounds the resample
            crate::pipeline::SPATIAL_STRIDE as i32,
            crate::pipeline::CANVAS_SHORT_EDGE as i32,
            i64::from(crate::pipeline::CANVAS_MAX_PIXELS),
            crate::denoise::MINIMAX_H3_FPS,
        )
        .expect("a well-formed clip normalizes")
    }

    /// A reference clip is resampled **from the rate it declares**, and declaring the wrong rate
    /// silently changes which frames the model sees (sc-17149).
    ///
    /// This branch was unreachable until `Conditioning::ReferenceVideo` gave the request surface a
    /// per-clip rate: the previous carrier had no rate field, so every reference arrived declaring
    /// exactly 24.0 and took the equality early-return. These are the three cases that now differ.
    #[test]
    fn a_clip_is_resampled_from_the_rate_it_declares() {
        let five: Vec<Image> = [10u8, 20, 30, 40, 50].iter().map(|m| tagged(*m)).collect();

        // 24 fps — the identity case, and the ONLY behaviour the old request surface could produce.
        assert_eq!(
            markers(&normalize_at(24.0, &five)),
            vec![10, 20, 30, 40, 50]
        );

        // 30 fps — frames are DROPPED (ffmpeg `fps` slotting: 5 · 24/30 keeps 4). The third frame is
        // the one that goes. Declaring this clip at 24 fps instead would have played it 25% fast
        // with nothing raised — the exact defect the carrier's `fps` field exists to prevent.
        assert_eq!(markers(&normalize_at(30.0, &five)), vec![10, 20, 40, 50]);

        // 12 fps — frames are DUPLICATED, each held for two output slots.
        let three: Vec<Image> = [10u8, 20, 30].iter().map(|m| tagged(*m)).collect();
        assert_eq!(
            markers(&normalize_at(12.0, &three)),
            vec![10, 10, 20, 20, 30, 30]
        );

        // Nothing is blended: every output frame is one of the inputs, verbatim.
        for f in normalize_at(30.0, &five) {
            assert!(
                f.pixels.iter().all(|p| *p == f.pixels[0]),
                "a resampled frame must be a held source frame, never an interpolation"
            );
        }
    }

    fn vid_ref(with_audio: bool) -> Ref2VaReference {
        Ref2VaReference::Video(VideoReference {
            frames: vec![image(64, 64)],
            fps: 24.0,
            audio: with_audio.then(track),
        })
    }

    fn aud_ref() -> Ref2VaReference {
        Ref2VaReference::Audio(AudioReference { audio: track() })
    }

    /// The caps are the documented ones, and the combined cap is **not** their sum — which is the
    /// whole reason it is a separate gate.
    #[test]
    fn the_caps_are_the_documented_ones_and_the_total_is_not_their_sum() {
        assert_eq!(MAX_IMAGE_REFERENCES, 9);
        assert_eq!(MAX_VIDEO_REFERENCES, 3);
        assert_eq!(MAX_AUDIO_REFERENCES, 3);
        assert_eq!(MAX_TOTAL_REFERENCES, 12);
        const {
            assert!(
                MAX_TOTAL_REFERENCES
                    < MAX_IMAGE_REFERENCES + MAX_VIDEO_REFERENCES + MAX_AUDIO_REFERENCES,
                "the combined cap must bind something the per-modality caps do not"
            )
        };
    }

    /// **Each cap boundary, one over and one at.** The at-cap arms matter as much as the over-cap
    /// ones: a gate written with `>=` instead of `>` would reject every legal saturated request and
    /// every rejection test in the suite would still pass.
    #[test]
    fn each_cap_admits_its_boundary_and_refuses_one_more() {
        // 9 images is legal, 10 is not.
        assert!(Ref2VaReferences::new(vec![img_ref(); 9]).is_ok());
        let e = Ref2VaReferences::new(vec![img_ref(); 10])
            .unwrap_err()
            .to_string();
        assert!(e.contains("at most 9 image"), "{e}");

        // 3 clips is legal, 4 is not.
        assert!(Ref2VaReferences::new(vec![vid_ref(false); 3]).is_ok());
        let e = Ref2VaReferences::new(vec![vid_ref(false); 4])
            .unwrap_err()
            .to_string();
        assert!(e.contains("at most 3 video"), "{e}");

        // 3 audio (paired with an image) is legal, 4 is not.
        let mut ok = vec![img_ref()];
        ok.extend(vec![aud_ref(); 3]);
        assert!(Ref2VaReferences::new(ok).is_ok());
        let mut bad = vec![img_ref()];
        bad.extend(vec![aud_ref(); 4]);
        let e = Ref2VaReferences::new(bad).unwrap_err().to_string();
        assert!(e.contains("at most 3 audio"), "{e}");
    }

    /// The combined cap is reachable **without** tripping any per-modality cap — the 15-file case
    /// the epic calls out — and 12 itself is accepted.
    #[test]
    fn the_combined_cap_refuses_a_request_every_per_modality_cap_admits() {
        // 9 + 3 + 3 = 15: every per-modality cap is satisfied, the total is not.
        let mut fifteen = vec![img_ref(); 9];
        fifteen.extend(vec![vid_ref(false); 3]);
        fifteen.extend(vec![aud_ref(); 3]);
        assert_eq!(fifteen.len(), 15);
        let e = Ref2VaReferences::new(fifteen).unwrap_err().to_string();
        assert!(e.contains("at most 12 references in total"), "{e}");

        // 13 trips it too — the story's fourth over-cap case.
        let mut thirteen = vec![img_ref(); 9];
        thirteen.extend(vec![vid_ref(false); 3]);
        thirteen.push(aud_ref());
        assert_eq!(thirteen.len(), 13);
        assert!(Ref2VaReferences::new(thirteen).is_err());

        // …and exactly 12 is accepted, so the gate is not off by one.
        let mut twelve = vec![img_ref(); 9];
        twelve.extend(vec![vid_ref(false); 2]);
        twelve.push(aud_ref());
        assert_eq!(twelve.len(), 12);
        assert!(Ref2VaReferences::new(twelve).is_ok());
    }

    /// An audio reference is never alone, and an empty list is `t2va` rather than a degenerate
    /// `ref2va`.
    #[test]
    fn audio_is_never_the_only_modality_and_empty_is_not_ref2va() {
        let e = Ref2VaReferences::new(Vec::new()).unwrap_err().to_string();
        assert!(e.contains("at least one reference"), "{e}");

        let e = Ref2VaReferences::new(vec![aud_ref()])
            .unwrap_err()
            .to_string();
        assert!(e.contains("cannot be used on its own"), "{e}");
        assert!(Ref2VaReferences::new(vec![aud_ref(); 3]).is_err());

        // Paired with either visual modality it is fine.
        assert!(Ref2VaReferences::new(vec![img_ref(), aud_ref()]).is_ok());
        assert!(Ref2VaReferences::new(vec![vid_ref(false), aud_ref()]).is_ok());
        // A video reference carrying its OWN soundtrack is not an audio reference, so it does not
        // rescue an otherwise audio-only request — but it is also not audio-only itself.
        assert!(Ref2VaReferences::new(vec![vid_ref(true)]).is_ok());
    }

    /// A video reference's own soundtrack consumes an `<Audio j>` **label** but not an
    /// audio-reference **cap slot**. Conflating the two would refuse a legal 3-clip-with-sound
    /// request, or admit six soundtracks.
    #[test]
    fn a_videos_own_soundtrack_is_a_label_not_a_cap_slot() {
        // Three clips that all carry sound, plus three standalone audio references: legal, because
        // only the three standalone ones count against MAX_AUDIO_REFERENCES.
        let mut refs = vec![vid_ref(true); 3];
        refs.extend(vec![aud_ref(); 3]);
        let r = Ref2VaReferences::new(refs).unwrap();
        assert_eq!(
            r.audio_count(),
            3,
            "only standalone audio counts to the cap"
        );
        assert_eq!(
            r.audio_label_count(),
            6,
            "every soundtrack takes an <Audio j> label"
        );
        assert_eq!(r.counts().videos, 3);
        assert_eq!(r.counts().audios, 3);
    }

    /// Degenerate media is refused at the boundary rather than propagated into a shape error deep
    /// inside the DiT.
    #[test]
    fn degenerate_media_is_refused_with_a_message_naming_the_reference() {
        let zero = Ref2VaReference::Image(Image {
            width: 0,
            height: 8,
            pixels: Vec::new(),
        });
        let e = Ref2VaReferences::new(vec![zero]).unwrap_err().to_string();
        assert!(e.contains("zero extent"), "{e}");

        // Outside 1:4 .. 4:1.
        let wide = Ref2VaReference::Image(image(500, 100));
        let e = Ref2VaReferences::new(vec![wide]).unwrap_err().to_string();
        assert!(e.contains("1:4"), "{e}");
        // …and exactly 4:1 is legal, so the bound is inclusive.
        assert!(Ref2VaReferences::new(vec![Ref2VaReference::Image(image(400, 100))]).is_ok());

        let no_frames = Ref2VaReference::Video(VideoReference {
            frames: Vec::new(),
            fps: 24.0,
            audio: None,
        });
        let e = Ref2VaReferences::new(vec![no_frames])
            .unwrap_err()
            .to_string();
        assert!(e.contains("no frames"), "{e}");

        // A frame rate that is zero, negative or NaN is refused — silently defaulting it to 24
        // would resample at the wrong speed with nothing to raise.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let r = Ref2VaReference::Video(VideoReference {
                frames: vec![image(32, 32)],
                fps: bad,
                audio: None,
            });
            assert!(
                Ref2VaReferences::new(vec![r]).is_err(),
                "fps {bad} must be refused"
            );
        }

        let empty_wave = Ref2VaReference::Audio(AudioReference {
            audio: AudioTrack {
                samples: Vec::new(),
                sample_rate: 24_000,
                channels: 1,
                stems: Vec::new(),
            },
        });
        let e = Ref2VaReferences::new(vec![img_ref(), empty_wave])
            .unwrap_err()
            .to_string();
        assert!(e.contains("empty waveform"), "{e}");
    }

    /// `has_audio` is what orders the presentation, so it is pinned per variant rather than left
    /// to the call site.
    #[test]
    fn has_audio_distinguishes_the_three_variants() {
        assert!(!img_ref().has_audio());
        assert!(!vid_ref(false).has_audio());
        assert!(vid_ref(true).has_audio());
        assert!(aud_ref().has_audio());

        assert_eq!(img_ref().kind(), ReferenceKind::Image);
        assert_eq!(vid_ref(true).kind(), ReferenceKind::Video);
        assert_eq!(aud_ref().kind(), ReferenceKind::Audio);
    }
}
