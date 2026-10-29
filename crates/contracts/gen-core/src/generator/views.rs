//! Borrowed views over a request's conditioning (keyframes, clips, audio-edit regions, control clips).

use super::*;

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

/// Which edit operation a [`Conditioning::AudioEdit`] requests of a prompted audio editor (sc-12847),
/// mapped by the provider onto ACE-Step 1.5's native audio-to-audio task modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioEditMode {
    /// Regenerate a bounded interior span fresh from the prompt while keeping the rest of the clip
    /// (ACE-Step `repaint`, the edit window silence-substituted so the model fills it anew).
    /// Requires a [`TimeRegion`].
    Inpaint,
    /// Regenerate a bounded span, the model conditioning on the surrounding source for continuity
    /// (ACE-Step `repaint`). Requires a [`TimeRegion`]. Distinct from [`Inpaint`](Self::Inpaint) at
    /// the contract level; the shared ACE-Step machinery differs only in whether the window is
    /// seeded from silence.
    Repaint,
    /// Continue the clip past its end: the appended tail is generated from the prompt while the
    /// original audio is preserved (ACE-Step `repaint` with the generate window at the tail). The
    /// [`TimeRegion`]'s `start_secs` is where generation begins (defaults to the source length) and
    /// `end_secs` names the new total length.
    Extend,
    /// Restyle the whole clip from a new prompt (ACE-Step `cover`). Whole-clip; any [`TimeRegion`]
    /// is ignored.
    Cover,
}

/// A half-open time span `[start_secs, end_secs)` in seconds — the edit region of a
/// [`Conditioning::AudioEdit`] (sc-12847). `end_secs = None` means "to the end of the clip" (and for
/// [`AudioEditMode::Extend`] names the new total length). Both bounds join the finiteness floor.
///
/// Expressed in **seconds** (not latent frames) so the contract stays VAE-stride-agnostic: the
/// provider converts to latent-frame indices via its own `latents_per_second`. This is the audio
/// analogue of the image lane's masked-region conditioning (the pixel [`Conditioning::Mask`] / the
/// video [`Conditioning::ControlClip`]'s `start_frame`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeRegion {
    /// Region start (seconds from the clip start). Must be finite and `>= 0`.
    pub start_secs: f32,
    /// Region end (seconds). `None` ⇒ the end of the clip. When present must be finite and
    /// `> start_secs`.
    pub end_secs: Option<f32>,
}

/// A prompted source-audio edit — a borrowed, normalized view of a [`Conditioning::AudioEdit`].
/// Returned by [`GenerationRequest::audio_edit`] (sc-12847).
#[derive(Clone, Copy, Debug)]
pub struct AudioEditRef<'a> {
    pub audio: &'a AudioTrack,
    pub mode: AudioEditMode,
    pub region: Option<TimeRegion>,
    pub strength: Option<f32>,
}

/// A **multi-region** prompted source-audio edit — a borrowed, normalized view of a
/// [`Conditioning::AudioEditRegions`]. Returned by [`GenerationRequest::audio_edit_regions`]
/// (sc-14549).
///
/// Deliberately a separate accessor from [`AudioEditRef`] rather than a widened one: the two
/// carriers are different [`ConditioningKind`]s, so a provider that serves only the single-region
/// shape keeps refusing multi-region through the shared allowlist without writing any code.
#[derive(Clone, Copy, Debug)]
pub struct AudioEditRegionsRef<'a> {
    pub audio: &'a AudioTrack,
    pub mode: AudioEditMode,
    /// One or more spans to regenerate in a **single** pass. Non-empty; every `end_secs` is
    /// `Some` (see [`Conditioning::AudioEditRegions`] for why). Caller order is not significant.
    pub regions: &'a [TimeRegion],
    pub strength: Option<f32>,
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

/// Exact SCAIL-2 public animation/replacement carrier: one character image, its paired color mask, and one
/// driving clip with a one-to-one frame/mask sequence. The three physical conditioning entries are
/// one conceptual character reference for memory-evidence geometry.
#[derive(Clone, Copy, Debug)]
pub struct Scail2AnimationConditioningRef<'a> {
    pub character: &'a Image,
    pub character_mask: &'a Image,
    pub driving_frames: &'a [Image],
    pub driving_masks: &'a [Image],
}

impl Scail2AnimationConditioningRef<'_> {
    pub const fn reference_count(&self) -> u32 {
        1
    }

    pub fn identity_shape(&self, mode: &str) -> crate::Result<String> {
        let control = self
            .driving_frames
            .first()
            .ok_or_else(|| Error::Unsupported("scail2 carrier has no driving frames".to_owned()))?;
        Ok(format!(
            "{mode}:reference:{}x{}:control:{}x{}x{}",
            self.character.width,
            self.character.height,
            control.width,
            control.height,
            self.driving_frames.len()
        ))
    }
}
