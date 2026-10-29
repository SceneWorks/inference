//! Request validation and resolution: keyframe anchors and images, Ref2VA references and the
//! render geometry.

use super::*;

/// Reject a request this model cannot serve, before any weight is read.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if let Some(veda) = &req.veda {
        veda.validate(50, req.steps.unwrap_or(DEFAULT_STEPS) as usize)
            .map_err(Error::Msg)?;
    }
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
    // The video sigma-shift override (sc-18729), refused **here** rather than 20 minutes downstream
    // where `SigmaSchedule::with_shift` would finally reject it — `validate` is the only gate that
    // runs before the 53 GB text encoder maps. The shared float floor in `caps.validate_request`
    // already rejects NaN/±inf for every knob; this adds the sign, which is specific to a shift
    // (`σ' = s·σ / (1 + (s − 1)·σ)` has a pole and flips sign for `s <= 0`).
    if let Some(shift) = req.scheduler_shift {
        if shift <= 0.0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: scheduler_shift is the video sigma shift and must be positive, got \
                 {shift} (the base checkpoint ships {VIDEO_SIGMA_SHIFT}; the 768p 4-step turbo \
                 variant is trained at 6.0)"
            )));
        }
    }
    // **The `ref2va` reference gate, at the engine boundary** (sc-17149). Every cap — 9 images,
    // 3 clips, 3 audio, 12 combined — plus the audio-is-never-alone rule and the both-tasks-at-once
    // refusal, all before a single weight is read.
    //
    // Enforced here and not only at the API for the reason the whole epic exists: a pipeline handed
    // conditioning it cannot consume does not fail, it renders a plausible clip that silently
    // ignored the thirteenth reference. `Ref2VaReferences` is constructible only through its
    // validating constructor, so the render path cannot reach a reference list that skipped this.
    let references = request_references(req)?;
    MiniMaxH3Task::resolve(!req.keyframes().is_empty(), references.is_some())?;

    // sc-23402 — the reference-image short edge, refused here for the same reason every gate above
    // is: `validate` is the only thing that runs before the 53 GB text encoder maps. The check is
    // unconditional rather than ref2va-only, so a value typed onto a request that carries no image
    // reference is still caught rather than silently inert.
    validate_reference_image_short_edge(MODEL_ID, req)?;

    // sc-19571 — the conditioning-strength refusal runs at the request boundary, not 20 minutes
    // into a render, for the same reason every other gate above does.
    reject_keyframe_strength(&req.keyframes())?;

    // The geometry gate itself — the same call `generate` makes, so `validate` and the render agree
    // by construction rather than by two copies of the lattice arithmetic.
    let geometry = request_geometry(req)?;

    // The clip-length floor (sc-27256, the shared `reference` module's
    // `MIN_REFERENCE_CLIP_FRAMES`): the surviving frame count depends on the request's
    // `num_frames`, so it is checked here against the resolved geometry. Without it `validate`
    // would admit a clip `normalize_reference_clip` then refuses inside `generate`.
    if let Some(references) = &references {
        for r in references.as_slice() {
            if let Ref2VaReference::Video(v) = r {
                crate::reference::normalized_clip_frame_count(
                    v.frames.len(),
                    v.fps,
                    geometry.joint.num_frames.max(0) as usize,
                    crate::denoise::MINIMAX_H3_FPS,
                )?;
            }
        }
    }
    Ok(())
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
/// **Refuse a conditioning strength rather than ignore it** (sc-19571).
///
/// `Conditioning::Keyframe` carries a `strength` that its docs define as a `1 − strength` denoise
/// mask, and SceneWorks exposes it as the first/last-frame conditioning sliders. MiniMax-H3 has no
/// such mask: an anchor is mixed at the checkpoint's own trained-in
/// [`KEYFRAME_NOISE_AUG`](crate::conditioning::KEYFRAME_NOISE_AUG_T) = `0.999` and its rows are told
/// they sit at exactly that `t` (`PackedLayout::row_timesteps`). That number is a property of how
/// the released model was trained, not a knob — `crate::conditioning`'s own docs say conditioning an
/// anchor anywhere else is off-distribution — so there is nothing here a caller-supplied strength
/// could weight without inventing a regime the checkpoint never saw.
///
/// The sc-19571 rule is that a control either works or is refused with a clear error, never
/// silently dropped. This is the refusal, and it names the mechanism so the caller can tell the
/// difference between "unsupported here" and "you typed it wrong".
pub(super) fn reject_keyframe_strength(keyframes: &[KeyframeRef<'_>]) -> Result<()> {
    for kf in keyframes {
        if kf.strength != 1.0 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: keyframe conditioning strength is not supported (got {}) — a \
                 MiniMax-H3 anchor is held at the checkpoint's own trained-in noise augmentation \
                 t = {}, with no denoise mask to weight; use strength 1.0",
                kf.strength,
                crate::conditioning::KEYFRAME_NOISE_AUG_T
            )));
        }
    }
    Ok(())
}

pub(crate) fn keyframe_anchors(
    keyframes: &[KeyframeRef<'_>],
    num_frames: i32,
) -> Result<Vec<crate::dit::positions::KeyframeAnchor>> {
    use crate::dit::positions::KeyframeAnchor;
    reject_keyframe_strength(keyframes)?;
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

/// Map a request's **ordered** conditioning onto `ref2va`'s reference list.
///
/// Returns `None` when the request carries no reference of any modality, which is what makes this
/// the `t2va` / `fl2va` discriminator as well.
///
/// # Order is taken from the request and never re-sorted
///
/// [`GenerationRequest::conditioning`] is a `Vec`, and `ref2va`'s order is semantic — it fixes the
/// `"<Picture i>"` / `"<Audio j>"` / `"<Video k>"` labels **and** advances the shared rotary clock.
/// So this walks the vector once and preserves position. Grouping by modality first (the obvious
/// implementation, and what [`keyframe_images`] legitimately does for `fl2va`'s two fixed slots)
/// would silently rewrite the request.
///
/// # The `ReferenceVideo` decision (sc-17149), and the two carriers it replaced
///
/// A `ref2va` video reference maps to [`Conditioning::ReferenceVideo`] — a carrier added to
/// gen-core for this task, rather than either of the two video variants that already existed. The
/// story asks for this to be deliberate, so:
///
/// * **`VideoSync` is the wrong meaning.** Its contract is video→audio Foley — "the whole-clip
///   visual condition an audio decoder attends to, to synthesize a synchronized soundtrack for a
///   *silent clip*", and its own docs say the frames "are **not** spliced into a video latent".
///   Both halves are false here: a `ref2va` video reference is VAE-encoded into video latent rows,
///   and it conditions a *generated* clip rather than scoring a supplied one. Advertising
///   `VideoSync` would also advertise a Foley capability MiniMax-H3 does not have, and the kind is
///   what routing reads.
/// * **`VideoClip` is the right *mechanism* but the wrong *vocabulary*.** Its latent handling does
///   describe a reference block — VAE-encoded, appended as extra rows, never written by the denoise
///   loop. But its payload is `{frames, frame_idx, strength}`, and a reference can use exactly one
///   of those three fields:
///     * `frame_idx` is a **position in the generated timeline**, which is the defining thing a
///       reference does not have (that is the whole difference from a keyframe);
///     * `strength` is a `1 − strength` denoise mask, and reference rows are fully pinned at the
///       checkpoint's own conditioning timestep ([`crate::denoise::KEYFRAME_NOISE_AUG`] for visual
///       rows, [`crate::denoise::REFERENCE_AUDIO_TIMESTEP`] for audio), never caller-selectable.
///
///   An earlier revision of this function did ride `VideoClip` and rejected both fields unless they
///   were `0` and `1.0`. That shipped a request vocabulary in which two of three fields were traps,
///   and it still could not carry the two things a reference actually needs — see below.
///
/// # What `VideoClip` could not carry, and why that mattered more than the traps
///
/// * **The clip's own frame rate.** [`VideoReference::fps`] is required data, not a hint:
///   MiniMax-H3 resamples every reference onto its own 24 fps by dropping and duplicating whole
///   frames, so a rate that was lost is a reference conditioned on **at the wrong speed with
///   nothing to raise about it**. `VideoClip` has no rate, so the old mapping read the
///   request-level `req.fps` — which [`request_geometry`] independently *rejects* unless it is
///   exactly [`crate::denoise::MINIMAX_H3_FPS`]. Between them, a reference's rate could only ever
///   resolve to 24.0: a 30 fps reference clip was silently treated as 24 fps, which is precisely
///   the failure [`crate::reference`] says the field exists to prevent. `req.fps` is the rate of
///   the *generated output*; a reference's is the rate of *supplied input media*, and for a
///   reference — which does not bind the output geometry at all — those are different quantities.
/// * **The clip's own soundtrack.** [`VideoReference::audio`] is conditioned on as that reference's
///   own, rotary-aligned with its video rows and sharing their origin. `VideoClip` has no audio
///   field, so a soundtrack could only arrive as a separate [`Conditioning::ReferenceAudio`] — legal,
///   but a *standalone* reference with its own rotary slot, which also consumes one of the
///   [`crate::reference::MAX_AUDIO_REFERENCES`] cap slots a video's own soundtrack does not. A
///   different request, silently substituted for the one the caller meant.
///
/// [`Conditioning::ReferenceVideo`] carries `{frames, fps, audio}` — the engine type's own shape —
/// so both reach the packer and neither field-trap exists to reject. A standalone
/// [`Conditioning::ReferenceAudio`] remains available and remains a genuinely different request.
///
/// The SceneWorks-side payload and mode-reachability work rides sc-17160 / sc-17159.
pub(crate) fn request_references(req: &GenerationRequest) -> Result<Option<Ref2VaReferences>> {
    let mut refs: Vec<Ref2VaReference> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => {
                refs.push(Ref2VaReference::Image(image.clone()));
            }
            Conditioning::ReferenceAudio { audio, .. } => {
                refs.push(Ref2VaReference::Audio(crate::reference::AudioReference {
                    audio: audio.clone(),
                }));
            }
            Conditioning::ReferenceVideo { frames, fps, audio } => {
                refs.push(Ref2VaReference::Video(VideoReference {
                    frames: frames.clone(),
                    // The clip's own rate, carried by the variant. Range-checked by
                    // `Ref2VaReferences::new`, which owns every ref2va rule at the engine boundary.
                    fps: f64::from(*fps),
                    audio: audio.clone(),
                }));
            }
            // Keyframes belong to `fl2va` and are resolved by `keyframe_anchors`; everything else
            // is refused upstream by `Capabilities::accepts`, which default-denies any kind this
            // descriptor does not advertise.
            _ => {}
        }
    }
    if refs.is_empty() {
        return Ok(None);
    }
    Ok(Some(Ref2VaReferences::new(refs)?))
}

/// Resolve the request's geometry: the canvas, and the frame count from `frames` (exact) or
/// `duration` (aligned).
pub(crate) fn request_geometry(req: &GenerationRequest) -> Result<RequestGeometry> {
    // The shared resolver the Candle engine and allocation admission (`gen_core::veda::workspace`)
    // use, so the three can never resolve one request to different frame counts (sc-26691).
    let frames = mlx_gen::gen_core::h3_memory::resolve_request_frames(req.frames, req.duration)?;
    let frames = i32::try_from(frames)
        .map_err(|_| Error::Msg(format!("{MODEL_ID}: frames {frames} does not fit an i32")))?;
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
