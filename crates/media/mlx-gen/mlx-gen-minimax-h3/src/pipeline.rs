//! The **`t2va` pipeline** (sc-17147): request geometry, latent packing, the cancellable render
//! core, and the delivery-time audio/video length policy.
//!
//! ```text
//! prompt ─► text encoder (66 GB) ─► context ─┐            drop + drain
//!                                            ▼
//!                       DiT (62 GB → 38.7 GB after the AdaLN evict) ─► denoise_av
//!                                            │            drop + drain
//!                    ┌───────────────────────┴────────────────────┐
//!             video rows ─► unpatchify ─► video VAE ─► frames      audio rows ─► audio VAE ─► track
//! ```
//!
//! # Geometry is a lattice, not a range
//!
//! Two constraints, and neither is a "round it to something sensible" one:
//!
//! * **spatial stride 32** — [`SPATIAL_STRIDE`], the VAE's 16× compression times the DiT's width
//!   patch of 2. A canvas that survives the VAE but is not a whole number of patch columns wide has
//!   no packed representation at all;
//! * **frames `17n + 5`** — [`crate::denoise::LEGAL_FRAME_COUNTS`], 124…345 under the hardcoded
//!   5–15 s duration clamp. This is *not* the `4k + 1` a 4× temporal downsample suggests: the video
//!   VAE decodes in 17-frame chunks producing 5 latents each, so `5k + 2` latents cover `17k + 5`
//!   frames and every other count desynchronizes the picture from the soundtrack
//!   ([`crate::denoise::geometry`]).
//!
//! [`resolve_geometry`] **rejects** anything off the lattice rather than snapping to it. That is a
//! deliberate reversal of the in-tree image-family habit of rounding to a stride: SceneWorks
//! normalizes request dimensions upstream, so an engine gate that silently refits is a gate that
//! never fires — the caller never learns its request was changed, and at video scale the change is a
//! different duration, not a few pixels. [`align_frames_for_duration`] exists for a caller that
//! *wants* the nearest legal count and asks for it by name.
//!
//! # The AV length policy
//!
//! Audio is quantized on a 25 ms latent grid and video on a 41.667 ms frame grid, so the two tracks
//! are the same length at only **5 of the 14** legal durations
//! ([`crate::denoise::JointGeometry::av_drift_seconds`], bounded at ±8.33 ms). See
//! [`fit_audio_to_video`] for the decision and its reasoning.

use mlx_rs::ops::{add, clip, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::{AudioTrack, Image};
use mlx_gen::{CancelFlag, Error, Result};

use crate::audio_config::{AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::config::{LATENT_CHANNELS, VAE_RATIO};
use crate::denoise::{
    adaln_schedule, align_num_frames, denoise_av, video_latent_num_frames, JointGeometry,
    JointSchedule, JointVelocity, PackedLayout, MAX_AV_DRIFT_SECONDS, MINIMAX_H3_FPS,
};

/// What generated width and height must be a multiple of: the VAE's 16× spatial compression times
/// the DiT's `patch_size[2]`.
///
/// `canvas_multiple` in the reference pipeline. **Not** `VAE_RATIO` alone: a 16-aligned canvas can
/// still be an odd number of latent columns, which has no patched representation.
pub const SPATIAL_STRIDE: u32 = VAE_RATIO as u32 * 2;

/// Short edge the released checkpoint generates on by default.
pub const CANVAS_SHORT_EDGE: u32 = 768;

/// Area budget of the released checkpoint's canvas, `768 · 1344` = 1 032 192 px.
///
/// **Enforced by [`resolve_geometry`] since sc-17152.** It was declared and unused, and the gap was
/// not cosmetic: `Capabilities::max_size` bounds each edge at 1344 independently, so `1344 x 1344`
/// passed every check at 1.75x this budget. Canvas area is the *dominant* term in the packed
/// sequence — it sets rows per latent frame, and attention is quadratic in the sequence — so an
/// unbounded area is an unbounded render cost regardless of what the duration bound says.
/// Together with [`MAX_DURATION_SECONDS`] this closes the envelope: the longest sequence any
/// accepted request can build is **104 030 rows** ([`crate::cost`]).
pub const CANVAS_MAX_PIXELS: u32 = 768 * 1344;

/// Shortest clip the released model generates, in seconds — **the lattice floor**, 124 frames.
///
/// See [`MAX_DURATION_SECONDS`] for why both bounds are derived from
/// [`crate::denoise::LEGAL_FRAME_COUNTS`] rather than declared.
pub const MIN_DURATION_SECONDS: f32 =
    crate::denoise::LEGAL_FRAME_COUNTS[0] as f32 / MINIMAX_H3_FPS as f32;

/// Longest clip the released model generates, in seconds — **the lattice ceiling**, 345 frames.
///
/// # 14.375 s, not the advertised 15.0 (sc-17152)
///
/// The reference pipeline's `max_duration` is **15.0 s**, but its `align_num_frames` walks the
/// `17n + 5` lattice and the largest rung inside that declaration is `LEGAL_FRAME_COUNTS[13] = 345`
/// = **14.375 s**. The next rung, 362 frames, is 15.083 s and past it. **No lattice point sits in
/// `[14.375, 15.0]`**, so a caller asking for the documented maximum has nothing exact to land on —
/// and the floor had the mirror-image gap at 5.0 s against 124 frames' 5.1667 s.
///
/// Both bounds are therefore **derived from the lattice** rather than declared alongside it. Three
/// reasons, in the order they decided it:
///
/// 1. **An advertised bound must be reachable.** sc-17147 kept 15.0 and clamped the aligned count to
///    the ceiling, which delivers 14.375 s of picture for a 15 s request without telling the caller
///    — the same quiet refit the `frames` path exists to prevent, applied to three quarters of a
///    second. Refusing 15.0 s is worse than delivering 14.375 s **only if 15.0 s is real**, and it
///    is not: nothing upstream can produce it either.
/// 2. **Admitting 362 frames would be an untested extension, not a fix.** The lattice has no upper
///    term of its own and MM-RoPE is computed rather than tabulated, so 362 is arithmetically fine —
///    but the reference pipeline never emits it, so no upstream render has ever exercised it. Taking
///    the envelope *past* what upstream ships, on our own authority, is a larger commitment than
///    trimming an unreachable 0.625 s.
/// 3. **Derived bounds cannot drift apart again.** The declaration and the lattice disagreeing is
///    the defect; making one the source of the other removes the class.
///
/// The consequence is that [`align_frames_for_duration`] needs no ceiling clamp: every duration in
/// range now aligns *upward* onto a real rung, at both ends, which is the single policy sc-17147
/// asked for.
pub const MAX_DURATION_SECONDS: f32 = crate::denoise::LEGAL_FRAME_COUNTS
    [crate::denoise::LEGAL_FRAME_COUNTS.len() - 1] as f32
    / MINIMAX_H3_FPS as f32;

/// Per-channel mean the video VAE's pixel space is normalized by — ImageNet's.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Per-channel standard deviation the video VAE's pixel space is normalized by — ImageNet's.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The DiT's `(t, h, w)` patch. Read from the loaded config in the pipeline; declared here so the
/// stride arithmetic above has a named source.
pub const PATCH_SIZE: [i32; 3] = [1, 2, 2];

/// One request's resolved geometry: the pixel canvas, the latent canvas and every derived count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestGeometry {
    /// Output width in pixels — a multiple of [`SPATIAL_STRIDE`].
    pub width: u32,
    /// Output height in pixels — a multiple of [`SPATIAL_STRIDE`].
    pub height: u32,
    /// The frame / latent / audio-token counts.
    pub joint: JointGeometry,
}

impl RequestGeometry {
    /// Clip duration implied by the frame count, in seconds.
    pub fn duration_seconds(&self) -> f64 {
        f64::from(self.joint.num_frames) / MINIMAX_H3_FPS
    }

    /// Samples **per channel** the delivered soundtrack carries — see [`fit_audio_to_video`].
    pub fn delivered_audio_samples(&self) -> usize {
        (self.duration_seconds() * f64::from(AUDIO_SAMPLE_RATE)).round() as usize
    }
}

/// Resolve a request's geometry, **rejecting** anything off the lattice.
///
/// Errors — never rounds — on a canvas that is not a multiple of [`SPATIAL_STRIDE`], on a frame
/// count that is not `17n + 5`, and on a duration outside the model's hardcoded 5–15 s.
///
/// # Why this refuses rather than refits
///
/// SceneWorks normalizes request dimensions before a job reaches an engine, so a stride gate that
/// silently snaps is indistinguishable from no gate at all — it can never be observed to fire. And
/// at video scale a refit is not cosmetic: the nearest legal frame count above 125 is **141**, three
/// quarters of a second of extra picture and sound the caller did not ask for and is not told about.
/// A caller that wants the nearest legal count asks for it with [`align_frames_for_duration`].
pub fn resolve_geometry(width: u32, height: u32, num_frames: i32) -> Result<RequestGeometry> {
    if width == 0 || height == 0 {
        return Err(Error::Msg(format!(
            "minimax_h3: width/height must be positive, got {width}x{height}"
        )));
    }
    if !width.is_multiple_of(SPATIAL_STRIDE) || !height.is_multiple_of(SPATIAL_STRIDE) {
        return Err(Error::Msg(format!(
            "minimax_h3: width/height must be a multiple of {SPATIAL_STRIDE} (the VAE's \
             {VAE_RATIO}x spatial compression times the DiT's width patch of {}), got \
             {width}x{height}",
            PATCH_SIZE[2]
        )));
    }
    // The checkpoint's own area budget. Checked as a product because the per-edge cap is not the
    // same constraint: 1344x1344 is inside `Capabilities::max_size` on both edges and 75 % over the
    // area the model generates at. See `CANVAS_MAX_PIXELS`.
    if width.saturating_mul(height) > CANVAS_MAX_PIXELS {
        return Err(Error::Msg(format!(
            "minimax_h3: {width}x{height} is {} px, over the released checkpoint's \
             {CANVAS_MAX_PIXELS} px canvas budget ({CANVAS_SHORT_EDGE} short edge at 16:9). Canvas \
             area sets the packed sequence length and the attention is quadratic in it",
            u64::from(width) * u64::from(height)
        )));
    }
    // **The lattice gate runs first, then the range gate.** The two reject disjoint mistakes and the
    // caller needs to be told which one it made: a count that is not `17n + 5` has no geometry at
    // any duration, while one that is on the lattice but outside 5.1667-14.375 s is a legal shape
    // the model was not trained at. Since sc-17152 derived the range FROM the lattice the two
    // boundaries touch — 123 frames is both off-lattice and (at 5.125 s) below the floor — and
    // whichever check runs first owns the message.
    //
    // `video_latent_num_frames` is the `17n + 5` gate; `JointGeometry::new` below then re-derives
    // and validates the audio-token count and the shared rotary clock.
    video_latent_num_frames(num_frames)?;
    let seconds = f64::from(num_frames) / MINIMAX_H3_FPS;
    if seconds < f64::from(MIN_DURATION_SECONDS) - 1e-9
        || seconds > f64::from(MAX_DURATION_SECONDS) + 1e-9
    {
        return Err(Error::Msg(format!(
            "minimax_h3: {num_frames} frames is {seconds:.4} s, outside the model's \
             {MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS} s range ({}-{} frames at \
             {MINIMAX_H3_FPS} fps)",
            crate::denoise::LEGAL_FRAME_COUNTS[0],
            crate::denoise::LEGAL_FRAME_COUNTS[crate::denoise::LEGAL_FRAME_COUNTS.len() - 1],
        )));
    }
    let latent_height = i32::try_from(height / VAE_RATIO as u32).map_err(|_| {
        Error::Msg(format!(
            "minimax_h3: height {height} does not fit an i32 latent"
        ))
    })?;
    let latent_width = i32::try_from(width / VAE_RATIO as u32).map_err(|_| {
        Error::Msg(format!(
            "minimax_h3: width {width} does not fit an i32 latent"
        ))
    })?;
    Ok(RequestGeometry {
        width,
        height,
        joint: JointGeometry::new(num_frames, latent_height, latent_width)?,
    })
}

/// The **smallest** legal render: 124 frames, i.e. 5.1667 s.
///
/// Named because it is what a gating first-light render should use — the lattice's floor, not the
/// `4k + 1` a temporal-downsample argument suggests.
pub const SMALLEST_LEGAL_FRAMES: i32 = 124;

/// The nearest legal frame count at or above `seconds · 24`, for a caller that opts in to alignment.
///
/// The requested **duration** is range-checked against
/// [`MIN_DURATION_SECONDS`]–[`MAX_DURATION_SECONDS`], which are the lattice's own two ends, and the
/// count is then aligned **upward** onto a rung. Because the range is the lattice's, the walk can
/// never leave it: the largest in-range duration is exactly 345 frames, so no ceiling clamp is
/// needed and there is no duration in range that silently renders shorter than it asked for
/// (sc-17152 — see [`MAX_DURATION_SECONDS`] for why 15.0 s is not in range).
///
/// Alignment is upward at both ends, so a caller that asks for 5.1 s gets 124 frames (5.1667 s) and
/// one that asks for 14.3 s gets 345 (14.375 s) — always at or above the request, never below.
pub fn align_frames_for_duration(seconds: f32) -> Result<i32> {
    if !seconds.is_finite() {
        return Err(Error::Msg(format!(
            "minimax_h3: duration must be finite, got {seconds}"
        )));
    }
    if !(MIN_DURATION_SECONDS - 1e-6..=MAX_DURATION_SECONDS + 1e-6).contains(&seconds) {
        return Err(Error::Msg(format!(
            "minimax_h3: duration {seconds} s is outside the model's \
             {MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS} s range (the {} legal frame counts \
             124-345 at {MINIMAX_H3_FPS} fps; the reference pipeline advertises 5-15 s but its own \
             17n+5 lattice tops out at 14.375 s)",
            crate::denoise::LEGAL_FRAME_COUNTS.len()
        )));
    }
    let requested = (f64::from(seconds) * MINIMAX_H3_FPS).round() as i32;
    let aligned = align_num_frames(requested.max(1));
    debug_assert!(
        crate::denoise::LEGAL_FRAME_COUNTS.contains(&aligned),
        "an in-range duration must align onto a lattice rung, got {aligned}"
    );
    Ok(aligned)
}

/// Pack `[1, C, T, H, W]` video latents into transformer rows, **frame-major then row-major**.
///
/// `patchify_video_latents` in the reference. The permutation is the whole of it: reading the axes
/// in any other order produces a correctly-shaped `[rows, C·prod(patch)]` tensor whose rows do not
/// correspond to the `(t, h, w)` coordinates [`crate::dit::positions::video_position_ids`] assigns
/// them, which is a silently different model.
pub fn patchify_video_latents(latents: &Array, patch: [i32; 3]) -> Result<Array> {
    let s = latents.shape();
    if s.len() != 5 || s[0] != 1 {
        return Err(Error::Msg(format!(
            "minimax_h3 patchify: expected [1, C, T, H, W], got {s:?}"
        )));
    }
    let (c, t, h, w) = (s[1], s[2], s[3], s[4]);
    let [pt, ph, pw] = patch;
    if pt <= 0 || ph <= 0 || pw <= 0 || t % pt != 0 || h % ph != 0 || w % pw != 0 {
        return Err(Error::Msg(format!(
            "minimax_h3 patchify: a {t}x{h}x{w} latent is not divisible by the patch {patch:?}"
        )));
    }
    let blocked = latents.reshape(&[1, c, t / pt, pt, h / ph, ph, w / pw, pw])?;
    // (B, C, T', pt, H', ph, W', pw) -> (B, T', H', W', C, pt, ph, pw)
    let rows = blocked.transpose_axes(&[0, 2, 4, 6, 1, 3, 5, 7])?;
    Ok(rows.reshape(&[1, (t / pt) * (h / ph) * (w / pw), c * pt * ph * pw])?)
}

/// The exact inverse of [`patchify_video_latents`]: `[1, rows, C·prod(patch)]` → `[1, C, T, H, W]`.
pub fn unpatchify_video_rows(
    rows: &Array,
    channels: i32,
    num_latent_frames: i32,
    latent_height: i32,
    latent_width: i32,
    patch: [i32; 3],
) -> Result<Array> {
    let [pt, ph, pw] = patch;
    if pt <= 0 || ph <= 0 || pw <= 0 {
        return Err(Error::Msg(format!(
            "minimax_h3 unpatchify: the patch {patch:?} must be positive"
        )));
    }
    let (tt, hh, ww) = (
        num_latent_frames / pt,
        latent_height / ph,
        latent_width / pw,
    );
    let want = [1, tt * hh * ww, channels * pt * ph * pw];
    if rows.shape() != want {
        return Err(Error::Msg(format!(
            "minimax_h3 unpatchify: expected {want:?} video rows, got {:?}",
            rows.shape()
        )));
    }
    let blocked = rows.reshape(&[1, tt, hh, ww, channels, pt, ph, pw])?;
    // The inverse permutation of the patchify's (0, 2, 4, 6, 1, 3, 5, 7).
    let back = blocked.transpose_axes(&[0, 4, 1, 5, 2, 6, 3, 7])?;
    Ok(back.reshape(&[1, channels, num_latent_frames, latent_height, latent_width])?)
}

/// Unpack the channel-major audio rows into the audio VAE's `[1, channels, latent_channels, T]`.
///
/// The rows arrive as `[1, num_audio_latents · channels, latent_channels]`, every latent of channel
/// 0 then every latent of channel 1 — the packing
/// [`crate::dit::positions::audio_position_ids`] assigns coordinates to. The decoder is **mono**, so
/// the stereo axis is a batch axis, not a feature one.
/// `channels` and `num_audio_latents` enter the row count only as their **product**, so a caller
/// that transposed them would pass every shape check. They are therefore not inferred: the channel
/// count is required to be [`AUDIO_OUTPUT_CHANNELS`], which the model is fixed at (upstream's two
/// audio-position call sites disagree above stereo — see [`crate::dit::positions::audio_position_ids`]),
/// and the latent count comes from [`crate::denoise::JointGeometry`].
pub fn unpack_audio_rows(
    rows: &Array,
    num_audio_latents: i32,
    channels: i32,
    latent_channels: i32,
) -> Result<Array> {
    if channels != i32::from(AUDIO_OUTPUT_CHANNELS) {
        return Err(Error::Msg(format!(
            "minimax_h3 audio unpack: MiniMax-H3 is {AUDIO_OUTPUT_CHANNELS}-channel; a \
             {channels}-channel unpack would be indistinguishable by shape from a transposed \
             (latents, channels) pair"
        )));
    }
    let want = [1, num_audio_latents * channels, latent_channels];
    if rows.shape() != want {
        return Err(Error::Msg(format!(
            "minimax_h3 audio unpack: expected {want:?} audio rows, got {:?}",
            rows.shape()
        )));
    }
    let per_channel = rows.reshape(&[channels, num_audio_latents, latent_channels])?;
    // (C, T, L) -> (C, L, T), then the batch axis the audio VAE takes. The trailing reshape only
    // PREPENDS an axis, so it does not force the transpose to materialize — and a strided array is
    // read wrongly by `Array::as_slice`, which returns the physical buffer. Every MLX op honours the
    // strides, so the tensor is already correct for the decode; the explicit copy is what makes a
    // host readback (and therefore any test of this function) read the logical order.
    let ncl = per_channel.transpose_axes(&[0, 2, 1])?;
    mlx_gen::array::contiguous(&ncl.reshape(&[1, channels, latent_channels, num_audio_latents])?)
}

/// Revert the video VAE's ImageNet pixel normalization and clamp to `[0, 1]`.
///
/// `video · PIXEL_STD + PIXEL_MEAN`, clamped. The VAE decodes into ImageNet-normalized RGB over a
/// `[0, 1]` base range, so a port that skipped this produces a picture with the right structure and
/// the wrong colour balance — plausible enough to ship.
pub fn revert_pixel_normalization(video: &Array) -> Result<Array> {
    let s = video.shape();
    if s.len() != 5 || s[1] != 3 {
        return Err(Error::Msg(format!(
            "minimax_h3 decode: expected [B, 3, T, H, W], got {s:?}"
        )));
    }
    let x = video.as_dtype(Dtype::Float32)?;
    let mean = Array::from_slice(&PIXEL_MEAN, &[1, 3, 1, 1, 1]);
    let std = Array::from_slice(&PIXEL_STD, &[1, 3, 1, 1, 1]);
    Ok(clip(&add(&multiply(&x, &std)?, &mean)?, (0.0, 1.0))?)
}

/// `[1, 3, T, H, W]` in `[0, 1]` → one 8-bit RGB [`Image`] per frame.
pub fn frames_to_images(video: &Array) -> Result<Vec<Image>> {
    let s = video.shape();
    if s.len() != 5 || s[0] != 1 || s[1] != 3 {
        return Err(Error::Msg(format!(
            "minimax_h3 decode: expected [1, 3, T, H, W], got {s:?}"
        )));
    }
    let (t, h, w) = (s[2], s[3], s[4]);
    // NCTHW -> NTHWC, then the frame-major `[T, H·W·3]` flatten. The collapse is what forces the
    // transpose-strided tensor into logical C-order: `as_slice` returns the PHYSICAL buffer, so a
    // raw read of the transposed array would interleave the channels wrongly. Never forms the whole
    // `t·h·w·3` product as a single dimension.
    let bytes = multiply(&video.as_dtype(Dtype::Float32)?, Array::from_f32(255.0))?
        .round(None)?
        .as_dtype(Dtype::Uint8)?
        .transpose_axes(&[0, 2, 3, 4, 1])?
        .reshape(&[t, h * w * 3])?;
    let flat = bytes.as_slice::<u8>();
    let per = (h as usize) * (w as usize) * 3;
    Ok((0..t as usize)
        .map(|i| Image {
            width: w as u32,
            height: h as u32,
            pixels: flat[i * per..(i + 1) * per].to_vec(),
        })
        .collect())
}

/// How the delivered soundtrack is reconciled with the delivered picture.
///
/// # The decision: **the audio is fitted to the picture** — trimmed or silence-padded at the tail.
///
/// The two tracks are the same length at only 5 of the 14 legal durations. The residual is
/// `(round(num_frames / 24 · 40) − num_frames / 24 · 40) / 40` seconds — bounded at
/// [`MAX_AV_DRIFT_SECONDS`] (12.5 ms) and in practice ±8.33 ms or 0, cycling with period 3.
///
/// Four things decide it:
///
/// 1. **The picture is the exact quantity and the soundtrack is the derived one.** `num_frames` is
///    on the model's own `17n + 5` lattice and its duration is exact; `num_audio_latents` is a
///    `round` of it. Correcting the rounded side is correcting the side that carries the error.
/// 2. **The correction is inaudible and cannot be otherwise.** At most 400 samples of a 165 000+
///    sample track — a third of one 25 ms audio token, and roughly half the ~20 ms threshold at
///    which audio/video desync becomes perceptible at all. Trimming removes the tail of the final
///    token; padding appends silence to it.
/// 3. **The delivery contract has no place to express two durations.**
///    `GenerationOutput::Video { frames, fps, audio }` carries one frame count and one track; a
///    consumer computes the clip length as `frames / fps` and then either truncates or pads the
///    audio *itself*, with no knowledge of which end of it is safe to touch. Deciding here, where
///    the geometry is known, replaces a player-dependent behaviour with a stated one.
/// 4. **The alternative — pad or trim the picture — is worse.** Adding or dropping a frame moves
///    `num_frames` off `17n + 5`, which is the one thing
///    [`crate::denoise::JointGeometry::validate`] exists to prevent, and it changes the creative
///    payload rather than an inaudible tail.
///
/// Enforced rather than documented: this function is the only path from decoded PCM to the delivered
/// [`AudioTrack`], it computes the target from `num_frames` and `fps` alone, and it **errors** if the
/// correction it would apply exceeds [`MAX_AV_DRIFT_SECONDS`] — a correction that large means the
/// geometry is wrong, not that the mux policy has work to do.
pub fn fit_audio_to_video(track: AudioTrack, geometry: &RequestGeometry) -> Result<AudioTrack> {
    let channels = usize::from(track.channels);
    if channels == 0 {
        return Err(Error::Msg(
            "minimax_h3 mux: the decoded soundtrack declares zero channels".into(),
        ));
    }
    if !track.samples.len().is_multiple_of(channels) {
        return Err(Error::Msg(format!(
            "minimax_h3 mux: {} interleaved samples is not a whole number of {channels}-channel \
             frames",
            track.samples.len()
        )));
    }
    let have = track.samples.len() / channels;
    let want = geometry.delivered_audio_samples();
    let drift = (have as f64 - want as f64) / f64::from(track.sample_rate.max(1));
    if drift.abs() > MAX_AV_DRIFT_SECONDS + 1e-6 {
        return Err(Error::Msg(format!(
            "minimax_h3 mux: the decoded soundtrack is {have} samples against the {want} that \
             {} frames at {MINIMAX_H3_FPS} fps implies — {drift:+.4} s, beyond the \
             {MAX_AV_DRIFT_SECONDS} s the audio-latent rounding can produce. That is a geometry \
             error, not a mux one",
            geometry.joint.num_frames
        )));
    }

    let mut out = track;
    match have.cmp(&want) {
        std::cmp::Ordering::Greater => out.samples.truncate(want * channels),
        std::cmp::Ordering::Less => out.samples.resize(want * channels, 0.0),
        std::cmp::Ordering::Equal => {}
    }
    for stem in &mut out.stems {
        match stem.samples.len().cmp(&(want * channels)) {
            std::cmp::Ordering::Greater => stem.samples.truncate(want * channels),
            std::cmp::Ordering::Less => stem.samples.resize(want * channels, 0.0),
            std::cmp::Ordering::Equal => {}
        }
    }
    Ok(out)
}

/// The two latent blocks a request starts from, drawn from one seed.
///
/// Video noise is drawn as a `[1, C, T, H, W]` latent and then patchified — **not** drawn directly
/// in row layout — because that is the order the reference's generator produces, and patchify is a
/// pure permutation so the two differ only in which sample lands on which row.
pub fn initial_latents(
    geometry: &RequestGeometry,
    patch: [i32; 3],
    seed: u64,
) -> Result<(Array, Array)> {
    let g = &geometry.joint;
    let key = mlx_rs::random::key(seed)?;
    let video = mlx_rs::random::normal::<f32>(
        &[
            1,
            LATENT_CHANNELS as i32,
            g.num_latent_frames,
            g.latent_height,
            g.latent_width,
        ],
        None,
        None,
        Some(&key),
    )?;
    let video_rows = patchify_video_latents(&video, patch)?;

    let audio_key = mlx_rs::random::key(seed.wrapping_add(1))?;
    let audio_rows = mlx_rs::random::normal::<f32>(
        &[
            1,
            g.num_audio_latents * i32::from(AUDIO_OUTPUT_CHANNELS),
            crate::audio_config::AUDIO_LATENT_CHANNELS as i32,
        ],
        None,
        None,
        Some(&audio_key),
    )?;
    Ok((video_rows, audio_rows))
}

/// The result of [`render_latents`]: the two denoised latent blocks, unpacked into the shapes the
/// two VAEs consume.
#[derive(Debug, Clone)]
pub struct RenderedLatents {
    /// `[1, LATENT_CHANNELS, num_latent_frames, latent_height, latent_width]`.
    pub video: Array,
    /// `[1, AUDIO_OUTPUT_CHANNELS, AUDIO_LATENT_CHANNELS, num_audio_latents]`.
    pub audio: Array,
}

/// **The cancellable render core**: build the schedules, run the joint loop, unpack the results, and
/// force the whole thing to have happened before returning.
///
/// The terminal [`mlx_rs::transforms::eval`] is not an optimization. MLX is lazily evaluated, so
/// without it the unpatchify — and, if the caller chains it, the decode — would be deferred to the
/// caller's first host readback, which sits **outside** every cancel check. `denoise_av` already
/// forces each step for the same reason; this closes the same hole for the tail.
/// `tests/pipeline.rs::the_render_core_keeps_its_compute_inside_the_cancel_checked_region` gates it
/// **directly** — it reads back whether the returned latents are materialized, rather than timing
/// the call against the caller's readback as it once did (sc-19452: that ratio moved with host
/// load, and deleting this very line left it passing at 97.8%).
#[allow(clippy::too_many_arguments)]
pub fn render_latents(
    model: &mut dyn JointVelocity,
    layout: &PackedLayout,
    schedule: &JointSchedule,
    video_rows: &Array,
    audio_rows: &Array,
    patch: [i32; 3],
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<RenderedLatents> {
    let g = *layout.geometry();
    let adaln = adaln_schedule(schedule)?;
    let (video, audio) = denoise_av(
        model, layout, schedule, &adaln, video_rows, audio_rows, cancel, on_step,
    )?;

    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // Only the generated tail is delivered; `fl2va` anchors (sc-17148) sit ahead of it.
    let generated_video = crate::tensor::slice_axis(
        &video,
        1,
        layout.num_condition_video_rows(),
        video.shape()[1],
    )?;
    let generated_audio = crate::tensor::slice_axis(
        &audio,
        1,
        layout.num_condition_audio_rows(),
        audio.shape()[1],
    )?;
    let out = RenderedLatents {
        video: unpatchify_video_rows(
            &generated_video,
            LATENT_CHANNELS as i32,
            g.num_latent_frames,
            g.latent_height,
            g.latent_width,
            patch,
        )?,
        audio: unpack_audio_rows(
            &generated_audio,
            g.num_audio_latents,
            i32::from(AUDIO_OUTPUT_CHANNELS),
            crate::audio_config::AUDIO_LATENT_CHANNELS as i32,
        )?,
    };
    mlx_rs::transforms::eval([&out.video, &out.audio])?;
    Ok(out)
}

/// Build the packed layout for a plain `t2va` request — no keyframe anchors, no reference audio.
pub fn t2va_layout(
    geometry: &RequestGeometry,
    num_text_tokens: i32,
    patch: [i32; 3],
) -> Result<PackedLayout> {
    if num_text_tokens <= 0 {
        return Err(Error::Msg(format!(
            "minimax_h3: the packed sequence needs at least one text row, got {num_text_tokens}"
        )));
    }
    let tags = vec![crate::denoise::TEXT_TAG; num_text_tokens as usize];
    PackedLayout::build(
        geometry.joint,
        patch,
        &tags,
        i32::from(AUDIO_OUTPUT_CHANNELS),
        &[],
    )
}

/// Build the packed layout for an `fl2va` request — keyframe anchors, and **per-row** text tags.
///
/// The two differences from [`t2va_layout`] are not cosmetic:
///
/// * `text_token_tags` is supplied per row rather than filled with [`crate::denoise::TEXT_TAG`],
///   because a keyframe's vision-block rows are tagged **video** and therefore address a different
///   block of the AdaLN modulation table;
/// * `anchors` reserves the conditioning rows that lead the video stream, at their own rotary
///   times and their own row class.
///
/// `t2va` is `fl2va` with empty `anchors` **at the layout level only**. It is not the same path at
/// the block level: the reference selects a different text-encoder step, a different latent prep
/// and a different core denoise step on the presence of a keyframe, and this crate mirrors that
/// split rather than pretending one is a special case of the other.
pub fn fl2va_layout(
    geometry: &RequestGeometry,
    text_token_tags: &[i32],
    anchors: &[crate::dit::positions::KeyframeAnchor],
    patch: [i32; 3],
) -> Result<PackedLayout> {
    if text_token_tags.is_empty() {
        return Err(Error::Msg(
            "minimax_h3: the packed sequence needs at least one text row".into(),
        ));
    }
    PackedLayout::build(
        geometry.joint,
        patch,
        text_token_tags,
        i32::from(AUDIO_OUTPUT_CHANNELS),
        anchors,
    )
}

/// A reference soundtrack as the audio VAE encoder's input: `[channels, 1, samples]`,
/// **de-interleaved**.
///
/// # The model is mono, and stereo is a BATCH axis
///
/// `AudioTrack` carries interleaved PCM (`L R L R …`), and the audio VAE is mono — a stereo
/// reference is encoded as **two batch items** through the same weights, exactly as
/// [`crate::audio_vae::MiniMaxH3AudioVae::decode_stereo`] emits them. So this de-interleaves rather
/// than reshaping: handing the encoder `[1, 1, 2·n]` would encode the two channels as one waveform
/// at twice the rate, which runs, produces latents of a plausible shape, and is wrong.
///
/// The sample **rate** is not resampled here. The reference resamples a soundtrack onto the audio
/// VAE's own rate before encoding, and a track arriving at another rate is rejected rather than
/// silently encoded at the wrong speed.
pub fn audio_track_to_encoder_input(track: &mlx_gen::media::AudioTrack) -> Result<Array> {
    let channels = i32::from(track.channels);
    if channels <= 0 {
        return Err(Error::Msg(
            "minimax_h3: a reference soundtrack declares zero channels".into(),
        ));
    }
    if track.samples.is_empty() {
        return Err(Error::Msg(
            "minimax_h3: a reference soundtrack carries no samples".into(),
        ));
    }
    if !track.samples.len().is_multiple_of(channels as usize) {
        return Err(Error::Msg(format!(
            "minimax_h3: a {}-channel soundtrack cannot hold {} interleaved samples",
            channels,
            track.samples.len()
        )));
    }
    if track.sample_rate != AUDIO_SAMPLE_RATE {
        return Err(Error::Msg(format!(
            "minimax_h3: a reference soundtrack must be resampled onto the audio VAE's \
             {AUDIO_SAMPLE_RATE} Hz before encoding, got {} Hz",
            track.sample_rate
        )));
    }
    let per_channel = track.samples.len() / channels as usize;
    // De-interleave into channel-major order: every sample of channel 0, then channel 1.
    let mut planar = Vec::with_capacity(track.samples.len());
    for c in 0..channels as usize {
        planar.extend(
            track
                .samples
                .iter()
                .skip(c)
                .step_by(channels as usize)
                .copied(),
        );
    }
    Ok(Array::from_slice(
        &planar,
        &[channels, 1, per_channel as i32],
    ))
}

/// Build the packed layout for a **`ref2va`** request — ordered multi-modal reference blocks.
///
/// `references` is in packed order and carries the *encoded* latent geometry of every reference, so
/// the layout and the conditioning rows are described once. See
/// [`PackedLayout::build_ref2va`] for why this is a separate constructor rather than an option on
/// the `fl2va` one, and [`crate::reference`] for why the order is semantic.
pub fn ref2va_layout(
    geometry: &RequestGeometry,
    text_token_tags: &[i32],
    references: &[crate::dit::positions::ReferenceLatentGeometry],
    patch: [i32; 3],
) -> Result<PackedLayout> {
    if text_token_tags.is_empty() {
        return Err(Error::Msg(
            "minimax_h3: the packed sequence needs at least one text row".into(),
        ));
    }
    PackedLayout::build_ref2va(
        geometry.joint,
        patch,
        text_token_tags,
        i32::from(AUDIO_OUTPUT_CHANNELS),
        references,
    )
}

/// Prepend a `ref2va` request's **reference soundtrack** rows to its freshly-drawn audio rows.
///
/// The audio sibling of [`prepend_condition_rows`], and checked against the layout for the same
/// reason: a soundtrack block of the wrong height concatenates cleanly and produces a runnable,
/// silently misaligned sequence. `t2va` / `fl2va` reserve zero such rows and pass `None`.
///
/// The reference rows ride at a clean [`crate::denoise::REFERENCE_AUDIO_TIMESTEP`] (`1.0`) rather
/// than the `0.999` visual anchors sit at — the audio conditioning is genuinely clean, and that
/// difference is a real class in [`crate::denoise::RowClass`], not a rounding of the same idea.
pub fn prepend_condition_audio_rows(
    layout: &PackedLayout,
    condition_rows: Option<&Array>,
    audio_rows: &Array,
) -> Result<Array> {
    prepend_rows(
        layout.num_condition_audio_rows(),
        condition_rows,
        audio_rows,
        "reference soundtrack",
    )
}

/// The shape-and-count contract both prepends share.
fn prepend_rows(
    expected: i32,
    condition_rows: Option<&Array>,
    generated: &Array,
    what: &str,
) -> Result<Array> {
    let Some(rows) = condition_rows else {
        if expected != 0 {
            return Err(Error::Msg(format!(
                "minimax_h3: the layout reserves {expected} {what} rows but none were supplied"
            )));
        }
        return Ok(generated.clone());
    };
    let (rs, vs) = (rows.shape(), generated.shape());
    if rs.len() != 3 || rs[0] != 1 || vs.len() != 3 || vs[0] != 1 {
        return Err(Error::Msg(format!(
            "minimax_h3: expected [1, rows, features] blocks, got {rs:?} and {vs:?}"
        )));
    }
    let got = rs[1];
    if got != expected {
        return Err(Error::Msg(format!(
            "minimax_h3: {got} {what} rows against a layout reserving {expected}"
        )));
    }
    if rs[2] != vs[2] {
        return Err(Error::Msg(format!(
            "minimax_h3: {what} rows are {} wide, generated rows {}",
            rs[2], vs[2]
        )));
    }
    Ok(mlx_rs::ops::concatenate_axis(
        &[rows.clone(), generated.clone()],
        1,
    )?)
}

/// Prepend a request's conditioning rows to its freshly-drawn video rows.
///
/// `MiniMaxH3FL2VAPrepareLatentsStep` in one line: the anchors **lead** the video row stream, and
/// the scheduler then writes only the tail ([`PackedLayout::generated_video_rows`]), which is how
/// they ride through every step untouched.
///
/// Checked against the layout rather than trusted, because a conditioning block of the wrong
/// height still concatenates cleanly and produces a runnable — and silently misaligned — sequence.
pub fn prepend_condition_rows(
    layout: &PackedLayout,
    condition_rows: Option<&Array>,
    video_rows: &Array,
) -> Result<Array> {
    prepend_rows(
        layout.num_condition_video_rows(),
        condition_rows,
        video_rows,
        "conditioning",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise::LEGAL_FRAME_COUNTS;

    /// **The stride is 32, not 16.** A 16-aligned canvas that is an odd number of latent columns has
    /// no patched representation, and the crate's `SIZE_MULTIPLE` advertises the same number.
    #[test]
    fn the_spatial_stride_is_the_vae_ratio_times_the_patch() {
        assert_eq!(SPATIAL_STRIDE, 32);
        assert_eq!(SPATIAL_STRIDE, VAE_RATIO as u32 * PATCH_SIZE[2] as u32);
        assert_eq!(crate::SIZE_MULTIPLE, SPATIAL_STRIDE);
        // 16-aligned but not 32-aligned: rejected, not rounded.
        let e = resolve_geometry(592, 320, 124).unwrap_err().to_string();
        assert!(e.contains("multiple of 32"), "{e}");
        resolve_geometry(576, 320, 124).unwrap();
    }

    /// **Off-lattice geometry is rejected, not refit.** Every one of these is a positive, plausible
    /// request that a rounding gate would happily service.
    #[test]
    fn off_lattice_geometry_is_rejected_rather_than_refit() {
        // Frame counts: 4k+1 is the plausible wrong lattice, and 125 is one past the floor.
        for frames in [121, 123, 125, 129, 140, 200, 241] {
            let e = resolve_geometry(576, 320, frames).unwrap_err().to_string();
            assert!(
                e.contains("17n + 5") || e.contains("outside the model"),
                "{frames} frames: {e}"
            );
        }
        // Durations outside 5-15 s, even when they ARE 17n+5.
        for frames in [5, 39, 107, 362, 379] {
            assert!(resolve_geometry(576, 320, frames).is_err(), "{frames}");
        }
        // ...and every legal count resolves.
        for &frames in &LEGAL_FRAME_COUNTS {
            let g = resolve_geometry(576, 320, frames).unwrap();
            assert_eq!(g.joint.num_frames, frames);
            assert_eq!((g.width, g.height), (576, 320));
            assert_eq!((g.joint.latent_height, g.joint.latent_width), (20, 36));
        }
        assert!(resolve_geometry(0, 320, 124).is_err());
        assert!(resolve_geometry(576, 0, 124).is_err());
    }

    /// **The canvas area budget is enforced, not just declared** (sc-17152).
    ///
    /// `Capabilities::max_size` caps each edge at 1344 *independently*, so the square 1344×1344 was
    /// accepted at 1.75× the area the checkpoint generates at — and area is the dominant term in the
    /// packed sequence length, hence in an attention cost that is quadratic in it.
    #[test]
    fn the_canvas_area_budget_is_enforced() {
        assert_eq!(CANVAS_MAX_PIXELS, 1_032_192);
        // The declared canvas itself, in both orientations, is exactly at the budget and passes.
        resolve_geometry(1344, 768, 124).unwrap();
        resolve_geometry(768, 1344, 124).unwrap();

        // Every edge here is inside `Capabilities::max_size` (1344) and 32-aligned, so nothing else
        // in the stack refuses them.
        for (w, h) in [(1344u32, 1344u32), (1344, 1024), (1088, 1088)] {
            assert!(w <= 1344 && h <= 1344 && w % 32 == 0 && h % 32 == 0);
            let e = resolve_geometry(w, h, 124).unwrap_err().to_string();
            assert!(e.contains("canvas budget"), "{w}x{h}: {e}");
        }

        // The payoff: with the area budget and the lattice ceiling both enforced, the longest packed
        // sequence any accepted request can build is bounded — 104 030 rows at a 64-token prompt.
        // Without the area gate, 1344x1344 x 345 frames would build 181 142.
        let top = resolve_geometry(1344, 768, 345).unwrap();
        let seq = crate::cost::packed_seq_len(&top.joint, PATCH_SIZE, 64, 0, 2).unwrap();
        assert_eq!(seq, 104_030);
        let unbounded = crate::denoise::JointGeometry::new(345, 84, 84).unwrap();
        assert_eq!(
            crate::cost::packed_seq_len(&unbounded, PATCH_SIZE, 64, 0, 2).unwrap(),
            181_142,
            "the sequence the un-enforced square canvas would have built"
        );
    }

    /// The smallest legal render is 124 frames / 5.1667 s — the value a gating first-light run uses.
    #[test]
    fn the_smallest_legal_render_is_one_hundred_and_twenty_four_frames() {
        assert_eq!(SMALLEST_LEGAL_FRAMES, LEGAL_FRAME_COUNTS[0]);
        let g = resolve_geometry(576, 320, SMALLEST_LEGAL_FRAMES).unwrap();
        assert!((g.duration_seconds() - 5.1666).abs() < 1e-3);
        assert_eq!(g.joint.num_latent_frames, 37);
        assert_eq!(g.joint.num_audio_latents, 207);
        // Nothing shorter is legal.
        assert!(resolve_geometry(576, 320, SMALLEST_LEGAL_FRAMES - 17).is_err());
    }

    /// **The advertised duration range IS the lattice** (sc-17152), at both ends.
    ///
    /// The reference pipeline declares 5–15 s while its own `17n + 5` lattice runs 5.1667–14.375 s,
    /// so both declared endpoints are unreachable. Deriving the bounds from
    /// [`LEGAL_FRAME_COUNTS`] makes every in-range duration land on a rung at or above itself, and
    /// makes both out-of-range endpoints a refusal rather than a silent 0.625 s short delivery.
    #[test]
    fn the_advertised_duration_range_is_the_lattice_at_both_ends() {
        // Derived, not declared — the two must agree by construction.
        assert!((f64::from(MIN_DURATION_SECONDS) - 124.0 / 24.0).abs() < 1e-6);
        assert_eq!(MAX_DURATION_SECONDS, 14.375);
        assert!(
            (f64::from(LEGAL_FRAME_COUNTS[13]) / MINIMAX_H3_FPS - 14.375).abs() < 1e-9,
            "the lattice ceiling really is short of the reference's declared 15 s"
        );

        // Both declared-but-unreachable endpoints are now refused rather than quietly refit.
        for (label, seconds) in [
            ("the reference's declared 5.0 s floor", 5.0f32),
            ("the reference's declared 15.0 s ceiling", 15.0),
        ] {
            let e = align_frames_for_duration(seconds)
                .expect_err(&format!(
                    "{label} is not on the lattice and must be refused"
                ))
                .to_string();
            assert!(e.contains("outside the model"), "{label}: {e}");
        }

        // Every in-range duration aligns UPWARD onto a real rung — never below the request, and
        // never off the lattice.
        for (seconds, frames) in [
            (5.1667f32, 124),
            (5.3, 141),
            (8.0, 192),
            (12.0, 294),
            (14.3, 345),
            (14.375, 345),
        ] {
            let got = align_frames_for_duration(seconds).unwrap();
            assert_eq!(got, frames, "{seconds} s");
            assert!(LEGAL_FRAME_COUNTS.contains(&got));
            assert!(
                f64::from(got) / MINIMAX_H3_FPS >= f64::from(seconds) - 1e-4,
                "{seconds} s aligned DOWN to {got} frames"
            );
        }
        // ...and the aligned count always resolves, i.e. alignment cannot produce a geometry the
        // gate then rejects.
        for &frames in &LEGAL_FRAME_COUNTS {
            let seconds = frames as f32 / MINIMAX_H3_FPS as f32;
            let aligned = align_frames_for_duration(seconds).unwrap();
            assert_eq!(aligned, frames);
            resolve_geometry(576, 320, aligned).unwrap();
        }

        assert!(align_frames_for_duration(0.5).is_err(), "below the floor");
        assert!(align_frames_for_duration(16.0).is_err(), "past the ceiling");
        assert!(align_frames_for_duration(f32::NAN).is_err());
    }

    /// Patchify and unpatchify are exact inverses, and the row order really is frame-major.
    #[test]
    fn patchify_round_trips_and_is_frame_major() {
        let (c, t, h, w) = (2, 3, 4, 6);
        let n = c * t * h * w;
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let latent = Array::from_slice(&v, &[1, c, t, h, w]);
        let rows = patchify_video_latents(&latent, [1, 2, 2]).unwrap();
        assert_eq!(rows.shape(), &[1, t * (h / 2) * (w / 2), c * 4]);
        let back = unpatchify_video_rows(&rows, c, t, h, w, [1, 2, 2]).unwrap();
        assert_eq!(back.as_slice::<f32>(), latent.as_slice::<f32>());

        // Frame-major: the first `(h/2)·(w/2)` rows all come from frame 0, i.e. every value in them
        // is below the per-frame stride of one channel-plane.
        let flat = rows.as_slice::<f32>();
        let rows_per_frame = ((h / 2) * (w / 2)) as usize;
        let per_channel_frame = (h * w) as usize;
        for r in 0..rows_per_frame {
            for k in 0..(c * 4) as usize {
                let value = flat[r * (c * 4) as usize + k] as usize;
                // channel 0's frame 0 is [0, h·w); channel 1's frame 0 is [c0_size, ...).
                let within_frame = value % (t as usize * per_channel_frame) < per_channel_frame;
                assert!(
                    within_frame,
                    "row {r} slot {k} = {value} is not from frame 0"
                );
            }
        }
        assert!(patchify_video_latents(&latent, [1, 5, 5]).is_err());
        assert!(unpatchify_video_rows(&rows, c, t, h, w, [1, 1, 1]).is_err());
    }

    /// The audio unpack is channel-major, and transposing the two channel meanings is rejected.
    #[test]
    fn audio_rows_unpack_channel_major() {
        let (latents, channels, chan_dim) = (3, 2, 4);
        let v: Vec<f32> = (0..latents * channels * chan_dim)
            .map(|i| i as f32)
            .collect();
        let rows = Array::from_slice(&v, &[1, latents * channels, chan_dim]);
        let out = unpack_audio_rows(&rows, latents, channels, chan_dim).unwrap();
        assert_eq!(out.shape(), &[1, channels, chan_dim, latents]);
        let flat = out.as_slice::<f32>();
        // Channel 0's first latent is row 0; channel 1's first latent is row `latents`.
        assert_eq!(flat[0], 0.0, "channel 0, feature 0, latent 0");
        assert_eq!(flat[1], chan_dim as f32, "channel 0, feature 0, latent 1");
        let c1 = (chan_dim * latents) as usize;
        assert_eq!(
            flat[c1],
            (latents * chan_dim) as f32,
            "channel 1 starts at row `num_audio_latents`, not interleaved"
        );
        // The transposed pair is shape-identical (the product is symmetric), so it is caught by
        // pinning the channel count to the model's stereo constant rather than by a shape check.
        let e = unpack_audio_rows(&rows, channels, latents, chan_dim)
            .unwrap_err()
            .to_string();
        assert!(e.contains("indistinguishable by shape"), "{e}");
        assert!(unpack_audio_rows(&rows, latents + 1, channels, chan_dim).is_err());
    }

    /// **The AV mux policy.** The delivered track is exactly `round(num_frames / fps · 32000)`
    /// samples per channel at every one of the 14 legal durations; the correction is bounded, and it
    /// is non-zero at exactly the 9 durations the geometry says it should be.
    #[test]
    fn the_delivered_soundtrack_is_fitted_to_the_picture() {
        let mut corrected = 0;
        for &frames in &LEGAL_FRAME_COUNTS {
            let g = resolve_geometry(576, 320, frames).unwrap();
            // What the audio VAE actually emits: `num_audio_latents · hop` samples per channel.
            let hop = 800;
            let decoded = (g.joint.num_audio_latents * hop) as usize;
            let track = AudioTrack {
                samples: vec![0.5; decoded * 2],
                sample_rate: AUDIO_SAMPLE_RATE,
                channels: 2,
                stems: Vec::new(),
            };
            let fitted = fit_audio_to_video(track, &g).unwrap();
            let want = g.delivered_audio_samples();
            assert_eq!(
                fitted.samples.len(),
                want * 2,
                "{frames} frames: the track must be exactly the picture's length"
            );
            let delta = (decoded as f64 - want as f64) / f64::from(AUDIO_SAMPLE_RATE);
            assert!(
                delta.abs() <= MAX_AV_DRIFT_SECONDS + 1e-9,
                "{frames} frames: correction {delta:.5} s exceeds the bound"
            );
            if decoded != want {
                corrected += 1;
            }
            // The delivered duration matches frames/fps to well inside one 25 ms audio token.
            let delivered = want as f64 / f64::from(AUDIO_SAMPLE_RATE);
            assert!(
                (delivered - g.duration_seconds()).abs() < 0.001,
                "{frames} frames: {delivered} s of sound against {} s of picture",
                g.duration_seconds()
            );
        }
        assert_eq!(
            corrected, 9,
            "9 of the 14 legal durations need a correction; 141/192/243/294/345 are exact"
        );
    }

    /// A correction larger than the audio-latent rounding can produce is a **geometry** error and is
    /// refused rather than quietly applied.
    #[test]
    fn a_mux_correction_beyond_the_rounding_bound_is_refused() {
        let g = resolve_geometry(576, 320, 124).unwrap();
        let track = AudioTrack {
            samples: vec![0.1; 1000],
            sample_rate: AUDIO_SAMPLE_RATE,
            channels: 2,
            stems: Vec::new(),
        };
        let e = fit_audio_to_video(track, &g).unwrap_err().to_string();
        assert!(e.contains("geometry error"), "{e}");

        // ...and a malformed track is rejected rather than silently truncated.
        let odd = AudioTrack {
            samples: vec![0.1; 3],
            sample_rate: AUDIO_SAMPLE_RATE,
            channels: 2,
            stems: Vec::new(),
        };
        assert!(fit_audio_to_video(odd, &g).is_err());
    }

    /// The pixel de-normalization is ImageNet's and clamps, and skipping it is observable.
    #[test]
    fn the_pixel_denormalization_is_imagenet_and_clamps() {
        // `[1, 3, T=2, 1, 1]` is channel-major: each channel owns two consecutive slots. Every
        // channel gets a 0 (which must map to its own mean) and a 4 (which must clamp).
        let x = Array::from_slice(&[0.0f32, 4.0, 0.0, 4.0, 0.0, 4.0], &[1, 3, 2, 1, 1]);
        let out = revert_pixel_normalization(&x).unwrap();
        let v = out.as_slice::<f32>();
        for c in 0..3 {
            assert!(
                (v[c * 2] - PIXEL_MEAN[c]).abs() < 1e-6,
                "channel {c} at 0 must be its own mean: {v:?}"
            );
            // mean + 4·std is 1.40/1.35/1.31 for the three channels — all clamped.
            assert!((v[c * 2 + 1] - 1.0).abs() < 1e-6, "must clamp: {v:?}");
        }
        assert!(
            (PIXEL_MEAN[0] - PIXEL_MEAN[2]).abs() > 0.05,
            "the three channels must differ, or a skipped de-normalization is invisible"
        );
        assert!(revert_pixel_normalization(&Array::from_slice(&[1.0f32], &[1])).is_err());
    }

    /// Frames come out row-major RGB at the right size, and the channel axis is not transposed.
    #[test]
    fn frames_decode_to_row_major_rgb() {
        // 2 frames, 1x2, with channel 0 = 1.0 and the others 0 -> pure red.
        let mut v = vec![0.0f32; 3 * 2 * 2];
        for slot in v.iter_mut().take(4) {
            *slot = 1.0; // channel 0 over both frames
        }
        let video = Array::from_slice(&v, &[1, 3, 2, 1, 2]);
        let images = frames_to_images(&video).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!((images[0].width, images[0].height), (2, 1));
        assert_eq!(images[0].pixels, vec![255, 0, 0, 255, 0, 0]);
        assert!(frames_to_images(&Array::from_slice(&[1.0f32], &[1])).is_err());
    }

    /// The `t2va` layout carries no conditioning rows at all, and its row count is the geometry's.
    #[test]
    fn the_t2va_layout_has_no_conditioning_rows() {
        let g = resolve_geometry(576, 320, 124).unwrap();
        let l = t2va_layout(&g, 8, PATCH_SIZE).unwrap();
        assert_eq!(l.num_condition_video_rows(), 0);
        assert_eq!(l.num_condition_audio_rows(), 0);
        assert_eq!(l.rows_per_frame(), (20 / 2) * (36 / 2));
        assert_eq!(
            l.seq_len(),
            8 + 207 * 2 + 37 * l.rows_per_frame(),
            "text + stereo audio + video"
        );
        assert!(t2va_layout(&g, 0, PATCH_SIZE).is_err());
    }

    /// The initial latents are the two shapes the packed sequence expects, and the seed drives them.
    #[test]
    fn initial_latents_are_seeded_and_correctly_shaped() {
        let g = resolve_geometry(576, 320, 124).unwrap();
        let (v, a) = initial_latents(&g, PATCH_SIZE, 17).unwrap();
        assert_eq!(v.shape(), &[1, 37 * 10 * 18, LATENT_CHANNELS as i32 * 4]);
        assert_eq!(a.shape(), &[1, 207 * 2, 32]);
        let (v2, _) = initial_latents(&g, PATCH_SIZE, 17).unwrap();
        assert_eq!(v.as_slice::<f32>(), v2.as_slice::<f32>(), "seed is honored");
        let (v3, _) = initial_latents(&g, PATCH_SIZE, 18).unwrap();
        assert_ne!(v.as_slice::<f32>(), v3.as_slice::<f32>());
    }
}
