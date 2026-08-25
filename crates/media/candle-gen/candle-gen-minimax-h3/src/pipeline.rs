//! The **`t2va` / `fl2va` pipeline** (sc-17156, candle): request geometry, latent packing, the
//! cancellable render core, and the delivery-time audio/video length policy.
//!
//! ```text
//! prompt ─► text encoder (66.7 GB) ─► context ─┐            drop + free
//!                                              ▼
//!                        DiT (66.3 GB → 40.4 GB after the AdaLN evict) ─► denoise_av
//!                                              │            drop + free
//!                     ┌────────────────────────┴───────────────────────┐
//!              video rows ─► unpatchify ─► video VAE ─► frames    audio rows ─► audio VAE ─► track
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
//!   frames and every other count desynchronizes the picture from the soundtrack.
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
//! ([`crate::denoise::JointGeometry::av_drift_seconds`], bounded at ±12.5 ms). See
//! [`fit_audio_to_video`] for the decision and its reasoning.
//!
//! # Where this diverges from the MLX sibling, and why
//!
//! MLX is lazily evaluated, so its render core ends in an explicit `eval` to keep the compute inside
//! the cancel-checked region. candle is **eager**: every op in [`render_latents`] has already
//! happened by the time it returns, and `denoise_av` synchronizes the device at each step boundary,
//! so there is no deferred tail to force. The invariant is the same; only MLX needs an instrument
//! for it.

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::{AudioTrack, CancelFlag, Image};
use candle_gen::seed::seeded_normal_vec;
use candle_gen::{CandleError, Result};
use rand::SeedableRng;

use crate::audio_config::{AUDIO_LATENT_CHANNELS, AUDIO_OUTPUT_CHANNELS, AUDIO_SAMPLE_RATE};

/// [`AUDIO_OUTPUT_CHANNELS`] as a count. The constant is `u16` because it is also the `AudioTrack`
/// channel field's type; every shape here is `usize`.
const AUDIO_CHANNELS: usize = AUDIO_OUTPUT_CHANNELS as usize;
use crate::config::{LATENT_CHANNELS, VAE_RATIO};
use crate::denoise::{
    adaln_schedule, align_num_frames, denoise_av, video_latent_num_frames, JointGeometry,
    JointSchedule, JointVelocity, PackedLayout, LEGAL_FRAME_COUNTS, MAX_AV_DRIFT_SECONDS,
    MINIMAX_H3_FPS, TEXT_TAG,
};
use crate::dit::positions::{KeyframeAnchor, ReferenceLatentGeometry};

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
/// Enforced by [`resolve_geometry`], not merely declared: the per-edge `Capabilities::max_size`
/// never constrains a product, so `1344 x 1344` would otherwise pass every check at 1.75x this
/// budget. Canvas area is the *dominant* term in the packed sequence — it sets rows per latent
/// frame, and attention is quadratic in the sequence — so an unbounded area is an unbounded render
/// cost regardless of what the duration bound says.
pub const CANVAS_MAX_PIXELS: u32 = 768 * 1344;

/// Widest edge any canvas this model resolves to can carry — **2016**, the long edge at the 4:1
/// aspect ceiling. This is what `Capabilities::max_size` declares (sc-17152).
///
/// # Derived from the canvas resolver, not picked
///
/// [`crate::keyframe::resolve_canvas_size`] is the model's own arithmetic for turning an aspect
/// ratio into a canvas: it lays the short edge at [`CANVAS_SHORT_EDGE`], scales both edges by
/// `sqrt(CANVAS_MAX_PIXELS / area)` once the area is over budget, then snaps each axis to
/// [`SPATIAL_STRIDE`]. At [`crate::keyframe::MAX_ASPECT_RATIO`] — the widest ratio the model
/// accepts at all — that yields `512 x 2016`. So 2016 is the widest edge the model will ever put
/// a picture on, and a per-edge ceiling below it would refuse a canvas the model resolves to on
/// its own. `capability_ceiling_is_the_widest_canvas_the_resolver_emits` pins the two together so
/// this constant cannot drift from the resolver that justifies it.
///
/// # This does not widen the envelope
///
/// [`CANVAS_MAX_PIXELS`] still carries the real constraint and is checked as a *product*. Raising
/// the per-edge ceiling admits exactly one shape it did not before — `2016 x 512` and its
/// transpose, which is **exactly** at the area budget — and continues to refuse `1536 x 1536` and
/// `1344 x 1344`, both of which are inside this ceiling on both edges and far over the area.
pub const MAX_CANVAS_EDGE: u32 = 2016;

/// Shortest clip the released model generates, in seconds — **the lattice floor**, 124 frames.
pub const MIN_DURATION_SECONDS: f64 = LEGAL_FRAME_COUNTS[0] as f64 / MINIMAX_H3_FPS;

/// Longest clip the released model generates, in seconds — **the lattice ceiling**, 345 frames.
///
/// # 14.375 s, not the advertised 15.0
///
/// The reference pipeline's `max_duration` is **15.0 s**, but its `align_num_frames` walks the
/// `17n + 5` lattice and the largest rung inside that declaration is `LEGAL_FRAME_COUNTS[13] = 345`
/// = **14.375 s**. The next rung, 362 frames, is 15.083 s and past it. **No lattice point sits in
/// `[14.375, 15.0]`**, so a caller asking for the documented maximum has nothing exact to land on —
/// and the floor had the mirror-image gap at 5.0 s against 124 frames' 5.1667 s.
///
/// Both bounds are therefore **derived from the lattice** rather than declared alongside it, so the
/// declaration and the lattice cannot drift apart. The consequence is that
/// [`align_frames_for_duration`] needs no ceiling clamp: every duration in range aligns *upward*
/// onto a real rung, at both ends.
pub const MAX_DURATION_SECONDS: f64 =
    LEGAL_FRAME_COUNTS[LEGAL_FRAME_COUNTS.len() - 1] as f64 / MINIMAX_H3_FPS;

/// Per-channel mean the video VAE's pixel space is normalized by — ImageNet's.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Per-channel standard deviation the video VAE's pixel space is normalized by — ImageNet's.
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The DiT's `(t, h, w)` patch. Read from the loaded config in the generator; declared here so the
/// stride arithmetic above has a named source.
pub const PATCH_SIZE: [usize; 3] = [1, 2, 2];

/// The **smallest** legal render: 124 frames, i.e. 5.1667 s.
///
/// Named because it is what a gating first-light render should use — the lattice's floor, not the
/// `4k + 1` a temporal-downsample argument suggests.
pub const SMALLEST_LEGAL_FRAMES: usize = 124;

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
        self.joint.num_frames as f64 / MINIMAX_H3_FPS
    }

    /// Samples **per channel** the delivered soundtrack carries — see [`fit_audio_to_video`].
    ///
    /// Delegates to [`crate::denoise::delivered_audio_samples`] rather than restating the
    /// arithmetic: the geometry module owns the AV mux quantities, and an inline copy here is
    /// exactly the two-readers-drift shape this crate keeps finding.
    pub fn delivered_audio_samples(&self) -> usize {
        crate::denoise::delivered_audio_samples(self.joint.num_frames)
    }
}

/// Resolve a request's geometry, **rejecting** anything off the lattice.
///
/// Errors — never rounds — on a canvas that is not a multiple of [`SPATIAL_STRIDE`], on a canvas
/// over [`CANVAS_MAX_PIXELS`], on a frame count that is not `17n + 5`, and on a duration outside the
/// model's own 5.1667–14.375 s range.
pub fn resolve_geometry(width: u32, height: u32, num_frames: usize) -> Result<RequestGeometry> {
    if width == 0 || height == 0 {
        return Err(CandleError::Msg(format!(
            "minimax_h3: width/height must be positive, got {width}x{height}"
        )));
    }
    if !width.is_multiple_of(SPATIAL_STRIDE) || !height.is_multiple_of(SPATIAL_STRIDE) {
        return Err(CandleError::Msg(format!(
            "minimax_h3: width/height must be a multiple of {SPATIAL_STRIDE} (the VAE's \
             {VAE_RATIO}x spatial compression times the DiT's width patch of {}), got \
             {width}x{height}",
            PATCH_SIZE[2]
        )));
    }
    // Checked as a product because the per-edge cap is not the same constraint: 1344x1344 is inside
    // `Capabilities::max_size` on both edges and 75 % over the area the model generates at.
    if u64::from(width) * u64::from(height) > u64::from(CANVAS_MAX_PIXELS) {
        return Err(CandleError::Msg(format!(
            "minimax_h3: {width}x{height} is {} px, over the released checkpoint's \
             {CANVAS_MAX_PIXELS} px canvas budget ({CANVAS_SHORT_EDGE} short edge at 16:9). Canvas \
             area sets the packed sequence length and the attention is quadratic in it",
            u64::from(width) * u64::from(height)
        )));
    }
    // **The lattice gate runs first, then the range gate.** The two reject disjoint mistakes and the
    // caller needs to be told which one it made: a count that is not `17n + 5` has no geometry at
    // any duration, while one that is on the lattice but outside 5.1667-14.375 s is a legal shape
    // the model was not trained at.
    video_latent_num_frames(num_frames)?;
    let seconds = num_frames as f64 / MINIMAX_H3_FPS;
    if !(MIN_DURATION_SECONDS - 1e-9..=MAX_DURATION_SECONDS + 1e-9).contains(&seconds) {
        return Err(CandleError::Msg(format!(
            "minimax_h3: {num_frames} frames is {seconds:.4} s, outside the model's \
             {MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS} s range ({}-{} frames at \
             {MINIMAX_H3_FPS} fps)",
            LEGAL_FRAME_COUNTS[0],
            LEGAL_FRAME_COUNTS[LEGAL_FRAME_COUNTS.len() - 1],
        )));
    }
    Ok(RequestGeometry {
        width,
        height,
        joint: JointGeometry::new(
            num_frames,
            height as usize / VAE_RATIO,
            width as usize / VAE_RATIO,
        )?,
    })
}

/// The nearest legal frame count at or above `seconds · 24`, for a caller that opts in to alignment.
///
/// The requested **duration** is range-checked against
/// [`MIN_DURATION_SECONDS`]–[`MAX_DURATION_SECONDS`], which are the lattice's own two ends, and the
/// count is then aligned **upward** onto a rung. Because the range is the lattice's, the walk can
/// never leave it, so no ceiling clamp is needed and there is no duration in range that silently
/// renders shorter than it asked for.
pub fn align_frames_for_duration(seconds: f32) -> Result<usize> {
    if !seconds.is_finite() {
        return Err(CandleError::Msg(format!(
            "minimax_h3: duration must be finite, got {seconds}"
        )));
    }
    let seconds = f64::from(seconds);
    if !(MIN_DURATION_SECONDS - 1e-6..=MAX_DURATION_SECONDS + 1e-6).contains(&seconds) {
        return Err(CandleError::Msg(format!(
            "minimax_h3: duration {seconds} s is outside the model's \
             {MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS} s range (the {} legal frame counts \
             124-345 at {MINIMAX_H3_FPS} fps; the reference pipeline advertises 5-15 s but its own \
             17n+5 lattice tops out at 14.375 s)",
            LEGAL_FRAME_COUNTS.len()
        )));
    }
    let requested = (seconds * MINIMAX_H3_FPS).round().max(1.0) as usize;
    let aligned = align_num_frames(requested);
    debug_assert!(
        LEGAL_FRAME_COUNTS.contains(&aligned),
        "an in-range duration must align onto a lattice rung, got {aligned}"
    );
    Ok(aligned)
}

/// Pack `[1, C, T, H, W]` video latents into transformer rows, **frame-major then row-major**.
///
/// `patchify_video_latents` in the reference. The permutation is the whole of it: reading the axes
/// in any other order produces a correctly-shaped `[rows, C·prod(patch)]` tensor whose rows do not
/// correspond to the `(t, h, w)` coordinates [`crate::dit::positions`] assigns them, which is a
/// silently different model.
pub fn patchify_video_latents(latents: &Tensor, patch: [usize; 3]) -> Result<Tensor> {
    let s = latents.dims();
    if s.len() != 5 || s[0] != 1 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 patchify: expected [1, C, T, H, W], got {s:?}"
        )));
    }
    let (c, t, h, w) = (s[1], s[2], s[3], s[4]);
    let [pt, ph, pw] = patch;
    if pt == 0 || ph == 0 || pw == 0 || t % pt != 0 || h % ph != 0 || w % pw != 0 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 patchify: a {t}x{h}x{w} latent is not divisible by the patch {patch:?}"
        )));
    }
    // Rank 8 — past the arity candle's tuple `Shape`/`Dims` impls cover, so both the reshape and
    // the permutation are spelled as slices.
    let blocked = latents
        .reshape(&[1, c, t / pt, pt, h / ph, ph, w / pw, pw][..])?
        .contiguous()?;
    // (B, C, T', pt, H', ph, W', pw) -> (B, T', H', W', C, pt, ph, pw)
    let rows = blocked
        .permute(&[0, 2, 4, 6, 1, 3, 5, 7][..])?
        .contiguous()?;
    Ok(rows.reshape((1, (t / pt) * (h / ph) * (w / pw), c * pt * ph * pw))?)
}

/// The exact inverse of [`patchify_video_latents`]: `[1, rows, C·prod(patch)]` → `[1, C, T, H, W]`.
pub fn unpatchify_video_rows(
    rows: &Tensor,
    channels: usize,
    num_latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    patch: [usize; 3],
) -> Result<Tensor> {
    let [pt, ph, pw] = patch;
    if pt == 0 || ph == 0 || pw == 0 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 unpatchify: the patch {patch:?} must be positive"
        )));
    }
    let (tt, hh, ww) = (
        num_latent_frames / pt,
        latent_height / ph,
        latent_width / pw,
    );
    let want = [1, tt * hh * ww, channels * pt * ph * pw];
    if rows.dims() != want {
        return Err(CandleError::Msg(format!(
            "minimax_h3 unpatchify: expected {want:?} video rows, got {:?}",
            rows.dims()
        )));
    }
    let blocked = rows
        .reshape(&[1, tt, hh, ww, channels, pt, ph, pw][..])?
        .contiguous()?;
    // The inverse permutation of the patchify's (0, 2, 4, 6, 1, 3, 5, 7).
    let back = blocked
        .permute(&[0, 4, 1, 5, 2, 6, 3, 7][..])?
        .contiguous()?;
    Ok(back.reshape((1, channels, num_latent_frames, latent_height, latent_width))?)
}

/// Unpack the channel-major audio rows into the audio VAE's `[1, channels, latent_channels, T]`.
///
/// The rows arrive as `[1, num_audio_latents · channels, latent_channels]`, every latent of channel
/// 0 then every latent of channel 1 — the packing [`crate::dit::positions`] assigns coordinates to.
/// The decoder is **mono**, so the stereo axis is a batch axis, not a feature one.
///
/// `channels` and `num_audio_latents` enter the row count only as their **product**, so a caller
/// that transposed them would pass every shape check. They are therefore not inferred: the channel
/// count is required to be [`AUDIO_OUTPUT_CHANNELS`], which the model is fixed at.
pub fn unpack_audio_rows(
    rows: &Tensor,
    num_audio_latents: usize,
    channels: usize,
    latent_channels: usize,
) -> Result<Tensor> {
    if channels != AUDIO_CHANNELS {
        return Err(CandleError::Msg(format!(
            "minimax_h3 audio unpack: MiniMax-H3 is {AUDIO_OUTPUT_CHANNELS}-channel; a \
             {channels}-channel unpack would be indistinguishable by shape from a transposed \
             (latents, channels) pair"
        )));
    }
    let want = [1, num_audio_latents * channels, latent_channels];
    if rows.dims() != want {
        return Err(CandleError::Msg(format!(
            "minimax_h3 audio unpack: expected {want:?} audio rows, got {:?}",
            rows.dims()
        )));
    }
    let per_channel = rows.reshape((channels, num_audio_latents, latent_channels))?;
    // (C, T, L) -> (C, L, T), then the batch axis the audio VAE takes.
    let ncl = per_channel.permute((0, 2, 1))?.contiguous()?;
    Ok(ncl.reshape((1, channels, latent_channels, num_audio_latents))?)
}

/// Revert the video VAE's ImageNet pixel normalization and clamp to `[0, 1]`.
///
/// `video · PIXEL_STD + PIXEL_MEAN`, clamped. The VAE decodes into ImageNet-normalized RGB over a
/// `[0, 1]` base range, so a port that skipped this produces a picture with the right structure and
/// the wrong colour balance — plausible enough to ship.
pub fn revert_pixel_normalization(video: &Tensor) -> Result<Tensor> {
    let s = video.dims();
    if s.len() != 5 || s[1] != 3 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 decode: expected [B, 3, T, H, W], got {s:?}"
        )));
    }
    let dev = video.device();
    let x = video.to_dtype(DType::F32)?;
    let mean = Tensor::from_vec(PIXEL_MEAN.to_vec(), (1, 3, 1, 1, 1), dev)?;
    let std = Tensor::from_vec(PIXEL_STD.to_vec(), (1, 3, 1, 1, 1), dev)?;
    Ok(x.broadcast_mul(&std)?
        .broadcast_add(&mean)?
        .clamp(0f32, 1f32)?)
}

/// `[1, 3, T, H, W]` in `[0, 1]` → one 8-bit RGB [`Image`] per frame.
pub fn frames_to_images(video: &Tensor) -> Result<Vec<Image>> {
    let s = video.dims();
    if s.len() != 5 || s[0] != 1 || s[1] != 3 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 decode: expected [1, 3, T, H, W], got {s:?}"
        )));
    }
    let (t, h, w) = (s[2], s[3], s[4]);
    // NCTHW -> NTHWC, made contiguous BEFORE the flatten: a permuted candle tensor is a view, and
    // `to_vec1` over a non-contiguous view is exactly the interleave bug the MLX sibling's
    // `as_slice` note describes. Never forms the whole `t·h·w·3` product as one dimension.
    let bytes = (video.to_dtype(DType::F32)? * 255.0)?
        .round()?
        .clamp(0f32, 255f32)?
        .to_dtype(DType::U8)?
        .permute((0, 2, 3, 4, 1))?
        .contiguous()?
        .reshape((t, h * w * 3))?;
    let per = h * w * 3;
    let mut out = Vec::with_capacity(t);
    for i in 0..t {
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels: bytes.i(i)?.to_vec1::<u8>()?,
        });
        debug_assert_eq!(out[i].pixels.len(), per);
    }
    Ok(out)
}

/// How the delivered soundtrack is reconciled with the delivered picture.
///
/// # The decision: **the audio is fitted to the picture** — trimmed or silence-padded at the tail.
///
/// The two tracks are the same length at only 5 of the 14 legal durations. The residual is bounded
/// at [`MAX_AV_DRIFT_SECONDS`] and in practice ±8.33 ms or 0, cycling with period 3.
///
/// Four things decide it:
///
/// 1. **The picture is the exact quantity and the soundtrack is the derived one.** `num_frames` is
///    on the model's own `17n + 5` lattice and its duration is exact; `num_audio_latents` is a
///    `round` of it. Correcting the rounded side is correcting the side that carries the error.
/// 2. **The correction is inaudible and cannot be otherwise.** At most 400 samples of a 165 000+
///    sample track — a third of one 25 ms audio token, and roughly half the ~20 ms threshold at
///    which audio/video desync becomes perceptible at all.
/// 3. **The delivery contract has no place to express two durations.** `GenerationOutput::Video`
///    carries one frame count and one track; a consumer computes the clip length as `frames / fps`
///    and then truncates or pads the audio *itself*, with no knowledge of which end is safe to
///    touch. Deciding here, where the geometry is known, replaces a player-dependent behaviour with
///    a stated one.
/// 4. **The alternative — pad or trim the picture — is worse.** Adding or dropping a frame moves
///    `num_frames` off `17n + 5`, and it changes the creative payload rather than an inaudible tail.
///
/// Enforced rather than documented: this is the only path from decoded PCM to the delivered
/// [`AudioTrack`], it computes the target from `num_frames` and `fps` alone, and it **errors** if
/// the correction it would apply exceeds [`MAX_AV_DRIFT_SECONDS`] — a correction that large means
/// the geometry is wrong, not that the mux policy has work to do.
pub fn fit_audio_to_video(track: AudioTrack, geometry: &RequestGeometry) -> Result<AudioTrack> {
    let channels = usize::from(track.channels);
    if channels == 0 {
        return Err(CandleError::Msg(
            "minimax_h3 mux: the decoded soundtrack declares zero channels".into(),
        ));
    }
    if !track.samples.len().is_multiple_of(channels) {
        return Err(CandleError::Msg(format!(
            "minimax_h3 mux: {} interleaved samples is not a whole number of {channels}-channel \
             frames",
            track.samples.len()
        )));
    }
    let have = track.samples.len() / channels;
    let want = geometry.delivered_audio_samples();
    let drift = (have as f64 - want as f64) / f64::from(track.sample_rate.max(1));
    if drift.abs() > MAX_AV_DRIFT_SECONDS + 1e-6 {
        return Err(CandleError::Msg(format!(
            "minimax_h3 mux: the decoded soundtrack is {have} samples against the {want} that {} \
             frames at {MINIMAX_H3_FPS} fps implies — {drift:+.4} s, beyond the \
             {MAX_AV_DRIFT_SECONDS} s the audio-latent rounding can produce. That is a geometry \
             error, not a mux one",
            geometry.joint.num_frames
        )));
    }

    let mut out = track;
    fit_len(&mut out.samples, want * channels);
    for stem in &mut out.stems {
        fit_len(&mut stem.samples, want * channels);
    }
    Ok(out)
}

fn fit_len(samples: &mut Vec<f32>, want: usize) {
    match samples.len().cmp(&want) {
        std::cmp::Ordering::Greater => samples.truncate(want),
        std::cmp::Ordering::Less => samples.resize(want, 0.0),
        std::cmp::Ordering::Equal => {}
    }
}

/// The two latent blocks a request starts from, drawn from one seed.
///
/// Video noise is drawn as a `[1, C, T, H, W]` latent and then patchified — **not** drawn directly
/// in row layout — because that is the order the reference's generator produces, and patchify is a
/// pure permutation so the two differ only in which sample lands on which row.
///
/// Both blocks are f32, which is the dtype the reference's latents are and the dtype the sigma
/// schedule steps in.
pub fn initial_latents(
    geometry: &RequestGeometry,
    patch: [usize; 3],
    seed: u64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let g = &geometry.joint;
    let shape = (
        1usize,
        LATENT_CHANNELS,
        g.num_latent_frames,
        g.latent_height,
        g.latent_width,
    );
    let n = shape.1 * shape.2 * shape.3 * shape.4;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let video = Tensor::from_vec(seeded_normal_vec(&mut rng, n), shape, device)?;
    let video_rows = patchify_video_latents(&video, patch)?;

    // A separate stream, keyed off the same request seed: the two modalities are drawn from one
    // seed but must not share samples, or the soundtrack's noise would be a reshaped copy of the
    // picture's.
    let mut arng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(1));
    let audio_shape = (
        1usize,
        g.num_audio_latents * AUDIO_CHANNELS,
        AUDIO_LATENT_CHANNELS,
    );
    let an = audio_shape.1 * audio_shape.2;
    let audio_rows = Tensor::from_vec(seeded_normal_vec(&mut arng, an), audio_shape, device)?;
    Ok((video_rows, audio_rows))
}

/// The result of [`render_latents`]: the two denoised latent blocks, unpacked into the shapes the
/// two VAEs consume.
#[derive(Debug, Clone)]
pub struct RenderedLatents {
    /// `[1, LATENT_CHANNELS, num_latent_frames, latent_height, latent_width]`.
    pub video: Tensor,
    /// `[1, AUDIO_OUTPUT_CHANNELS, AUDIO_LATENT_CHANNELS, num_audio_latents]`.
    pub audio: Tensor,
}

/// **The cancellable render core**: build the AdaLN schedule, run the joint loop, and unpack the
/// results into the two VAEs' shapes.
///
/// Unlike the MLX sibling this needs no terminal force: candle is eager, and `denoise_av`
/// synchronizes the device at each step boundary, so every op below has already landed when this
/// returns — inside the cancel-checked region by construction rather than by instrument.
#[allow(clippy::too_many_arguments)]
pub fn render_latents(
    model: &mut dyn JointVelocity,
    layout: &PackedLayout,
    schedule: &JointSchedule,
    video_rows: &Tensor,
    audio_rows: &Tensor,
    patch: [usize; 3],
    device: &Device,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<RenderedLatents> {
    let g = *layout.geometry();
    let adaln = adaln_schedule(schedule)?;
    let (video, audio) = denoise_av(
        model, layout, schedule, &adaln, video_rows, audio_rows, device, cancel, on_step,
    )?;

    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }
    // Only the generated tail is delivered; `fl2va` anchors sit ahead of it.
    let vskip = layout.num_condition_video_rows();
    let askip = layout.num_condition_audio_rows();
    let generated_video = video.narrow(1, vskip, video.dim(1)? - vskip)?;
    let generated_audio = audio.narrow(1, askip, audio.dim(1)? - askip)?;
    Ok(RenderedLatents {
        video: unpatchify_video_rows(
            &generated_video.contiguous()?,
            LATENT_CHANNELS,
            g.num_latent_frames,
            g.latent_height,
            g.latent_width,
            patch,
        )?,
        audio: unpack_audio_rows(
            &generated_audio.contiguous()?,
            g.num_audio_latents,
            AUDIO_CHANNELS,
            AUDIO_LATENT_CHANNELS,
        )?,
    })
}

/// Build the packed layout for a plain `t2va` request — no keyframe anchors, no reference audio.
pub fn t2va_layout(
    geometry: &RequestGeometry,
    num_text_tokens: usize,
    patch: [usize; 3],
    device: &Device,
) -> Result<PackedLayout> {
    if num_text_tokens == 0 {
        return Err(CandleError::Msg(
            "minimax_h3: the packed sequence needs at least one text row".into(),
        ));
    }
    let tags = vec![TEXT_TAG; num_text_tokens];
    PackedLayout::build(geometry.joint, patch, &tags, AUDIO_CHANNELS, &[], device)
}

/// Build the packed layout for an `fl2va` request — keyframe anchors, and **per-row** text tags.
///
/// The two differences from [`t2va_layout`] are not cosmetic:
///
/// * `text_token_tags` is supplied per row rather than filled with [`TEXT_TAG`], because a
///   keyframe's vision-block rows are tagged **video** and therefore address a different block of
///   the AdaLN modulation table;
/// * `anchors` reserves the conditioning rows that lead the video stream, at their own rotary times
///   and their own row class.
///
/// `t2va` is `fl2va` with empty `anchors` **at the layout level only**. It is not the same path at
/// the block level: the reference selects a different text-encoder step, a different latent prep and
/// a different core denoise step on the presence of a keyframe, and this crate mirrors that split
/// rather than pretending one is a special case of the other.
pub fn fl2va_layout(
    geometry: &RequestGeometry,
    text_token_tags: &[u32],
    anchors: &[KeyframeAnchor],
    patch: [usize; 3],
    device: &Device,
) -> Result<PackedLayout> {
    if text_token_tags.is_empty() {
        return Err(CandleError::Msg(
            "minimax_h3: the packed sequence needs at least one text row".into(),
        ));
    }
    PackedLayout::build(
        geometry.joint,
        patch,
        text_token_tags,
        AUDIO_CHANNELS,
        anchors,
        device,
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
pub fn audio_track_to_encoder_input(track: &AudioTrack, device: &Device) -> Result<Tensor> {
    let channels = track.channels as usize;
    if channels == 0 {
        return Err(CandleError::Msg(
            "minimax_h3: a reference soundtrack declares zero channels".into(),
        ));
    }
    if track.samples.is_empty() {
        return Err(CandleError::Msg(
            "minimax_h3: a reference soundtrack carries no samples".into(),
        ));
    }
    if !track.samples.len().is_multiple_of(channels) {
        return Err(CandleError::Msg(format!(
            "minimax_h3: a {}-channel soundtrack cannot hold {} interleaved samples",
            channels,
            track.samples.len()
        )));
    }
    if track.sample_rate != AUDIO_SAMPLE_RATE {
        return Err(CandleError::Msg(format!(
            "minimax_h3: a reference soundtrack must be resampled onto the audio VAE's \
             {AUDIO_SAMPLE_RATE} Hz before encoding, got {} Hz",
            track.sample_rate
        )));
    }
    let per_channel = track.samples.len() / channels;
    // De-interleave into channel-major order: every sample of channel 0, then channel 1.
    let mut planar = Vec::with_capacity(track.samples.len());
    for c in 0..channels {
        planar.extend(track.samples.iter().skip(c).step_by(channels).copied());
    }
    Ok(Tensor::from_vec(
        planar,
        (channels, 1, per_channel),
        device,
    )?)
}

/// Build the packed layout for a **`ref2va`** request — ordered multi-modal reference blocks.
///
/// `references` is in packed order and carries the *encoded* latent geometry of every reference, so
/// the layout and the conditioning rows are described once. See
/// [`PackedLayout::build_ref2va`] for why this is a separate constructor rather than an option on
/// the `fl2va` one, and [`crate::reference`] for why the order is semantic.
pub fn ref2va_layout(
    geometry: &RequestGeometry,
    text_token_tags: &[u32],
    references: &[ReferenceLatentGeometry],
    patch: [usize; 3],
    device: &Device,
) -> Result<PackedLayout> {
    if text_token_tags.is_empty() {
        return Err(CandleError::Msg(
            "minimax_h3: the packed sequence needs at least one text row".into(),
        ));
    }
    PackedLayout::build_ref2va(
        geometry.joint,
        patch,
        text_token_tags,
        AUDIO_CHANNELS,
        references,
        device,
    )
}

/// Prepend a request's conditioning rows to its freshly-drawn video rows.
///
/// `MiniMaxH3FL2VAPrepareLatentsStep` in one line: the anchors **lead** the video row stream, and
/// the scheduler then writes only the tail ([`PackedLayout::generated_video_rows`]), which is how
/// they ride through every step untouched.
///
/// Checked against the layout rather than trusted, because a conditioning block of the wrong height
/// still concatenates cleanly and produces a runnable — and silently misaligned — sequence.
pub fn prepend_condition_rows(
    layout: &PackedLayout,
    condition_rows: Option<&Tensor>,
    video_rows: &Tensor,
) -> Result<Tensor> {
    prepend_rows(
        layout.num_condition_video_rows(),
        condition_rows,
        video_rows,
        "conditioning video",
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
    condition_rows: Option<&Tensor>,
    audio_rows: &Tensor,
) -> Result<Tensor> {
    prepend_rows(
        layout.num_condition_audio_rows(),
        condition_rows,
        audio_rows,
        "reference soundtrack",
    )
}

/// The shape-and-count contract both prepends share.
fn prepend_rows(
    expected: usize,
    condition_rows: Option<&Tensor>,
    generated: &Tensor,
    what: &str,
) -> Result<Tensor> {
    match (expected, condition_rows) {
        (0, None) => Ok(generated.clone()),
        (0, Some(_)) => Err(CandleError::Msg(format!(
            "minimax_h3: {what} rows were supplied for a layout that reserves none"
        ))),
        (n, None) => Err(CandleError::Msg(format!(
            "minimax_h3: the layout reserves {n} {what} row(s) but none were supplied"
        ))),
        (n, Some(rows)) => {
            let d = rows.dims();
            if d.len() != 3 || d[0] != 1 || d[1] != n || d[2] != generated.dim(2)? {
                return Err(CandleError::Msg(format!(
                    "minimax_h3: expected [1, {n}, {}] {what} rows, got {d:?}",
                    generated.dim(2)?
                )));
            }
            Ok(Tensor::cat(&[&rows.to_dtype(generated.dtype())?, generated], 1)?.contiguous()?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::Cpu
    }

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
        // 4k+1 is the plausible wrong lattice, and 125 is one past the floor.
        for frames in [121, 123, 125, 129, 140, 200, 241] {
            assert!(resolve_geometry(576, 320, frames).is_err(), "{frames}");
        }
        // Durations outside the range, even when they ARE 17n+5.
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

    /// **The canvas area budget is enforced, not just declared.**
    ///
    /// `Capabilities::max_size` caps each edge *independently* and cannot bound a product, so the
    /// square 1344×1344 is
    /// accepted by every per-edge check at 1.75× the area the checkpoint generates at — and area is
    /// the dominant term in the packed sequence length, hence in an attention cost quadratic in it.
    #[test]
    fn the_canvas_area_budget_is_enforced() {
        assert_eq!(CANVAS_MAX_PIXELS, 1_032_192);
        resolve_geometry(1344, 768, 124).unwrap();
        resolve_geometry(768, 1344, 124).unwrap();
        for (w, h) in [(1344u32, 1344u32), (1344, 1024), (1088, 1088)] {
            let e = resolve_geometry(w, h, 124).unwrap_err().to_string();
            assert!(e.contains("canvas budget"), "{w}x{h}: {e}");
        }
    }

    /// The duration bounds are the lattice's own ends, and alignment is upward at both of them.
    #[test]
    fn duration_alignment_walks_up_onto_a_rung() {
        assert_eq!(align_frames_for_duration(5.1667).unwrap(), 124);
        assert_eq!(align_frames_for_duration(5.2).unwrap(), 141);
        assert_eq!(align_frames_for_duration(14.3).unwrap(), 345);
        // 15.0 s is REFUSED, not clamped down to 14.375 s.
        assert!(align_frames_for_duration(15.0).is_err());
        assert!(align_frames_for_duration(5.0).is_err());
        assert!(align_frames_for_duration(f32::NAN).is_err());
        for &f in &LEGAL_FRAME_COUNTS {
            let secs = f as f32 / MINIMAX_H3_FPS as f32;
            assert_eq!(align_frames_for_duration(secs).unwrap(), f);
        }
    }

    /// Patchify and unpatchify are exact inverses, and the permutation is the load-bearing part:
    /// a round trip through a *different* axis order also round-trips, so the test additionally
    /// pins which latent voxel lands on which row.
    #[test]
    fn patchify_round_trips_and_places_voxels_by_coordinate() {
        let d = dev();
        let (c, t, h, w) = (2usize, 3usize, 4usize, 6usize);
        let n = c * t * h * w;
        let x = Tensor::from_vec(
            (0..n).map(|i| i as f32).collect::<Vec<_>>(),
            (1, c, t, h, w),
            &d,
        )
        .unwrap();
        let rows = patchify_video_latents(&x, PATCH_SIZE).unwrap();
        assert_eq!(rows.dims(), &[1, t * (h / 2) * (w / 2), c * 4]);
        let back = unpatchify_video_rows(&rows, c, t, h, w, PATCH_SIZE).unwrap();
        assert_eq!(
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            back.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // Row 0 is the (t=0, h∈{0,1}, w∈{0,1}) block of both channels, channel-major then
        // (pt, ph, pw) — not the flat first 8 elements, which is what a missing permutation gives.
        let r0 = rows.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        let at =
            |ci: usize, ti: usize, hi: usize, wi: usize| (((ci * t + ti) * h + hi) * w + wi) as f32;
        assert_eq!(
            r0,
            vec![
                at(0, 0, 0, 0),
                at(0, 0, 0, 1),
                at(0, 0, 1, 0),
                at(0, 0, 1, 1),
                at(1, 0, 0, 0),
                at(1, 0, 0, 1),
                at(1, 0, 1, 0),
                at(1, 0, 1, 1),
            ]
        );
    }

    /// A latent that is not a whole number of patches is refused rather than truncated.
    #[test]
    fn an_unpatchable_latent_is_refused() {
        let d = dev();
        let x = Tensor::zeros((1, 2, 3, 5, 6), DType::F32, &d).unwrap();
        assert!(patchify_video_latents(&x, PATCH_SIZE).is_err());
        let bad = Tensor::zeros((1, 4, 8), DType::F32, &d).unwrap();
        assert!(unpatchify_video_rows(&bad, 2, 3, 4, 6, PATCH_SIZE).is_err());
    }

    /// The audio unpack is channel-major, and a transposed `(latents, channels)` pair is REFUSED —
    /// it would otherwise pass every shape check, since only the product reaches the row count.
    #[test]
    fn audio_unpack_is_channel_major_and_rejects_a_transpose() {
        let d = dev();
        let (lat, ch, lc) = (3usize, AUDIO_CHANNELS, 4usize);
        let n = lat * ch * lc;
        let rows = Tensor::from_vec(
            (0..n).map(|i| i as f32).collect::<Vec<_>>(),
            (1, lat * ch, lc),
            &d,
        )
        .unwrap();
        let out = unpack_audio_rows(&rows, lat, ch, lc).unwrap();
        assert_eq!(out.dims(), &[1, ch, lc, lat]);
        // Channel 0, latent-channel 0, over time = rows 0..lat at feature 0.
        let got = out.i((0, 0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, vec![0.0, 4.0, 8.0]);
        // A 1- or 3-channel unpack is a typed error, not a plausible reshape.
        assert!(unpack_audio_rows(&rows, lat * 2, 1, lc).is_err());
        assert!(unpack_audio_rows(&rows, 2, 3, lc).is_err());
    }

    /// The ImageNet de-normalization is applied per channel and clamped — a port that skipped it
    /// produces the right structure with the wrong colour balance.
    #[test]
    fn pixel_denormalization_is_per_channel_and_clamped() {
        let d = dev();
        let x = Tensor::zeros((1, 3, 1, 1, 1), DType::F32, &d).unwrap();
        let y = revert_pixel_normalization(&x).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            PIXEL_MEAN.to_vec()
        );
        // Out-of-range input clamps rather than wrapping.
        let hot = Tensor::ones((1, 3, 1, 1, 1), DType::F32, &d).unwrap();
        let hot = (hot * 100.0).unwrap();
        let y = revert_pixel_normalization(&hot).unwrap();
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|&v| v == 1.0));
        assert!(revert_pixel_normalization(
            &Tensor::zeros((1, 4, 1, 1, 1), DType::F32, &d).unwrap()
        )
        .is_err());
    }

    /// Frames come out one `Image` per time step, in HWC byte order — the transpose is materialized
    /// before the read, so channels are not interleaved across pixels.
    #[test]
    fn frames_to_images_emits_hwc_bytes_per_frame() {
        let d = dev();
        let (t, h, w) = (2usize, 2usize, 3usize);
        // A pure-red first frame and a pure-green second one: any channel interleave shows up.
        let mut v = vec![0f32; 3 * t * h * w];
        for ti in 0..t {
            let c = ti; // frame 0 -> channel 0 (red), frame 1 -> channel 1 (green)
            for hi in 0..h {
                for wi in 0..w {
                    v[((c * t + ti) * h + hi) * w + wi] = 1.0;
                }
            }
        }
        let x = Tensor::from_vec(v, (1, 3, t, h, w), &d).unwrap();
        let imgs = frames_to_images(&x).unwrap();
        assert_eq!(imgs.len(), t);
        assert_eq!((imgs[0].width, imgs[0].height), (w as u32, h as u32));
        assert_eq!(imgs[0].pixels.len(), h * w * 3);
        assert_eq!(&imgs[0].pixels[..3], &[255, 0, 0]);
        assert_eq!(&imgs[1].pixels[..3], &[0, 255, 0]);
    }

    fn track(samples: usize, channels: u16) -> AudioTrack {
        AudioTrack {
            samples: vec![0.5; samples * usize::from(channels)],
            sample_rate: AUDIO_SAMPLE_RATE,
            channels,
            ..Default::default()
        }
    }

    /// The soundtrack is fitted to the picture, in both directions, and only by the tail.
    #[test]
    fn the_soundtrack_is_fitted_to_the_picture() {
        let g = resolve_geometry(576, 320, 124).unwrap();
        let want = g.delivered_audio_samples();
        // Exact: unchanged.
        let out = fit_audio_to_video(track(want, 2), &g).unwrap();
        assert_eq!(out.samples.len(), want * 2);
        // Long by one 40 Hz token boundary's worth: trimmed.
        let out = fit_audio_to_video(track(want + 200, 2), &g).unwrap();
        assert_eq!(out.samples.len(), want * 2);
        // Short: silence-padded, and the padding is at the TAIL.
        let out = fit_audio_to_video(track(want - 200, 2), &g).unwrap();
        assert_eq!(out.samples.len(), want * 2);
        assert_eq!(out.samples[0], 0.5);
        assert_eq!(*out.samples.last().unwrap(), 0.0);
    }

    /// A correction larger than the audio-latent rounding can produce is a GEOMETRY error, and must
    /// not be silently applied — which is the whole difference between a mux policy and a cover-up.
    #[test]
    fn an_oversized_correction_is_a_geometry_error() {
        let g = resolve_geometry(576, 320, 124).unwrap();
        let want = g.delivered_audio_samples();
        let e = fit_audio_to_video(track(want + 8000, 2), &g)
            .unwrap_err()
            .to_string();
        assert!(e.contains("geometry error"), "{e}");
        assert!(fit_audio_to_video(track(1, 0), &g).is_err());
    }

    /// Initial latents are seed-reproducible, seed-sensitive, and the two modalities do not share
    /// samples — a single stream reshaped in two would make the soundtrack a copy of the picture.
    #[test]
    fn initial_latents_are_seeded_and_the_modalities_are_independent() {
        let d = dev();
        let g = resolve_geometry(64, 64, 124).unwrap();
        let (v1, a1) = initial_latents(&g, PATCH_SIZE, 7, &d).unwrap();
        let (v2, a2) = initial_latents(&g, PATCH_SIZE, 7, &d).unwrap();
        let (v3, a3) = initial_latents(&g, PATCH_SIZE, 8, &d).unwrap();
        let f = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(f(&v1), f(&v2));
        assert_eq!(f(&a1), f(&a2));
        assert_ne!(f(&v1), f(&v3));
        assert_ne!(f(&a1), f(&a3));
        // Independent streams: the audio block must not be a prefix of the video one.
        let (va, aa) = (f(&v1), f(&a1));
        assert_ne!(va[..aa.len().min(va.len())], aa[..aa.len().min(va.len())]);
        assert_eq!(
            v1.dims(),
            &[
                1,
                g.joint.num_latent_frames
                    * (g.joint.latent_height / 2)
                    * (g.joint.latent_width / 2),
                LATENT_CHANNELS * 4
            ]
        );
        assert_eq!(
            a1.dims(),
            &[
                1,
                g.joint.num_audio_latents * AUDIO_CHANNELS,
                AUDIO_LATENT_CHANNELS
            ]
        );
    }

    /// The conditioning prepend is checked against the layout in BOTH directions: a block of the
    /// wrong height, and a block supplied when the layout reserves none, both concatenate cleanly
    /// and would produce a runnable, silently misaligned sequence.
    #[test]
    fn the_conditioning_prepend_is_checked_against_the_layout() {
        let d = dev();
        let g = resolve_geometry(64, 64, 124).unwrap();
        let layout = t2va_layout(&g, 4, PATCH_SIZE, &d).unwrap();
        assert_eq!(layout.num_condition_video_rows(), 0);
        let rows = Tensor::zeros((1, 8, LATENT_CHANNELS * 4), DType::F32, &d).unwrap();
        // t2va: nothing to prepend, and supplying something is an error.
        assert_eq!(
            prepend_condition_rows(&layout, None, &rows).unwrap().dims(),
            rows.dims()
        );
        let stray = Tensor::zeros((1, 1, LATENT_CHANNELS * 4), DType::F32, &d).unwrap();
        assert!(prepend_condition_rows(&layout, Some(&stray), &rows).is_err());
    }

    /// `t2va_layout` tags every text row TEXT; `fl2va_layout` takes the tags verbatim, because a
    /// vision block's rows are VIDEO and address a different AdaLN modulation block.
    #[test]
    fn the_two_layouts_differ_in_their_row_tags() {
        let d = dev();
        let g = resolve_geometry(64, 64, 124).unwrap();
        let t2va = t2va_layout(&g, 5, PATCH_SIZE, &d).unwrap();
        // `token_tags` covers the WHOLE packed sequence; the text rows lead it.
        assert_eq!(t2va.num_text_tokens(), 5);
        assert_eq!(&t2va.token_tags()[..5], &[TEXT_TAG; 5]);

        let tags = vec![
            TEXT_TAG,
            crate::denoise::VIDEO_TAG,
            crate::denoise::VIDEO_TAG,
            TEXT_TAG,
            TEXT_TAG,
        ];
        let fl2va = fl2va_layout(&g, &tags, &[], PATCH_SIZE, &d).unwrap();
        assert_eq!(&fl2va.token_tags()[..5], tags.as_slice());
        assert_ne!(&fl2va.token_tags()[..5], &t2va.token_tags()[..5]);

        // An anchored layout reserves conditioning video rows; the plain one does not.
        let anchored = fl2va_layout(&g, &tags, &[KeyframeAnchor::First], PATCH_SIZE, &d).unwrap();
        assert!(anchored.num_condition_video_rows() > 0);
        assert_eq!(fl2va.num_condition_video_rows(), 0);

        assert!(t2va_layout(&g, 0, PATCH_SIZE, &d).is_err());
        assert!(fl2va_layout(&g, &[], &[], PATCH_SIZE, &d).is_err());
    }
}
