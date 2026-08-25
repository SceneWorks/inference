//! The `(t, h, w)` rotary coordinates every modality carries into [`super::rope`].
//!
//! [`super::rope`] is modality-blind: it consumes a `[seq_len, 3]` grid and knows nothing about
//! what produced it. **This module is where the model's actual position conventions live**, and
//! sc-17144 calls the audio one the highest-risk correctness detail in the port — audio tokens
//! have no `h`, so a plausible-looking guess (zero everywhere, or reusing the video clock verbatim)
//! is shape-identical to the truth.
//!
//! Everything here is read from `diffusers/modular_pipelines/minimax_h3/before_denoise.py`
//! (`MiniMaxH3PrepareLayoutStep.build_packed_sequence`, the `t2va` / `fl2va` layout), not inferred.
//!
//! It is **pure host arithmetic in f64** — no tensor library appears until [`to_tensor`] — so this
//! module is byte-for-byte the same computation as its MLX sibling rather than a re-derivation.
//! That matters: `tests/dit_parity.rs` compares the grid against the reference's own
//! `build_packed_sequence` output at 1e-5 absolute, and a coordinate error would otherwise be
//! attributed to backend numerics.
//!
//! # Scope
//!
//! These are the **coordinate grids**, i.e. the rotary's input contract. Assembling the packed
//! sequence itself — the row order `[text | keyframe conditions | target audio | target video]`,
//! the three index tensors, the modality tags and the scatter — is
//! [`crate::denoise::packing`]. The grids are here because the rotary cannot be pinned without
//! them.
//!
//! # The four conventions
//!
//! | rows | `t` | `h` | `w` |
//! |---|---|---|---|
//! | text | `0 … num_text_tokens-1` | 0 | 0 |
//! | audio | `num_text_tokens + 0 … num_audio_latents-1`, **tiled per channel** | **0** | `width_grid[0]` for channel 0, `width_grid[-1]` for the rest |
//! | keyframe condition (`fl2va`) | one constant [`keyframe_anchor_time`] for the whole block | [`frame_grid`] | [`frame_grid`] |
//! | video | [`temporal_grid`] from `num_text_tokens` | [`frame_grid`] | [`frame_grid`] |
//!
//! Two consequences that are easy to miss:
//!
//! * **Text length shifts the whole media clock.** Media rows start their time axis at
//!   `num_text_tokens`, so the same video at two prompt lengths gets different rotary times.
//! * **Audio is channel-major and pinned to the two extremes of the video's width grid.** It rides
//!   the video's clock at one unit per latent — 40 audio latents/s against 24 fps × 5/3 — and its
//!   only spatial identity is which end of the canvas it sits at.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};

/// Rotary time per *frame* of a latent, before the per-latent frame counts:
/// `40 audio-latents-per-second / 24 fps` — the shared clock runs at 40 rotary units per second,
/// so each of the 24 frames in a second advances it by 40/24 = 5/3. `_ROPE_FRAME_RESCALE`.
pub const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Frames each latent frame stands for. `_ROPE_FRAMES_PER_LATENT`, indexed **cyclically** by the
/// latent's own index: latents 0, 5, 10, … cover 1 frame and every other latent covers 4.
///
/// The cycle is the point — reading this as "the first latent covers 1 and the rest cover 4" gives
/// a monotone-but-wrong clock that only diverges from latent 5 onwards, which is past every tiny
/// fixture and inside every real render (17n + 5 frames is 5k + 2 latents, so the shortest legal
/// render already has 37).
pub const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];

/// The spatial grids are aspect-normalized onto the unit interval and then scaled by this.
/// `_ROPE_SPATIAL_SCALE`, so a square canvas spans `[0, 32)`.
pub const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// Channels the soundtrack is packed channel-major over. Stereo.
pub const AUDIO_CHANNELS: usize = 2;

/// One aspect-normalized spatial axis: `dim / patch` coordinates centred on the unit interval and
/// scaled by [`ROPE_SPATIAL_SCALE`], right endpoint **excluded**.
///
/// `sqrt_area` is `sqrt(latent_height · latent_width)`, shared by both axes, which is what makes
/// the grid aspect-normalized: a 16:9 canvas gets a wider `w` span than `h`.
///
/// The reference builds this with `np.linspace(..., endpoint=False)`, whose step is
/// `(stop - start) / num` — **not** `torch.linspace`'s `/(num - 1)`. Reproduced here deliberately:
/// the two differ at every point except the first.
pub fn spatial_axis_grid(dim: usize, patch: usize, sqrt_area: f64) -> Result<Vec<f64>> {
    if dim == 0 || patch == 0 || !dim.is_multiple_of(patch) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit positions: latent extent {dim} must be a positive multiple of patch \
             {patch}"
        )));
    }
    if !sqrt_area.is_finite() || sqrt_area <= 0.0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit positions: sqrt_area must be positive, got {sqrt_area}"
        )));
    }
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let num = dim / patch;
    Ok((0..num)
        .map(|i| (left + i as f64 * ratio / num as f64) * ROPE_SPATIAL_SCALE)
        .collect())
}

/// The `(h, w)` coordinates of one latent frame's rows, in `ij` mesh order, plus the width axis
/// they were built from.
///
/// The width axis is returned because the **audio** rows are pinned to its two extremes.
pub fn frame_grid(
    latent_height: usize,
    latent_width: usize,
    patch_h: usize,
    patch_w: usize,
) -> Result<(Vec<[f64; 2]>, Vec<f64>)> {
    let sqrt_area = ((latent_height as f64) * (latent_width as f64)).sqrt();
    let h_grid = spatial_axis_grid(latent_height, patch_h, sqrt_area)?;
    let w_grid = spatial_axis_grid(latent_width, patch_w, sqrt_area)?;
    let mut rows = Vec::with_capacity(h_grid.len() * w_grid.len());
    for &h in &h_grid {
        for &w in &w_grid {
            rows.push([h, w]);
        }
    }
    Ok((rows, w_grid))
}

/// The rotary time of every latent frame, starting at `origin`.
///
/// Spacing is **non-uniform**: `5/3 × (1, 4, 4, 4, 4)` cycling, so frame 1 sits `5/3` after frame 0
/// but frame 2 sits `20/3` after frame 1. A uniform ramp is the obvious wrong guess.
pub fn temporal_grid(num_latent_frames: usize, origin: f64) -> Result<Vec<f64>> {
    if num_latent_frames == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 dit positions: num_latent_frames must be positive, got 0".into(),
        ));
    }
    let mut out = Vec::with_capacity(num_latent_frames);
    let mut acc = origin;
    for index in 0..num_latent_frames {
        out.push(acc);
        acc += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[index % ROPE_FRAMES_PER_LATENT.len()];
    }
    Ok(out)
}

/// Text rows: `t` is the row index, `h` and `w` are zero.
///
/// The time axis is shared with the media rows, so the prompt occupies `[0, num_text_tokens)` of it
/// and every media row is offset by `num_text_tokens`.
pub fn text_position_ids(num_text_tokens: usize) -> Result<Vec<[f64; 3]>> {
    Ok((0..num_text_tokens).map(|i| [i as f64, 0.0, 0.0]).collect())
}

/// **The audio-token position convention.**
///
/// `num_audio_latents · audio_channels` rows, **channel-major** — every latent of channel 0, then
/// every latent of channel 1:
///
/// * `t` = `num_text_tokens + latent_index`, tiled per channel. One unit per audio latent, which
///   equals the video clock's unit because 40 latents/s × (24 fps × 5/3)⁻¹ = 1.
/// * `h` = **0**. Audio has no height, and the reference leaves the column at its zero
///   initialization rather than assigning anything.
/// * `w` = `width_grid[0]` for the first `num_audio_latents` rows and `width_grid.last()` for
///   every remaining row — the two extremes of *the video's own* width grid, so the two channels
///   sit at opposite ends of the canvas.
///
/// `width_grid` is the second return of [`frame_grid`].
///
/// Upstream's two call sites disagree above stereo: `build_packed_sequence` gives the first
/// `num_audio_latents` rows the left extreme and **all** remaining rows the right one (implemented
/// here), while `_fill_audio_positions` hard-codes two equal halves. They coincide at the shipped
/// [`AUDIO_CHANNELS`] = 2, which is the only value either path is exercised at.
pub fn audio_position_ids(
    num_text_tokens: usize,
    num_audio_latents: usize,
    audio_channels: usize,
    width_grid: &[f64],
) -> Result<Vec<[f64; 3]>> {
    if num_audio_latents == 0 || audio_channels == 0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit positions: audio latents {num_audio_latents} and channels \
             {audio_channels} must be positive"
        )));
    }
    let (&first, &last) = match (width_grid.first(), width_grid.last()) {
        (Some(f), Some(l)) => (f, l),
        _ => {
            return Err(CandleError::Msg(
                "minimax-h3 dit positions: audio rows are pinned to the video width grid's \
                 extremes, but the grid is empty"
                    .into(),
            ))
        }
    };
    let rows = num_audio_latents * audio_channels;
    Ok((0..rows)
        .map(|row| {
            let latent = row % num_audio_latents;
            let w = if row < num_audio_latents { first } else { last };
            [num_text_tokens as f64 + latent as f64, 0.0, w]
        })
        .collect())
}

/// Which end of the generated video a `fl2va` keyframe conditioning block is anchored to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeAnchor {
    /// Anchored at the first latent frame — the block sits at the video clock's origin.
    First,
    /// Anchored at the last latent frame.
    Last,
}

/// The single rotary time every row of a `fl2va` keyframe conditioning block carries.
///
/// A keyframe block occupies one frame's worth of rows, all at **one constant time**, with the full
/// [`frame_grid`] as its `(h, w)` — it is a literal frame pinned to one end of the clip's clock.
///
/// * [`KeyframeAnchor::First`] → `num_text_tokens`, the video clock's origin.
/// * [`KeyframeAnchor::Last`] → `num_text_tokens + Σ spans - ROPE_FRAME_RESCALE`, which is the
///   rotary time of the clip's **final video frame** — equivalently
///   `origin + ROPE_FRAME_RESCALE · (total_frames - 1)`.
///
/// That second one is worth stating carefully, because the obvious reading is wrong: it is **not**
/// the last *latent's* time. [`temporal_grid`] gives each latent the time of its *first* frame, and
/// every latent after the first spans 4 frames, so the last latent starts 5 rotary units before the
/// clip ends. At 3 latents from origin 5 the last latent sits at 13.33 while this anchor is 18.33.
/// They coincide only when `(num_latent_frames - 1) % 5 == 0`, which is exactly the case a small
/// fixture is most likely to pick. Semantically the anchor is right: an `fl2va` last-frame keyframe
/// *is* the final frame, not the final latent.
///
/// # The summation order is deliberately not replicated
///
/// The reference computes the `Last` sum with **numpy's pairwise summation** and notes that the
/// `ref2va` soundtrack path sums the same series **sequentially**, the two differing in the last
/// ulp of f64 from 16 latent frames onwards — so it keeps both, one per call site.
///
/// This uses a plain sequential sum, because that distinction cannot survive the f64 → f32 narrow
/// the model performs on entry (`position_ids.to(torch.float32)`). Measured: the two orders differ
/// by at most 1.1e-11 in f64 and are **bit-identical in f32 for every latent-frame count up to
/// 400** — far beyond the 102 the longest legal render produces.
/// `keyframe_anchor_summation_order_is_invisible_after_the_f32_narrow` pins that, so if the claim
/// ever stops holding this fails rather than drifting.
pub fn keyframe_anchor_time(
    anchor: KeyframeAnchor,
    num_text_tokens: usize,
    num_latent_frames: usize,
) -> Result<f64> {
    if num_latent_frames == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 dit positions: num_latent_frames must be positive, got 0".into(),
        ));
    }
    let origin = num_text_tokens as f64;
    Ok(match anchor {
        KeyframeAnchor::First => origin,
        KeyframeAnchor::Last => {
            let total: f64 = (0..num_latent_frames)
                .map(|i| {
                    ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()]
                })
                .sum();
            origin + total - ROPE_FRAME_RESCALE
        }
    })
}

/// One `fl2va` keyframe conditioning block: the whole [`frame_grid`] at a single
/// [`keyframe_anchor_time`].
pub fn keyframe_position_ids(
    anchor: KeyframeAnchor,
    num_text_tokens: usize,
    num_latent_frames: usize,
    frame_rows: &[[f64; 2]],
) -> Result<Vec<[f64; 3]>> {
    if frame_rows.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 dit positions: the frame grid is empty".into(),
        ));
    }
    let t = keyframe_anchor_time(anchor, num_text_tokens, num_latent_frames)?;
    Ok(frame_rows.iter().map(|&[h, w]| [t, h, w]).collect())
}

/// The **latent** geometry one `ref2va` reference block occupies — what the encoders actually
/// produced for it, not what the request asked for.
///
/// Taken from the encoded latents rather than re-derived from the source media, so the layout and
/// the conditioning rows can never disagree about a reference's row count. The reference
/// implementation is explicit about this ("the geometry of every reference block is the shape of
/// what the encoder produced for it, so the two can never disagree"), and a mismatch here surfaces
/// as an `index_copy` shape error 50 layers deep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceLatentGeometry {
    /// Which modality this reference is — it decides both the time layout and the clock advance.
    pub kind: crate::reference::ReferenceKind,
    /// Video latent frames. `1` for an image, the encoded count for a clip, `0` for audio.
    pub num_latent_frames: usize,
    /// Latent height of **this reference's own** encode — references are encoded at their own
    /// resolution and do not share the target canvas.
    pub latent_height: usize,
    /// Latent width of this reference's own encode.
    pub latent_width: usize,
    /// Audio latents **per channel**. Non-zero for a standalone audio reference and for a clip
    /// carrying its own soundtrack; `0` otherwise.
    pub num_audio_latents: usize,
}

/// One `ref2va` reference block's contribution: `(audio_rows, video_rows, clock_advance)`.
///
/// Named rather than spelled inline because the tuple's *order* is load-bearing — audio first,
/// mirroring the packed row order — and an unnamed triple of two identical vector types is exactly
/// the shape a caller silently swaps.
pub type ReferenceBlockRows = (Vec<[f64; 3]>, Vec<[f64; 3]>, f64);

/// One `ref2va` reference block's rows and the rotary time it consumes.
///
/// Returns `(audio_rows, video_rows, advance)` — **audio first**, because a clip's soundtrack rows
/// are packed immediately *before* its video rows and share their origin, exactly as the generated
/// audio and video do.
///
/// # The three modalities advance the shared clock differently
///
/// `rotary_time` is one clock shared by audio and video. It starts where the text rows end and every
/// block pushes it forward by the time that block occupies:
///
/// * **image** — a single frame at a single **constant** time, advancing the clock by exactly
///   `1.0`. Not `5/3`: an image takes one integer rotary slot rather than a latent frame's span.
/// * **audio** — `num_audio_latents` rows pinned to the *target* canvas's width extremes, advancing
///   by `num_audio_latents`.
/// * **video** — its soundtrack (on **its own** width grid, not the target's), then its frames on a
///   [`temporal_grid`] from the same origin, advancing by
///   `max(num_audio_latents, video_span)` so a clip and its sound stay rotary-aligned.
///
pub fn reference_block_position_ids(
    geometry: &ReferenceLatentGeometry,
    origin: f64,
    patch_h: usize,
    patch_w: usize,
    audio_channels: usize,
    target_width_grid: &[f64],
) -> Result<ReferenceBlockRows> {
    use crate::reference::ReferenceKind;

    if !origin.is_finite() {
        return Err(CandleError::Msg(format!(
            "minimax-h3 dit positions: the reference rotary clock is at {origin}"
        )));
    }
    let mut audio_rows = Vec::new();
    let mut video_rows = Vec::new();
    let mut advance = 0.0f64;

    // The visual half, and the width grid this reference's audio is pinned to. A standalone audio
    // reference has no grid of its own and borrows the target's.
    let width_grid: Vec<f64> = match geometry.kind {
        ReferenceKind::Audio => target_width_grid.to_vec(),
        ReferenceKind::Image | ReferenceKind::Video => {
            if geometry.num_latent_frames == 0 {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 dit positions: a {} reference must occupy at least one latent \
                     frame, got 0",
                    geometry.kind.noun()
                )));
            }
            let (grid, w) = frame_grid(
                geometry.latent_height,
                geometry.latent_width,
                patch_h,
                patch_w,
            )?;
            match geometry.kind {
                ReferenceKind::Image => {
                    // An image is one frame at ONE constant time — not a temporal grid.
                    if geometry.num_latent_frames != 1 {
                        return Err(CandleError::Msg(format!(
                            "minimax-h3 dit positions: an image reference is a single latent \
                             frame, got {}",
                            geometry.num_latent_frames
                        )));
                    }
                    video_rows.extend(grid.iter().map(|&[h, w]| [origin, h, w]));
                    advance = 1.0;
                }
                _ => {
                    let times = temporal_grid(geometry.num_latent_frames, origin)?;
                    for &t in &times {
                        video_rows.extend(grid.iter().map(|&[h, w]| [t, h, w]));
                    }
                    // The same series [`keyframe_anchor_time`] sums, summed the same plain
                    // sequential way — see that function's note on why the reference's
                    // pairwise/sequential split is not replicated anywhere in this crate.
                    let span: f64 = (0..geometry.num_latent_frames)
                        .map(|i| {
                            ROPE_FRAME_RESCALE
                                * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()]
                        })
                        .sum();
                    advance = span;
                }
            }
            w
        }
    };

    if geometry.num_audio_latents > 0 {
        // `audio_position_ids` takes its origin as a token count; the shared clock is a float, so
        // the rows are built at origin 0 and shifted. Equivalent by construction and it keeps one
        // implementation of the channel-major / width-extreme convention.
        let rows = audio_position_ids(0, geometry.num_audio_latents, audio_channels, &width_grid)?;
        audio_rows.extend(rows.into_iter().map(|[t, h, w]| [t + origin, h, w]));
        advance = advance.max(geometry.num_audio_latents as f64);
    } else if geometry.kind == ReferenceKind::Audio {
        return Err(CandleError::Msg(
            "minimax-h3 dit positions: an audio reference occupies no audio latents".into(),
        ));
    }

    Ok((audio_rows, video_rows, advance))
}

/// Target video rows, frame-major: each latent frame contributes the whole [`frame_grid`] at that
/// frame's [`temporal_grid`] time.
pub fn video_position_ids(
    num_text_tokens: usize,
    num_latent_frames: usize,
    frame_rows: &[[f64; 2]],
) -> Result<Vec<[f64; 3]>> {
    if frame_rows.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 dit positions: the frame grid is empty".into(),
        ));
    }
    let times = temporal_grid(num_latent_frames, num_text_tokens as f64)?;
    let mut out = Vec::with_capacity(times.len() * frame_rows.len());
    for &t in &times {
        for &[h, w] in frame_rows {
            out.push([t, h, w]);
        }
    }
    Ok(out)
}

/// Stack rows into the `[seq_len, 3]` tensor [`super::rope::MmRope::tables`] consumes.
///
/// **Always float32**, deliberately, and not caller-selectable. The reference builds the grid in
/// float64 and narrows exactly once, on entry to the model (`position_ids.to(torch.float32)`);
/// letting a caller pick bf16 here would quantize the coordinates — the spatial grid spans `[0,32)`
/// and the temporal one reaches ~170 at 15 s, where bf16's 8-bit significand cannot even represent
/// adjacent latent frames — and `MmRope::tables` would then silently widen the damaged values back
/// to f32, leaving no trace.
pub fn to_tensor(rows: &[[f64; 3]], device: &Device) -> Result<Tensor> {
    if rows.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 dit positions: cannot build an empty position grid".into(),
        ));
    }
    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    Ok(Tensor::from_vec(flat, (rows.len(), 3), device)?.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `endpoint=False` linspace, aspect-normalized. A square canvas spans `[0, 32)` exactly.
    #[test]
    fn a_square_canvas_spans_zero_to_thirty_two() {
        let g = spatial_axis_grid(4, 2, 4.0).unwrap();
        // ratio = 1, left = 0, num = 2, step = 32/2.
        assert_eq!(g.len(), 2);
        assert!((g[0] - 0.0).abs() < 1e-12);
        assert!((g[1] - 16.0).abs() < 1e-12);

        // ...and the endpoint is excluded: with 4 points the last is 24, not 32.
        let g = spatial_axis_grid(8, 2, 8.0).unwrap();
        assert_eq!(g.len(), 4);
        assert!((g[3] - 24.0).abs() < 1e-12);
    }

    /// A non-square canvas centres the SHORT axis and widens the long one, both about 16.
    #[test]
    fn a_wide_canvas_is_aspect_normalized_about_the_centre() {
        // 4 high, 9 wide -> sqrt_area = 6.
        let sqrt_area = ((4 * 9) as f64).sqrt();
        let h = spatial_axis_grid(4, 2, sqrt_area).unwrap();
        let w = spatial_axis_grid(9, 3, sqrt_area).unwrap();
        // h ratio 4/6 < 1 -> inset; w ratio 9/6 > 1 -> overhangs both ends.
        assert!(h[0] > 0.0, "the short axis is inset, got {}", h[0]);
        assert!(w[0] < 0.0, "the long axis overhangs, got {}", w[0]);
        // Both are centred on 16 (= 0.5 · 32).
        let mid = |g: &[f64], ratio: f64| g[0] + ratio * ROPE_SPATIAL_SCALE / 2.0;
        assert!((mid(&h, 4.0 / 6.0) - 16.0).abs() < 1e-9);
        assert!((mid(&w, 9.0 / 6.0) - 16.0).abs() < 1e-9);
    }

    /// The `5/3 × (1, 4, 4, 4, 4)` cycle. A uniform ramp is the plausible wrong answer, so this
    /// pins the actual gaps.
    #[test]
    fn temporal_spacing_is_non_uniform() {
        let t = temporal_grid(7, 10.0).unwrap();
        assert_eq!(t.len(), 7);
        assert!((t[0] - 10.0).abs() < 1e-12);
        // The gap leaving index `i` is 5/3 · FRAMES[i % 5]: 5/3 out of index 0, 20/3 out of 1..4.
        assert!((t[1] - t[0] - 5.0 / 3.0).abs() < 1e-12);
        assert!((t[2] - t[1] - 20.0 / 3.0).abs() < 1e-12);
        assert!((t[5] - t[4] - 20.0 / 3.0).abs() < 1e-12);
        // ...and the cycle restarts on the gap LEAVING index 5, not on the one arriving at it.
        assert!(
            (t[6] - t[5] - 5.0 / 3.0).abs() < 1e-12,
            "the 5-entry cycle must restart leaving index 5"
        );
        assert!(
            (t[2] - t[1] - (t[1] - t[0])).abs() > 1.0,
            "spacing must NOT be uniform"
        );
        assert!(temporal_grid(0, 0.0).is_err());
    }

    /// **The audio convention**, pinned field by field.
    #[test]
    fn audio_rows_are_channel_major_have_no_height_and_sit_at_the_width_extremes() {
        let width = vec![-4.0, 0.0, 4.0, 8.0];
        let rows = audio_position_ids(7, 3, 2, &width).unwrap();
        assert_eq!(rows.len(), 6, "3 latents × 2 channels");

        // t: text length offsets the clock, one unit per latent, TILED per channel.
        let times: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        assert_eq!(times, vec![7.0, 8.0, 9.0, 7.0, 8.0, 9.0]);

        // h: audio has none.
        assert!(
            rows.iter().all(|r| r[1] == 0.0),
            "audio rows carry no height"
        );

        // w: channel 0 at the left extreme, channel 1 at the right — NOT the interior points.
        let widths: Vec<f64> = rows.iter().map(|r| r[2]).collect();
        assert_eq!(widths, vec![-4.0, -4.0, -4.0, 8.0, 8.0, 8.0]);

        assert!(audio_position_ids(7, 3, 2, &[]).is_err());
        assert!(audio_position_ids(7, 0, 2, &width).is_err());
        assert!(audio_position_ids(7, 3, 0, &width).is_err());
    }

    /// Text rows occupy the head of the shared time axis; media rows continue from there. The
    /// offset is what makes prompt length change the video's rotary times.
    #[test]
    fn text_length_shifts_the_media_clock() {
        let text = text_position_ids(4).unwrap();
        assert_eq!(text.len(), 4);
        assert_eq!(text[3], [3.0, 0.0, 0.0]);

        let (grid, width) = frame_grid(2, 2, 2, 2).unwrap();
        let a = video_position_ids(4, 2, &grid).unwrap();
        let b = video_position_ids(9, 2, &grid).unwrap();
        assert!((a[0][0] - 4.0).abs() < 1e-12);
        assert!((b[0][0] - 9.0).abs() < 1e-12);

        // Audio shifts by the same offset.
        let audio_a = audio_position_ids(4, 2, 2, &width).unwrap();
        let audio_b = audio_position_ids(9, 2, 2, &width).unwrap();
        assert!((audio_b[0][0] - audio_a[0][0] - 5.0).abs() < 1e-12);
    }

    /// Video rows are frame-major: the whole spatial grid at frame 0's time, then frame 1's.
    #[test]
    fn video_rows_are_frame_major() {
        let (grid, _) = frame_grid(4, 4, 2, 2).unwrap();
        assert_eq!(grid.len(), 4, "2×2 patched rows per frame");
        let rows = video_position_ids(0, 3, &grid).unwrap();
        assert_eq!(rows.len(), 12);
        // All four rows of frame 0 share a time...
        assert!(rows[0..4].iter().all(|r| r[0] == rows[0][0]));
        // ...and their (h, w) walk the grid in ij order.
        assert_eq!([rows[0][1], rows[0][2]], grid[0]);
        assert_eq!([rows[3][1], rows[3][2]], grid[3]);
        // ...while frame 1 is 5/3 later with the same spatial grid.
        assert!((rows[4][0] - rows[0][0] - 5.0 / 3.0).abs() < 1e-12);
        assert_eq!([rows[4][1], rows[4][2]], grid[0]);
        assert!(video_position_ids(0, 3, &[]).is_err());
    }

    #[test]
    fn to_tensor_shapes_and_rejects_empty() {
        let dev = Device::Cpu;
        let a = to_tensor(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev).unwrap();
        assert_eq!(a.dims(), &[2, 3]);
        assert_eq!(
            a.dtype(),
            DType::F32,
            "the grid is never narrowed below f32"
        );
        assert_eq!(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(to_tensor(&[], &dev).is_err());
    }

    /// A `fl2va` keyframe block is one frame's rows at ONE constant time, with the full spatial
    /// grid — not a slice of the video's temporal ramp.
    #[test]
    fn a_keyframe_block_is_the_whole_frame_grid_at_one_time() {
        let (grid, _) = frame_grid(4, 4, 2, 2).unwrap();
        let rows = keyframe_position_ids(KeyframeAnchor::First, 5, 3, &grid).unwrap();
        assert_eq!(rows.len(), grid.len(), "one row per patched position");
        assert!(rows.iter().all(|r| r[0] == rows[0][0]), "one constant time");
        assert!(
            (rows[0][0] - 5.0).abs() < 1e-12,
            "`first` sits at the clock origin"
        );
        for (row, cell) in rows.iter().zip(&grid) {
            assert_eq!([row[1], row[2]], *cell, "the full frame grid, unchanged");
        }

        // `last` is the final FRAME's time, `origin + 5/3·(total_frames - 1)` — NOT the final
        // latent's time, which is 5 rotary units earlier whenever the last latent spans 4 frames.
        let (origin, latents) = (5.0f64, 3usize);
        let frames: f64 = (0..latents)
            .map(|i| ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()])
            .sum();
        let last = keyframe_anchor_time(KeyframeAnchor::Last, 5, latents).unwrap();
        assert!(
            (last - (origin + ROPE_FRAME_RESCALE * (frames - 1.0))).abs() < 1e-12,
            "`last` must be the final FRAME's rotary time, got {last}"
        );
        let times = temporal_grid(latents, origin).unwrap();
        assert!(
            (last - times[2]).abs() > 1.0,
            "`last` must NOT collapse onto the final latent's own time — that is the plausible \
             misreading, and it agrees only when (num_latent_frames - 1) % 5 == 0"
        );
        // ...and the two anchors are different, or the enum would be inert.
        assert!(last > keyframe_anchor_time(KeyframeAnchor::First, 5, 3).unwrap());
        assert!(keyframe_position_ids(KeyframeAnchor::Last, 5, 0, &grid).is_err());
        assert!(keyframe_position_ids(KeyframeAnchor::Last, 5, 3, &[]).is_err());
    }

    /// The reference computes the `Last` anchor with numpy's **pairwise** summation and the
    /// `ref2va` path sums the same series **sequentially**, differing in the last ulp of f64 from
    /// 16 latent frames on. This port sums sequentially; that is only sound because the
    /// distinction cannot survive the f64 → f32 narrow the model performs on entry.
    ///
    /// Pin it rather than assert it in prose: a pairwise sum is reproduced here and required to be
    /// **bit-identical in f32** across every latent-frame count a legal render can produce (17n+5
    /// frames ⇒ 5k+2 latents, so 37 … 102) and well beyond.
    #[test]
    fn keyframe_anchor_summation_order_is_invisible_after_the_f32_narrow() {
        /// numpy's `pairwise_sum` structure: a plain sum up to 8, an 8-accumulator unrolled
        /// pass up to 128, and a split at `(n/2) & !7` above that.
        fn pairwise(v: &[f64]) -> f64 {
            let n = v.len();
            if n <= 8 {
                return v.iter().sum();
            }
            if n <= 128 {
                let mut acc = [0.0f64; 8];
                acc.copy_from_slice(&v[..8]);
                let mut i = 8;
                while i + 8 <= n {
                    for (a, x) in acc.iter_mut().zip(&v[i..i + 8]) {
                        *a += x;
                    }
                    i += 8;
                }
                let mut total = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
                    + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                for x in &v[i..] {
                    total += x;
                }
                return total;
            }
            // `n > 128` makes `half >= 64`, so both halves are non-empty and the recursion ends.
            let half = (n / 2) & !7;
            pairwise(&v[..half]) + pairwise(&v[half..])
        }

        let mut worst = 0.0f64;
        for n in 1..=400usize {
            let spans: Vec<f64> = (0..n)
                .map(|i| {
                    ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()]
                })
                .collect();
            let sequential: f64 = spans.iter().sum();
            let paired = pairwise(&spans);
            worst = worst.max((sequential - paired).abs());
            assert_eq!(
                sequential as f32, paired as f32,
                "at {n} latent frames the two summation orders survive the f32 narrow; this port \
                 would then have to reproduce numpy's pairwise order per call site"
            );
        }
        assert!(worst < 1e-9, "f64 divergence grew to {worst:.3e}");
        println!(
            "  max f64 |pairwise - sequential| over 1..=400 latents: {worst:.3e}; f32 delta 0"
        );
    }

    #[test]
    fn a_latent_extent_that_is_not_a_patch_multiple_is_rejected() {
        assert!(spatial_axis_grid(5, 2, 5.0).is_err());
        assert!(spatial_axis_grid(0, 2, 5.0).is_err());
        assert!(spatial_axis_grid(4, 0, 5.0).is_err());
        assert!(spatial_axis_grid(4, 2, 0.0).is_err());
        assert!(frame_grid(4, 5, 2, 2).is_err());
    }
}
