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
//! # Scope
//!
//! These are the **coordinate grids**, i.e. the rotary's input contract. Assembling the packed
//! sequence itself — the row order `[text | keyframe conditions | target audio | target video]`,
//! the three index tensors, the modality tags and the scatter — belongs to the joint denoise loop
//! (sc-17146) and the pipeline (sc-17147). The grids are here because the rotary cannot be pinned
//! without them.
//!
//! # The three conventions
//!
//! | rows | `t` | `h` | `w` |
//! |---|---|---|---|
//! | text | `0 … num_text_tokens-1` | 0 | 0 |
//! | audio | `num_text_tokens + 0 … num_audio_latents-1`, **tiled per channel** | **0** | `width_grid[0]` for channel 0, `width_grid[-1]` for the rest |
//! | video | [`temporal_grid`] from `num_text_tokens` | [`frame_grid`] | [`frame_grid`] |
//!
//! Two consequences that are easy to miss:
//!
//! * **Text length shifts the whole media clock.** Media rows start their time axis at
//!   `num_text_tokens`, so the same video at two prompt lengths gets different rotary times.
//! * **Audio is channel-major and pinned to the two extremes of the video's width grid.** It rides
//!   the video's clock at one unit per latent — 40 audio latents/s against 24 fps × 5/3 — and its
//!   only spatial identity is which end of the canvas it sits at.

use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

/// Rotary time per *frame* of a latent, before the per-latent frame counts:
/// `24 fps / 40 audio-latents-per-second`. `_ROPE_FRAME_RESCALE`.
pub const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Frames each latent frame stands for, cycling. `_ROPE_FRAMES_PER_LATENT` — the first latent
/// covers 1 frame and every later one covers 4, which is the video VAE's `17n + 5` geometry.
pub const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];

/// The spatial grids are aspect-normalized onto the unit interval and then scaled by this.
/// `_ROPE_SPATIAL_SCALE`, so a square canvas spans `[0, 32)`.
pub const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// Channels the soundtrack is packed channel-major over. Stereo.
pub const AUDIO_CHANNELS: i32 = 2;

/// One aspect-normalized spatial axis: `dim / patch` coordinates centred on the unit interval and
/// scaled by [`ROPE_SPATIAL_SCALE`], right endpoint **excluded**.
///
/// `sqrt_area` is `sqrt(latent_height · latent_width)`, shared by both axes, which is what makes
/// the grid aspect-normalized: a 16:9 canvas gets a wider `w` span than `h`.
///
/// The reference builds this with `np.linspace(..., endpoint=False)`, whose step is
/// `(stop - start) / num` — **not** `torch.linspace`'s `/(num - 1)`. Reproduced here deliberately:
/// the two differ at every point except the first.
pub fn spatial_axis_grid(dim: i32, patch: i32, sqrt_area: f64) -> Result<Vec<f64>> {
    if dim <= 0 || patch <= 0 || dim % patch != 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 dit positions: latent extent {dim} must be a positive multiple of patch \
             {patch}"
        )));
    }
    if !sqrt_area.is_finite() || sqrt_area <= 0.0 {
        return Err(Error::Msg(format!(
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
    latent_height: i32,
    latent_width: i32,
    patch_h: i32,
    patch_w: i32,
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
pub fn temporal_grid(num_latent_frames: i32, origin: f64) -> Result<Vec<f64>> {
    if num_latent_frames <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 dit positions: num_latent_frames must be positive, got {num_latent_frames}"
        )));
    }
    let mut out = Vec::with_capacity(num_latent_frames as usize);
    let mut acc = origin;
    for index in 0..num_latent_frames {
        out.push(acc);
        acc += ROPE_FRAME_RESCALE
            * ROPE_FRAMES_PER_LATENT[(index as usize) % ROPE_FRAMES_PER_LATENT.len()];
    }
    Ok(out)
}

/// Text rows: `t` is the row index, `h` and `w` are zero.
///
/// The time axis is shared with the media rows, so the prompt occupies `[0, num_text_tokens)` of it
/// and every media row is offset by `num_text_tokens`.
pub fn text_position_ids(num_text_tokens: i32) -> Result<Vec<[f64; 3]>> {
    if num_text_tokens < 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 dit positions: num_text_tokens must be non-negative, got {num_text_tokens}"
        )));
    }
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
    num_text_tokens: i32,
    num_audio_latents: i32,
    audio_channels: i32,
    width_grid: &[f64],
) -> Result<Vec<[f64; 3]>> {
    if num_audio_latents <= 0 || audio_channels <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 dit positions: audio latents {num_audio_latents} and channels \
             {audio_channels} must be positive"
        )));
    }
    let (&first, &last) = match (width_grid.first(), width_grid.last()) {
        (Some(f), Some(l)) => (f, l),
        _ => {
            return Err(Error::Msg(
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

/// Target video rows, frame-major: each latent frame contributes the whole [`frame_grid`] at that
/// frame's [`temporal_grid`] time.
pub fn video_position_ids(
    num_text_tokens: i32,
    num_latent_frames: i32,
    frame_rows: &[[f64; 2]],
) -> Result<Vec<[f64; 3]>> {
    if frame_rows.is_empty() {
        return Err(Error::Msg(
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

/// Stack rows into the `[seq_len, 3]` array [`super::rope::MmRope::tables`] consumes.
///
/// The reference builds the grid in float64 and the model casts to float32 on entry, so the
/// coordinates are computed in `f64` here and narrowed once, at the boundary.
pub fn to_array(rows: &[[f64; 3]], dtype: Dtype) -> Result<Array> {
    if rows.is_empty() {
        return Err(Error::Msg(
            "minimax-h3 dit positions: cannot build an empty position grid".into(),
        ));
    }
    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    Ok(Array::from_slice(&flat, &[rows.len() as i32, 3]).as_dtype(dtype)?)
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
    }

    #[test]
    fn to_array_shapes_and_rejects_empty() {
        let a = to_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], Dtype::Float32).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(to_array(&[], Dtype::Float32).is_err());
    }

    #[test]
    fn a_latent_extent_that_is_not_a_patch_multiple_is_rejected() {
        assert!(spatial_axis_grid(5, 2, 5.0).is_err());
        assert!(spatial_axis_grid(0, 2, 5.0).is_err());
        assert!(frame_grid(4, 5, 2, 2).is_err());
    }
}
