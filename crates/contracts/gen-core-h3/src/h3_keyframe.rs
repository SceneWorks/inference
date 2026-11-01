//! **Keyframe geometry** for `fl2va` (sc-17148) — resolving the canvas a keyframe implies and
//! fitting the keyframes onto it. Backend-neutral and shared by `mlx-gen-minimax-h3` and
//! `candle-gen-minimax-h3` (hoisted from their mirrored `keyframe.rs` copies, sc-27256); each
//! engine re-exports it and adds the anchor shapes and the tensor-bound VAE pixel normalization.
//!
//! # A keyframe BINDS the canvas; a reference does not
//!
//! This is the fact that most distinguishes `fl2va` from `ref2va`. With a keyframe and no explicit
//! size, `height`/`width` come from **the first keyframe's aspect ratio** via
//! [`resolve_canvas_size`]; with no keyframe at all the model falls back to 16:9. A `ref2va`
//! reference image does *not* bind geometry and always defaults to 16:9. So the same image passed
//! as a keyframe and as a reference produces two different canvases, and there is no
//! `aspect_ratio` parameter anywhere to reconcile them with.
//!
//! # The first keyframe is STRETCHED; a second one is COVER-CROPPED
//!
//! [`KeyframeFit`] is not a style choice, it is positional. The geometry anchor — `keyframes[0]`,
//! which is `last_image` when only a last frame was passed — is resized straight onto the canvas
//! with **no aspect preservation**: the reference calls PIL's plain `resize((W, H), LANCZOS)`.
//! Letterboxing it instead would put bars into the conditioning signal.
//!
//! Every *later* keyframe is cover-cropped, because the canvas is already fixed by then and
//! stretching a second image with a different aspect would distort the two anchors inconsistently.
//!
//! # Rounding is banker's rounding
//!
//! [`resolve_canvas_size`] and the cover-crop arithmetic both go through Python's `round`, which is
//! **half-to-even**. Rust's `f64::round` is half-away-from-zero, and the two disagree on exactly
//! the inputs a 32-multiple canvas keeps landing on. [`round_half_to_even`] is the reference's.
//!
//! # What is NOT bit-exact, and where the parity boundary is drawn
//!
//! The resampling filter is the one place this port cannot claim exactness. The reference uses
//! **PIL's** Lanczos; this crate uses the `image` crate's `Lanczos3`. Both are a=3 windowed-sinc
//! with support scaled on downscale, and they agree closely, but they are not the same code and
//! nothing here claims they produce identical bytes.
//!
//! The parity boundary is therefore drawn **after** the resize: `tests/video_vae_encode_parity.rs`
//! gates the VAE encode on already-sized pixels, and the tests in this module gate the *policy* —
//! which canvas is chosen, which keyframe is stretched, which is cropped, and how the pixels are
//! normalized (the last in each engine). Conflating the two would let a resampling difference of a few LSBs read as a
//! conditioning defect, or worse, hide one.

use image::imageops::FilterType;
use image::RgbImage;

use crate::h3_memory::{CANVAS_MAX_PIXELS, CANVAS_SHORT_EDGE, SPATIAL_STRIDE};
use gen_core::{Error, Image, Result};

/// Narrowest aspect ratio MiniMax-H3 accepts — 1:4.
pub const MIN_ASPECT_RATIO: f64 = 1.0 / 4.0;

/// Widest aspect ratio MiniMax-H3 accepts — 4:1.
pub const MAX_ASPECT_RATIO: f64 = 4.0;

/// Python's `round`: half-to-even, **not** half-away-from-zero.
///
/// `round(2.5)` is `2`, not `3`. Rust's `f64::round` is the other convention, and the difference
/// bites exactly where a canvas divides evenly into half a stride — which, on a 32-multiple
/// lattice, is a case the arithmetic reaches constantly rather than rarely.
pub fn round_half_to_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// Resolve the canvas a keyframe of ratio `aspect_width : aspect_height` implies.
///
/// Returns `(height, width)`, matching the reference's return order. The algorithm:
///
/// 1. reject a ratio outside `1:4 … 4:1` — **rejected, not clamped**;
/// 2. lay the short edge at `short_edge` and derive the long one from the ratio;
/// 3. if the area exceeds `max_pixels`, scale both edges by `sqrt(max_pixels / area)`;
/// 4. round **each axis independently** to a multiple of `canvas_multiple`, floored at one
///    multiple.
///
/// Step 4 happens *after* the area cap, so the final area can land slightly **above** the
/// pre-rounding budget and the delivered ratio drifts from the requested one — 4:1 resolves to
/// 512x2016, which is 3.938:1. That is the reference's behaviour and not a rounding bug to fix.
pub fn resolve_canvas_size(
    aspect_width: f64,
    aspect_height: f64,
    canvas_multiple: i32,
    short_edge: i32,
    max_pixels: i64,
) -> Result<(i32, i32)> {
    if aspect_width <= 0.0
        || aspect_height <= 0.0
        || !aspect_width.is_finite()
        || !aspect_height.is_finite()
    {
        return Err(Error::Msg(format!(
            "minimax-h3 canvas: the aspect ratio must be positive, got \
             {aspect_width}:{aspect_height}"
        )));
    }
    if canvas_multiple <= 0 || short_edge <= 0 || max_pixels <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 canvas: multiple/short_edge/max_pixels must be positive, got \
             {canvas_multiple}/{short_edge}/{max_pixels}"
        )));
    }
    let ratio = aspect_width / aspect_height;
    if !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&ratio) {
        return Err(Error::Msg(format!(
            "minimax-h3 canvas: MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got \
             {aspect_width}:{aspect_height} ({ratio})"
        )));
    }
    let short = f64::from(short_edge);
    let (mut width, mut height) = if ratio >= 1.0 {
        (short * ratio, short)
    } else {
        (short, short / ratio)
    };
    let area = width * height;
    let budget = max_pixels as f64;
    if area > budget {
        let scale = (budget / area).sqrt();
        width *= scale;
        height *= scale;
    }
    let m = f64::from(canvas_multiple);
    let snap = |v: f64| -> i32 { (round_half_to_even(v / m) * m).max(m) as i32 };
    Ok((snap(height), snap(width)))
}

/// [`resolve_canvas_size`] at the shipped MiniMax-H3 constants ([`SPATIAL_STRIDE`],
/// [`CANVAS_SHORT_EDGE`], [`CANVAS_MAX_PIXELS`]) — stride 32, short edge 768, area
/// capped at `768 · 1344`.
pub fn resolve_keyframe_canvas(width: u32, height: u32) -> Result<(u32, u32)> {
    let (h, w) = resolve_canvas_size(
        f64::from(width),
        f64::from(height),
        SPATIAL_STRIDE as i32,
        CANVAS_SHORT_EDGE as i32,
        i64::from(CANVAS_MAX_PIXELS),
    )?;
    Ok((w as u32, h as u32))
}

/// How one keyframe is fitted onto an already-resolved canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeFit {
    /// **The geometry anchor** — resized straight onto the canvas with no aspect preservation.
    /// The canvas came from this image's own ratio, so the stretch is usually a no-op or close to
    /// it; it is a genuine distortion only when the caller supplied an explicit size.
    Stretch,
    /// A follower — scaled to cover the canvas, then centre-cropped. Used for every keyframe after
    /// the first, whose aspect the canvas was not derived from.
    CoverCrop,
}

impl KeyframeFit {
    /// The fit for the keyframe at `index` in packed order. Index 0 stretches; the rest crop.
    pub fn for_index(index: usize) -> Self {
        if index == 0 {
            Self::Stretch
        } else {
            Self::CoverCrop
        }
    }
}

/// Fit one keyframe onto a `width x height` canvas.
///
/// See the module docs for the filter caveat: this is the `image` crate's `Lanczos3`, not PIL's.
pub fn fit_keyframe(image: &Image, width: u32, height: u32, fit: KeyframeFit) -> Result<Image> {
    if width == 0 || height == 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 keyframe: canvas must be positive, got {width}x{height}"
        )));
    }
    if image.width == 0 || image.height == 0 {
        return Err(Error::Msg(
            "minimax-h3 keyframe: a keyframe must have a positive extent".into(),
        ));
    }
    let expected = image.width as usize * image.height as usize * 3;
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "minimax-h3 keyframe: a {}x{} RGB8 image is {expected} bytes, got {}",
            image.width,
            image.height,
            image.pixels.len()
        )));
    }
    if image.width == width && image.height == height {
        return Ok(image.clone());
    }
    let rgb =
        RgbImage::from_raw(image.width, image.height, image.pixels.clone()).ok_or_else(|| {
            Error::Msg("minimax-h3 keyframe: the pixel buffer is not a valid RGB8 image".into())
        })?;

    let fitted = match fit {
        KeyframeFit::Stretch => image::imageops::resize(&rgb, width, height, FilterType::Lanczos3),
        KeyframeFit::CoverCrop => {
            // MiniMax's own arithmetic: `round()` sizing (not floor) and `(src - dst) // 2`
            // centring (not `dst//2 - src//2`). `VaeImageProcessor`'s own `resize_mode="crop"`
            // disagrees with it on roughly half of sampled aspect ratios, so it must not be
            // substituted.
            let scale = (f64::from(width) / f64::from(image.width))
                .max(f64::from(height) / f64::from(image.height));
            let rw = (round_half_to_even(f64::from(image.width) * scale) as u32).max(width);
            let rh = (round_half_to_even(f64::from(image.height) * scale) as u32).max(height);
            let resized = image::imageops::resize(&rgb, rw, rh, FilterType::Lanczos3);
            let left = rw.saturating_sub(width) / 2;
            let top = rh.saturating_sub(height) / 2;
            image::imageops::crop_imm(&resized, left, top, width, height).to_image()
        }
    };
    Ok(Image {
        width,
        height,
        pixels: fitted.into_raw(),
    })
}

/// Fit every keyframe onto the canvas in packed order — index 0 stretched, the rest cover-cropped.
pub fn fit_keyframes(images: &[&Image], width: u32, height: u32) -> Result<Vec<Image>> {
    images
        .iter()
        .enumerate()
        .map(|(i, img)| fit_keyframe(img, width, height, KeyframeFit::for_index(i)))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The keyframe index contract (sc-26691)
// ---------------------------------------------------------------------------------------------

/// Which end of the generated video a `fl2va` keyframe conditioning block is anchored to.
///
/// Defined here, in the shared contract, so the candle and MLX engines cannot drift on what a
/// request's `frame_idx` resolves to; each engine re-exports it from `dit::positions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeAnchor {
    /// Anchored at the first latent frame — the block sits at the video clock's origin.
    First,
    /// Anchored at the last latent frame.
    Last,
}

/// A `Conditioning::Keyframe::frame_idx` that MiniMax-H3 refuses, from [`resolve_keyframe_anchor`]
/// and [`resolve_keyframe_slots`].
///
/// Typed so a caller (and a test) can tell *why* an index was refused; the engines flatten it into
/// their own error with the model id prefixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeIndexError {
    /// The clip has no frames, so no index can name either end.
    EmptyClip,
    /// A negative index other than `-1`. Only `-1` is a "from the end" alias; `-2` is not "the
    /// second-to-last frame" here (that would be an interior frame anyway) and is never clamped.
    Negative { frame_idx: i32 },
    /// An index at or past `num_frames`. Never clamped onto the last frame.
    OutOfRange { frame_idx: i32, num_frames: usize },
    /// An in-range index that is neither end. `fl2va` has exactly two anchor slots and no
    /// mid-clip conditioning row to occupy.
    Interior { frame_idx: i32, num_frames: usize },
    /// Two keyframes resolve to the same end of the clip.
    DuplicateAnchor {
        anchor: KeyframeAnchor,
        frame_idx: i32,
    },
}

impl std::fmt::Display for KeyframeIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyClip => write!(
                f,
                "a keyframe was supplied for a clip of 0 frames; there is no first or last frame \
                 to anchor"
            ),
            Self::Negative { frame_idx } => write!(
                f,
                "keyframe index {frame_idx} is negative; the only negative index accepted is -1 \
                 (the LAST frame)"
            ),
            Self::OutOfRange {
                frame_idx,
                num_frames,
            } => write!(
                f,
                "keyframe index {frame_idx} is out of range for a {num_frames}-frame clip; a \
                 keyframe must anchor the FIRST frame (0) or the LAST ({} or -1)",
                num_frames - 1
            ),
            Self::Interior {
                frame_idx,
                num_frames,
            } => write!(
                f,
                "keyframe index {frame_idx} is an interior frame, but fl2va has exactly two anchor \
                 slots — the FIRST frame (0) and the LAST ({} or -1); there is no mid-clip \
                 conditioning row to occupy",
                num_frames - 1
            ),
            Self::DuplicateAnchor { anchor, frame_idx } => write!(
                f,
                "two keyframes anchor the same end of the clip ({anchor:?}, index {frame_idx}); \
                 fl2va's two slots are the FIRST and the LAST frame"
            ),
        }
    }
}

impl std::error::Error for KeyframeIndexError {}

impl From<KeyframeIndexError> for Error {
    fn from(e: KeyframeIndexError) -> Self {
        Error::Msg(format!("minimax-h3 keyframe: {e}"))
    }
}

/// Resolve one keyframe `frame_idx` against a clip of `num_frames` frames — **the** MiniMax-H3
/// keyframe index contract, shared by both engines.
///
/// * `0` is the FIRST frame (including on a 1-frame clip, where it is also the last index);
/// * `num_frames - 1` **or** `-1` is the LAST frame — `-1` so a caller that knows only "the end of
///   the clip" need not resolve the frame count first;
/// * every other index is refused with a typed [`KeyframeIndexError`]: a negative index other than
///   `-1`, an index at or past `num_frames`, or an interior index.
///
/// Nothing is clamped. sc-26691: the candle engine used to resolve `frame_idx.max(0)`, so `-1`
/// silently became the FIRST frame while MLX read it as the LAST — a first+last request rendered
/// both anchors at the start on CUDA, and a last-only request rendered the last image at the start.
pub fn resolve_keyframe_anchor(
    frame_idx: i32,
    num_frames: usize,
) -> std::result::Result<KeyframeAnchor, KeyframeIndexError> {
    if num_frames == 0 {
        return Err(KeyframeIndexError::EmptyClip);
    }
    if frame_idx == 0 {
        return Ok(KeyframeAnchor::First);
    }
    if frame_idx == -1 {
        return Ok(KeyframeAnchor::Last);
    }
    if frame_idx < 0 {
        return Err(KeyframeIndexError::Negative { frame_idx });
    }
    let idx = frame_idx as usize; // non-negative i32 always fits usize on supported targets
    if idx >= num_frames {
        Err(KeyframeIndexError::OutOfRange {
            frame_idx,
            num_frames,
        })
    } else if idx == num_frames - 1 {
        Ok(KeyframeAnchor::Last)
    } else {
        Err(KeyframeIndexError::Interior {
            frame_idx,
            num_frames,
        })
    }
}

/// A request's keyframes resolved onto `fl2va`'s two anchor slots, each holding the position of
/// the keyframe in the **request** that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyframeSlots {
    /// Request position of the keyframe anchoring the FIRST frame.
    pub first: Option<usize>,
    /// Request position of the keyframe anchoring the LAST frame.
    pub last: Option<usize>,
}

impl KeyframeSlots {
    /// The filled anchors in packed order — first, then last — matching the reference's fixed
    /// `(("first", image), ("last", last_image))` pair filtered by presence.
    pub fn anchors(&self) -> Vec<KeyframeAnchor> {
        let mut v = Vec::with_capacity(2);
        if self.first.is_some() {
            v.push(KeyframeAnchor::First);
        }
        if self.last.is_some() {
            v.push(KeyframeAnchor::Last);
        }
        v
    }

    /// Request positions in packed order, positional with [`Self::anchors`]: `order()[i]` is the
    /// request keyframe that `anchors()[i]` describes. The caller may supply keyframes in any
    /// order; both engines pack them first-then-last so the geometry anchor (`packed[0]`) is the
    /// same image on either backend.
    pub fn order(&self) -> Vec<usize> {
        self.first.into_iter().chain(self.last).collect()
    }

    /// Whether no keyframe was supplied (`t2va`).
    pub fn is_empty(&self) -> bool {
        self.first.is_none() && self.last.is_none()
    }
}

/// Resolve every keyframe index of a request (in request order) onto the two anchor slots,
/// applying [`resolve_keyframe_anchor`] to each and refusing two keyframes on the same end.
pub fn resolve_keyframe_slots(
    frame_idxs: impl IntoIterator<Item = i32>,
    num_frames: usize,
) -> std::result::Result<KeyframeSlots, KeyframeIndexError> {
    let mut slots = KeyframeSlots::default();
    for (pos, frame_idx) in frame_idxs.into_iter().enumerate() {
        let anchor = resolve_keyframe_anchor(frame_idx, num_frames)?;
        let slot = match anchor {
            KeyframeAnchor::First => &mut slots.first,
            KeyframeAnchor::Last => &mut slots.last,
        };
        if slot.is_some() {
            return Err(KeyframeIndexError::DuplicateAnchor { anchor, frame_idx });
        }
        *slot = Some(pos);
    }
    Ok(slots)
}

#[cfg(test)]
mod keyframe_index_tests {
    use super::*;

    #[test]
    fn minus_one_and_n_minus_one_are_last_zero_is_first() {
        for n in [2usize, 3, 25, 121, 124] {
            assert_eq!(resolve_keyframe_anchor(-1, n), Ok(KeyframeAnchor::Last));
            assert_eq!(
                resolve_keyframe_anchor(n as i32 - 1, n),
                Ok(KeyframeAnchor::Last)
            );
            assert_eq!(resolve_keyframe_anchor(0, n), Ok(KeyframeAnchor::First));
        }
    }

    /// sc-26691 mutation guard: restore `frame_idx.max(0)` clamping and these turn green-to-red.
    #[test]
    fn other_negatives_and_out_of_range_are_refused_not_clamped() {
        for n in [2usize, 3, 124] {
            for bad in [-2, -3, i32::MIN] {
                assert_eq!(
                    resolve_keyframe_anchor(bad, n),
                    Err(KeyframeIndexError::Negative { frame_idx: bad })
                );
            }
            for bad in [n as i32, n as i32 + 1, i32::MAX] {
                assert_eq!(
                    resolve_keyframe_anchor(bad, n),
                    Err(KeyframeIndexError::OutOfRange {
                        frame_idx: bad,
                        num_frames: n
                    })
                );
            }
        }
    }

    #[test]
    fn interior_indices_are_refused() {
        assert_eq!(
            resolve_keyframe_anchor(60, 124),
            Err(KeyframeIndexError::Interior {
                frame_idx: 60,
                num_frames: 124
            })
        );
        assert_eq!(
            resolve_keyframe_anchor(1, 3),
            Err(KeyframeIndexError::Interior {
                frame_idx: 1,
                num_frames: 3
            })
        );
    }

    #[test]
    fn single_frame_and_empty_clips() {
        // On a 1-frame clip index 0 is both ends; it resolves FIRST, and -1 is the LAST slot.
        assert_eq!(resolve_keyframe_anchor(0, 1), Ok(KeyframeAnchor::First));
        assert_eq!(resolve_keyframe_anchor(-1, 1), Ok(KeyframeAnchor::Last));
        assert_eq!(
            resolve_keyframe_anchor(0, 0),
            Err(KeyframeIndexError::EmptyClip)
        );
        assert_eq!(
            resolve_keyframe_anchor(-1, 0),
            Err(KeyframeIndexError::EmptyClip)
        );
    }

    /// The product bug: first + last must land in two DISTINCT slots, whichever spelling of "last"
    /// the caller uses and in whichever order it lists them.
    #[test]
    fn first_and_last_are_distinct_for_several_lengths() {
        for n in [2usize, 3, 25, 49, 121, 124] {
            let last = n as i32 - 1;
            for idxs in [[0, -1], [0, last], [-1, 0], [last, 0]] {
                let slots = resolve_keyframe_slots(idxs, n).unwrap();
                assert_eq!(
                    slots.anchors(),
                    vec![KeyframeAnchor::First, KeyframeAnchor::Last],
                    "n={n} idxs={idxs:?}"
                );
                let first_pos = idxs.iter().position(|&i| i == 0).unwrap();
                assert_eq!(slots.order(), vec![first_pos, 1 - first_pos]);
            }
            // Last-only is LAST, never moved to the start.
            for idx in [-1, last] {
                let slots = resolve_keyframe_slots([idx], n).unwrap();
                assert_eq!(slots.anchors(), vec![KeyframeAnchor::Last]);
                assert_eq!(slots.order(), vec![0]);
            }
        }
    }

    #[test]
    fn slots_refuse_duplicates_and_propagate_index_errors() {
        assert!(resolve_keyframe_slots([], 124).unwrap().is_empty());
        assert_eq!(
            resolve_keyframe_slots([0, 0], 124),
            Err(KeyframeIndexError::DuplicateAnchor {
                anchor: KeyframeAnchor::First,
                frame_idx: 0
            })
        );
        // `-1` and `n - 1` are the same end.
        assert_eq!(
            resolve_keyframe_slots([-1, 123], 124),
            Err(KeyframeIndexError::DuplicateAnchor {
                anchor: KeyframeAnchor::Last,
                frame_idx: 123
            })
        );
        assert_eq!(
            resolve_keyframe_slots([0, -2], 124),
            Err(KeyframeIndexError::Negative { frame_idx: -2 })
        );
        assert_eq!(
            resolve_keyframe_slots([0, 124], 124),
            Err(KeyframeIndexError::OutOfRange {
                frame_idx: 124,
                num_frames: 124
            })
        );
        let msg = Error::from(KeyframeIndexError::Negative { frame_idx: -2 }).to_string();
        assert!(msg.contains("-2") && msg.contains("-1"), "{msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgb: [u8; 3]) -> Image {
        Image {
            width,
            height,
            pixels: (0..width as usize * height as usize)
                .flat_map(|_| rgb)
                .collect(),
        }
    }

    /// **The canvas table, measured off the running reference.** Every row is a
    /// `resolve_canvas_size` result, not a derivation.
    #[test]
    fn the_canvas_table_matches_the_reference() {
        let canvas = |w: f64, h: f64| {
            resolve_canvas_size(
                w,
                h,
                SPATIAL_STRIDE as i32,
                CANVAS_SHORT_EDGE as i32,
                i64::from(CANVAS_MAX_PIXELS),
            )
            .unwrap()
        };
        // (height, width)
        assert_eq!(canvas(16.0, 9.0), (768, 1344));
        assert_eq!(canvas(9.0, 16.0), (1344, 768));
        assert_eq!(canvas(1.0, 1.0), (768, 768));
        assert_eq!(canvas(1920.0, 1080.0), (768, 1344));
        assert_eq!(canvas(1024.0, 768.0), (768, 1024));
        assert_eq!(canvas(4.0, 1.0), (512, 2016));

        // Every result is on the stride-32 lattice, in BOTH axes.
        for (w, h) in [
            (16.0, 9.0),
            (9.0, 16.0),
            (1.0, 1.0),
            (4.0, 1.0),
            (1.0, 4.0),
            (832.0, 1216.0),
            (3.0, 2.0),
            (21.0, 9.0),
        ] {
            let (ch, cw) = canvas(w, h);
            assert_eq!(ch as u32 % SPATIAL_STRIDE, 0, "{w}:{h} height {ch}");
            assert_eq!(cw as u32 % SPATIAL_STRIDE, 0, "{w}:{h} width {cw}");
            assert!(ch > 0 && cw > 0);
        }
    }

    /// Ratios outside 1:4 … 4:1 are **rejected**, not clamped — the same posture
    /// each engine's `pipeline::resolve_geometry` takes for frame counts.
    #[test]
    fn out_of_range_ratios_are_rejected_not_clamped() {
        let canvas = |w: f64, h: f64| {
            resolve_canvas_size(
                w,
                h,
                SPATIAL_STRIDE as i32,
                CANVAS_SHORT_EDGE as i32,
                i64::from(CANVAS_MAX_PIXELS),
            )
        };
        for (w, h) in [(5.0, 1.0), (1.0, 5.0), (100.0, 1.0), (1.0, 100.0)] {
            let e = canvas(w, h).unwrap_err().to_string();
            assert!(
                e.contains("aspect ratio"),
                "{w}:{h} must be rejected, got: {e}"
            );
        }
        // The boundary ratios themselves are legal.
        canvas(4.0, 1.0).unwrap();
        canvas(1.0, 4.0).unwrap();
        // Degenerate inputs are typed errors.
        assert!(canvas(0.0, 1.0).is_err());
        assert!(canvas(1.0, 0.0).is_err());
        assert!(canvas(-1.0, 1.0).is_err());
    }

    /// The area cap is applied **before** rounding, so the delivered ratio drifts and the final
    /// area can exceed the pre-rounding budget. Pinned because it looks like a bug.
    #[test]
    fn rounding_happens_after_the_area_cap() {
        let (h, w) = resolve_canvas_size(
            4.0,
            1.0,
            SPATIAL_STRIDE as i32,
            CANVAS_SHORT_EDGE as i32,
            i64::from(CANVAS_MAX_PIXELS),
        )
        .unwrap();
        assert_eq!((h, w), (512, 2016));
        let delivered = f64::from(w) / f64::from(h);
        assert!(
            (delivered - 3.9375).abs() < 1e-9,
            "4:1 requested delivers {delivered}, not 4.0"
        );
        assert!(
            i64::from(w) * i64::from(h) <= i64::from(CANVAS_MAX_PIXELS),
            "this one happens to land on the budget exactly"
        );
    }

    /// Python's `round` is half-to-even and Rust's is not.
    #[test]
    fn rounding_is_bankers() {
        assert_eq!(round_half_to_even(0.5), 0.0);
        assert_eq!(round_half_to_even(1.5), 2.0);
        assert_eq!(round_half_to_even(2.5), 2.0);
        assert_eq!(round_half_to_even(3.5), 4.0);
        assert_eq!(round_half_to_even(-0.5), 0.0);
        assert_eq!(round_half_to_even(-1.5), -2.0);
        assert_eq!(round_half_to_even(-2.5), -2.0);
        // Non-halves are unaffected, so this cannot silently change ordinary arithmetic.
        for x in [0.4, 0.6, 1.2, -1.2, 7.49, 7.51] {
            assert_eq!(round_half_to_even(x), x.round(), "{x}");
        }
        // …and it really differs from `f64::round` on the halves.
        assert_ne!(round_half_to_even(2.5), 2.5f64.round());
    }

    /// Index 0 stretches, later ones cover-crop — and the two really do differ on a
    /// non-matching aspect.
    #[test]
    fn the_first_keyframe_is_stretched_and_the_rest_are_cropped() {
        assert_eq!(KeyframeFit::for_index(0), KeyframeFit::Stretch);
        assert_eq!(KeyframeFit::for_index(1), KeyframeFit::CoverCrop);
        assert_eq!(KeyframeFit::for_index(7), KeyframeFit::CoverCrop);

        // A 2:1 source onto a 1:1 canvas: stretching squashes the whole width in, cropping keeps
        // the centre. Build a source whose left and right halves differ so the two are
        // distinguishable.
        let mut src = solid(64, 32, [0, 0, 0]);
        for y in 0..32usize {
            for x in 0..64usize {
                let i = (y * 64 + x) * 3;
                let v = if x < 32 { 0u8 } else { 255u8 };
                src.pixels[i] = v;
                src.pixels[i + 1] = v;
                src.pixels[i + 2] = v;
            }
        }
        let stretched = fit_keyframe(&src, 32, 32, KeyframeFit::Stretch).unwrap();
        let cropped = fit_keyframe(&src, 32, 32, KeyframeFit::CoverCrop).unwrap();
        assert_eq!(stretched.width, 32);
        assert_eq!(stretched.height, 32);
        assert_eq!(cropped.width, 32);
        assert_eq!(cropped.height, 32);
        assert_ne!(
            stretched.pixels, cropped.pixels,
            "stretch and cover-crop must not coincide on a 2:1 source"
        );
        // The stretch keeps BOTH halves (dark left, bright right); the crop keeps the seam only.
        let row = 16 * 32 * 3;
        assert!(stretched.pixels[row] < 64, "stretched left edge stays dark");
        assert!(
            stretched.pixels[row + 31 * 3] > 192,
            "stretched right edge stays bright"
        );
    }

    /// A keyframe already at canvas size is passed through untouched — no resampling at all.
    #[test]
    fn an_exact_size_keyframe_is_not_resampled() {
        let src = solid(64, 48, [7, 9, 11]);
        for fit in [KeyframeFit::Stretch, KeyframeFit::CoverCrop] {
            let out = fit_keyframe(&src, 64, 48, fit).unwrap();
            assert_eq!(out.pixels, src.pixels, "{fit:?} must be a no-op at size");
        }
        // Malformed inputs are typed errors.
        assert!(fit_keyframe(&src, 0, 48, KeyframeFit::Stretch).is_err());
        let ragged = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 5],
        };
        assert!(fit_keyframe(&ragged, 8, 8, KeyframeFit::Stretch).is_err());
    }
}
