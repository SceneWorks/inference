//! **Keyframe geometry and pixel preparation** for `fl2va` (sc-17148) — resolving the canvas a
//! keyframe implies, fitting the keyframes onto it, and normalizing them for the video VAE.
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
//! normalized. Conflating the two would let a resampling difference of a few LSBs read as a
//! conditioning defect, or worse, hide one.

//!
//! The canvas resolution and the keyframe fit are backend-neutral and live in
//! [`mlx_gen::gen_core::h3_keyframe`] (sc-27256), shared with the Candle twin; this module
//! re-exports them so `keyframe::*` paths stay stable, and keeps the anchor shapes and the VAE
//! pixel normalization onto this crate's tensors.

pub use mlx_gen::gen_core::h3_keyframe::*;

use mlx_rs::Array;

use mlx_gen::media::Image;
use mlx_gen::{Error, Result};

use crate::dit::positions::KeyframeAnchor;
use crate::pipeline::{PIXEL_MEAN, PIXEL_STD};

/// Which anchors a request supplies, in the packed order the reference builds them in.
///
/// The reference filters a fixed `(("first", image), ("last", last_image))` pair by presence, so
/// only four shapes exist and `last`-only is one of them — **not** a special case bolted onto a
/// first-frame path. See [`KeyframeAnchor`].
pub fn anchors_for(has_first: bool, has_last: bool) -> Vec<KeyframeAnchor> {
    let mut v = Vec::with_capacity(2);
    if has_first {
        v.push(KeyframeAnchor::First);
    }
    if has_last {
        v.push(KeyframeAnchor::Last);
    }
    v
}

/// Normalize a canvas-sized keyframe into the video VAE's pixel space: `x/255`, then per-channel
/// **ImageNet** mean/std. Returns `[1, 3, 1, H, W]` — the single-frame shape
/// [`crate::vae::MiniMaxH3VideoVae::encode`] short-circuits on.
///
/// Note this is a *different* normalization from the one the Qwen3-VL vision tower applies to the
/// same image (`0.5/0.5` onto `[-1, 1]`, after its own bicubic resize to 16-px patches). A keyframe
/// really is preprocessed twice, two different ways, for the two paths it feeds.
pub fn keyframe_to_vae_pixels(image: &Image) -> Result<Array> {
    let (w, h) = (image.width as i32, image.height as i32);
    let expected = (w as usize) * (h as usize) * 3;
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "minimax-h3 keyframe: a {w}x{h} RGB8 image is {expected} bytes, got {}",
            image.pixels.len()
        )));
    }
    if w == 0 || h == 0 {
        return Err(Error::Msg(
            "minimax-h3 keyframe: cannot normalize an empty image".into(),
        ));
    }
    let floats: Vec<f32> = image.pixels.iter().map(|&b| f32::from(b)).collect();
    // HWC -> CHW, then insert the singleton batch and temporal axes.
    let hwc = Array::from_slice(&floats, &[h, w, 3]);
    let chw = hwc.transpose_axes(&[2, 0, 1])?;
    let mean = Array::from_slice(&PIXEL_MEAN, &[3, 1, 1]);
    let std = Array::from_slice(&PIXEL_STD, &[3, 1, 1]);
    let normed = chw
        .divide(Array::from_f32(255.0))?
        .subtract(&mean)?
        .divide(&std)?;
    Ok(normed.reshape(&[1, 3, 1, h, w])?)
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

    /// All four conditioning shapes come out of one ordered filter, and `last`-only is one of
    /// them rather than a special case.
    #[test]
    fn all_four_anchor_shapes_come_from_one_filter() {
        assert_eq!(anchors_for(false, false), vec![]);
        assert_eq!(anchors_for(true, false), vec![KeyframeAnchor::First]);
        assert_eq!(anchors_for(false, true), vec![KeyframeAnchor::Last]);
        assert_eq!(
            anchors_for(true, true),
            vec![KeyframeAnchor::First, KeyframeAnchor::Last]
        );
        // The order is packed order: first always precedes last.
        let both = anchors_for(true, true);
        assert_eq!(both[0], KeyframeAnchor::First);
        assert_eq!(both[1], KeyframeAnchor::Last);
    }

    /// Pixel normalization is `x/255` then ImageNet mean/std, in `[1, 3, 1, H, W]` — and it is
    /// **not** the `[-1, 1]` convention the vision tower uses on the same image.
    #[test]
    fn pixels_are_imagenet_normalized_into_a_single_frame() {
        let img = solid(2, 3, [255, 0, 128]);
        let x = keyframe_to_vae_pixels(&img).unwrap();
        assert_eq!(x.shape(), &[1, 3, 1, 3, 2], "[B, C, T=1, H, W]");
        let v = x.as_slice::<f32>();
        // Channel-major: 6 red entries, then 6 green, then 6 blue.
        let want_r = (1.0 - PIXEL_MEAN[0]) / PIXEL_STD[0];
        let want_g = (0.0 - PIXEL_MEAN[1]) / PIXEL_STD[1];
        let want_b = (128.0 / 255.0 - PIXEL_MEAN[2]) / PIXEL_STD[2];
        assert!((v[0] - want_r).abs() < 1e-5, "R {} != {want_r}", v[0]);
        assert!((v[6] - want_g).abs() < 1e-5, "G {} != {want_g}", v[6]);
        assert!((v[12] - want_b).abs() < 1e-5, "B {} != {want_b}", v[12]);
        // A `[-1, 1]` normalization would put white at exactly 1.0; ImageNet's does not.
        assert!(
            (want_r - 1.0).abs() > 0.5,
            "ImageNet normalization must not coincide with the 0.5/0.5 one"
        );

        let ragged = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 3],
        };
        assert!(keyframe_to_vae_pixels(&ragged).is_err());
    }

    /// The canvas really is derived from the keyframe: two keyframes of different aspect give
    /// two different canvases, which is what "a keyframe binds the canvas" means operationally.
    #[test]
    fn the_keyframe_binds_the_canvas() {
        let landscape = resolve_keyframe_canvas(1920, 1080).unwrap();
        let portrait = resolve_keyframe_canvas(1080, 1920).unwrap();
        let square = resolve_keyframe_canvas(512, 512).unwrap();
        assert_eq!(landscape, (1344, 768));
        assert_eq!(portrait, (768, 1344));
        assert_eq!(square, (768, 768));
        assert_ne!(landscape, portrait);
        // …and all three are legal render canvases, i.e. they survive `resolve_geometry`.
        for (w, h) in [landscape, portrait, square] {
            crate::pipeline::resolve_geometry(w, h, crate::pipeline::SMALLEST_LEGAL_FRAMES)
                .unwrap_or_else(|e| panic!("{w}x{h} must be a legal canvas: {e}"));
        }
    }
}
