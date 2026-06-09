//! InstantID face-restoration compositing (sc-3380).
//!
//! The face-restore pass (`instantid_adapter.py::_restore_face`, sc-2063) is an ADetailer-style
//! re-render of the cropped face through the **same InstantID pipe** (IdentityNet only) followed by a
//! feathered paste-back. This module holds the net-new image-compositing pieces — the feathered
//! elliptical alpha mask and the alpha paste-back; the crop/re-render orchestration lives in
//! [`crate::model::InstantId::restore_face`].
//!
//! The gate is **directional** (epic 3109: identity + seamlessness, not bit-exact vs PIL): the mask is
//! a filled ellipse in the inner `[0.1, 0.9]` box of the crop, Gaussian-blurred so the paste-back has
//! no hard edges — mirroring the reference's `ImageDraw.ellipse(...) + GaussianBlur`. A true separable
//! Gaussian (σ = blur radius) stands in for PIL's box-blur approximation; the result is a smooth
//! feather, which is all the composite needs.

use mlx_gen::media::Image;

/// Build the feathered elliptical alpha mask for a `crop_w × crop_h` face crop — a filled ellipse in
/// the inner `[0.1·w, 0.1·h, 0.9·w, 0.9·h]` box, Gaussian-blurred by `max(4, crop_w / 12)` (the
/// reference's feather radius). Returns alpha in `[0, 1]`, length `crop_w · crop_h` (row-major).
pub fn feather_mask(crop_w: usize, crop_h: usize) -> Vec<f32> {
    let (w, h) = (crop_w as f64, crop_h as f64);
    // PIL `ImageDraw.ellipse([x0,y0,x1,y1])` fills the ellipse inscribed in the box.
    let (x0, y0) = (0.1 * w, 0.1 * h);
    let (x1, y1) = (0.9 * w, 0.9 * h);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (rx, ry) = (((x1 - x0) / 2.0).max(1.0), ((y1 - y0) / 2.0).max(1.0));

    let mut mask = vec![0f32; crop_w * crop_h];
    for y in 0..crop_h {
        for x in 0..crop_w {
            // Pixel-center test against the ellipse.
            let dx = (x as f64 + 0.5 - cx) / rx;
            let dy = (y as f64 + 0.5 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * crop_w + x] = 1.0;
            }
        }
    }
    let radius = (crop_w / 12).max(4) as f64;
    gaussian_blur(&mask, crop_w, crop_h, radius)
}

/// Separable Gaussian blur of a single-channel image (clamp-to-edge), `sigma` = the blur radius. The
/// kernel half-width is `ceil(3·sigma)`. Used to feather the elliptical mask.
fn gaussian_blur(img: &[f32], w: usize, h: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        return img.to_vec();
    }
    let radius = (3.0 * sigma).ceil() as isize;
    // 1-D Gaussian kernel, normalized.
    let mut kernel = vec![0f64; (2 * radius + 1) as usize];
    let two_s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for (i, k) in kernel.iter_mut().enumerate() {
        let d = i as isize - radius;
        *k = (-(d * d) as f64 / two_s2).exp();
        sum += *k;
    }
    for k in &mut kernel {
        *k /= sum;
    }

    let clamp = |v: isize, hi: usize| -> usize { v.max(0).min(hi as isize - 1) as usize };

    // Horizontal pass.
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f64;
            for (i, &k) in kernel.iter().enumerate() {
                let sx = clamp(x as isize + i as isize - radius, w);
                acc += k * img[y * w + sx] as f64;
            }
            tmp[y * w + x] = acc as f32;
        }
    }
    // Vertical pass.
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f64;
            for (i, &k) in kernel.iter().enumerate() {
                let sy = clamp(y as isize + i as isize - radius, h);
                acc += k * tmp[sy * w + x] as f64;
            }
            out[y * w + x] = acc as f32;
        }
    }
    out
}

/// Alpha-composite a `crop_w × crop_h` RGB8 patch `small` onto `base` at top-left `(ax, ay)`, using
/// per-pixel `alpha` in `[0, 1]` — the feathered paste-back (`Image.paste(small, (a,b), mask)`).
/// `out = base·(1-α) + small·α`, rounded to u8. The crop box is assumed in-bounds (clamped by the
/// caller); any out-of-bounds pixel is skipped.
pub fn paste_alpha(
    base: &mut Image,
    small: &[u8],
    crop_w: usize,
    crop_h: usize,
    ax: usize,
    ay: usize,
    alpha: &[f32],
) {
    let bw = base.width as usize;
    let bh = base.height as usize;
    for y in 0..crop_h {
        let by = ay + y;
        if by >= bh {
            break;
        }
        for x in 0..crop_w {
            let bx = ax + x;
            if bx >= bw {
                break;
            }
            let a = alpha[y * crop_w + x].clamp(0.0, 1.0);
            let si = (y * crop_w + x) * 3;
            let di = (by * bw + bx) * 3;
            for c in 0..3 {
                let b = base.pixels[di + c] as f32;
                let s = small[si + c] as f32;
                base.pixels[di + c] = (b * (1.0 - a) + s * a).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-3380: the feather mask is opaque at the crop center and feathers to ~transparent at the
    /// corners (a seamless paste-back has no hard edge).
    #[test]
    fn feather_mask_is_centered_and_soft() {
        let (w, h) = (200usize, 240usize);
        let m = feather_mask(w, h);
        assert_eq!(m.len(), w * h);
        let center = m[(h / 2) * w + w / 2];
        let corner = m[0];
        let edge_mid = m[(h / 2) * w]; // left edge, vertical center (outside the inner ellipse)
        assert!(center > 0.95, "center alpha {center} should be ~opaque");
        assert!(
            corner < 0.05,
            "corner alpha {corner} should be ~transparent"
        );
        assert!(
            edge_mid < center,
            "edge alpha {edge_mid} should feather below the center {center}"
        );
        // No hard edge: every value is a valid, finite alpha in [0, 1].
        assert!(m.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    /// sc-3380: alpha paste-back is a straight per-pixel blend — α=1 replaces, α=0 keeps.
    #[test]
    fn paste_alpha_blends() {
        let mut base = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 3],
        };
        let small = vec![200u8; 2 * 2 * 3]; // a 2×2 patch of 200
        let alpha = vec![1.0, 0.0, 0.5, 1.0]; // one of each blend
        paste_alpha(&mut base, &small, 2, 2, 1, 1, &alpha);
        // (1,1) α=1 → 200; (2,1) α=0 → 0; (1,2) α=0.5 → 100; (2,2) α=1 → 200.
        let px = |x: usize, y: usize| base.pixels[(y * 4 + x) * 3];
        assert_eq!(px(1, 1), 200);
        assert_eq!(px(2, 1), 0);
        assert_eq!(px(1, 2), 100);
        assert_eq!(px(2, 2), 200);
        // Untouched pixel stays 0.
        assert_eq!(px(0, 0), 0);
    }
}
