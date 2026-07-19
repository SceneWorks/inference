//! Shared leaf helpers (mirrors `mlx-gen-sam3`'s `util`): weight-key joining, torch→MLX
//! conv-weight permutes, and shared device/host bilinear resizers.

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::Result;

/// `"{prefix}.{leaf}"` (or just `leaf` when `prefix` is empty).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// Permute a torch conv weight `[out, in, kH, kW]` (OIHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_w_ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Permute a torch transposed-conv weight `[in, out, kH, kW]` (IOHW) → MLX `[out, kH, kW, in]` (OHWI).
pub(crate) fn conv_transpose_w(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[1, 2, 3, 0])?)
}

/// Build a 1-D bilinear gather: for an axis of length `in_len` resized to `out_len`, return the two
/// integer source indices (`lo`, `hi`) and the fractional weight `frac` (= weight on `hi`) per output
/// position, following torch `interpolate(mode="bilinear")`.
///
/// `align_corners == true` maps output `i` to source `i * (in-1)/(out-1)`; `false` maps to the
/// pixel-center convention `(i + 0.5) * in/out - 0.5` (clamped to `[0, in-1]`).
fn bilinear_axis(in_len: i32, out_len: i32, align_corners: bool) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut lo = Vec::with_capacity(out_len as usize);
    let mut hi = Vec::with_capacity(out_len as usize);
    let mut frac = Vec::with_capacity(out_len as usize);
    let last = in_len - 1;
    for i in 0..out_len {
        let src = if align_corners {
            if out_len == 1 {
                0.0
            } else {
                i as f32 * (in_len - 1) as f32 / (out_len - 1) as f32
            }
        } else {
            let s = (i as f32 + 0.5) * in_len as f32 / out_len as f32 - 0.5;
            s.max(0.0)
        };
        let l = src.floor() as i32;
        let l = l.clamp(0, last);
        let h = (l + 1).min(last);
        lo.push(l);
        hi.push(h);
        frac.push((src - l as f32).clamp(0.0, 1.0));
    }
    (lo, hi, frac)
}

/// Resample one spatial axis (`axis` ∈ {1=H, 2=W} of an NHWC tensor) from its current length to
/// `out_len` by bilinear interpolation. Implemented as a gather of the two bracketing rows/cols and
/// a fractional blend, so it runs entirely in MLX (no host loop over pixels).
fn resample_axis(x: &Array, axis: i32, out_len: i32, align_corners: bool) -> Result<Array> {
    let in_len = x.shape()[axis as usize];
    if in_len == out_len {
        return Ok(x.clone());
    }
    let (lo, hi, frac) = bilinear_axis(in_len, out_len, align_corners);
    let lo = Array::from_slice(&lo, &[out_len]);
    let hi = Array::from_slice(&hi, &[out_len]);
    // Broadcast the per-output weight over the gathered tensor: shape [1,…,out_len,…,1].
    let mut wshape = vec![1i32; x.shape().len()];
    wshape[axis as usize] = out_len;
    let w_hi = Array::from_slice(&frac, &wshape);
    let ones = Array::from_slice(&vec![1.0f32; out_len as usize], &wshape);
    let w_lo = mlx_rs::ops::subtract(&ones, &w_hi)?;

    let g_lo = x.take_axis(&lo, axis)?;
    let g_hi = x.take_axis(&hi, axis)?;
    Ok(add(&multiply(&g_lo, &w_lo)?, &multiply(&g_hi, &w_hi)?)?)
}

/// NHWC bilinear resize `[B, H, W, C]` → `[B, out_h, out_w, C]` (torch `interpolate(mode="bilinear")`).
/// Separable: resample H then W. `align_corners` matches the torch flag used at the call site.
pub(crate) fn bilinear_resize(
    x: &Array,
    out_h: i32,
    out_w: i32,
    align_corners: bool,
) -> Result<Array> {
    let y = resample_axis(x, 1, out_h, align_corners)?;
    resample_axis(&y, 2, out_w, align_corners)
}

/// Host RGB8 HWC bilinear resize using half-pixel centers (`align_corners = false`) and clamped
/// edges. Returns interpolated channel values in the source byte range `[0, 255]`; callers retain
/// their own post-processing (unit scaling for model input or rounded RGB8 output).
pub(crate) fn bilinear_resize_rgb8_f32(
    rgb: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_h * out_w * 3];
    let sx = in_w as f32 / out_w as f32;
    let sy = in_h as f32 / out_h as f32;
    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(in_h - 1);
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(in_w - 1);
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let p = |y: usize, x: usize| rgb[(y * in_w + x) * 3 + c] as f32;
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[(oy * out_w + ox) * 3 + c] = top * (1.0 - wy) + bot * wy;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::bilinear_resize_rgb8_f32;

    #[test]
    fn host_bilinear_uses_half_pixel_centers_and_clamped_edges() {
        let rgb = [0, 0, 0, 10, 10, 10, 20, 20, 20, 30, 30, 30];
        let out = bilinear_resize_rgb8_f32(&rgb, 2, 2, 4, 4);
        let at = |y: usize, x: usize| out[(y * 4 + x) * 3];
        assert_eq!(at(0, 0), 0.0);
        assert_eq!(at(1, 1), 7.5);
        assert_eq!(at(2, 2), 22.5);
        assert_eq!(at(3, 3), 30.0);
        assert!(out
            .chunks_exact(3)
            .all(|px| px[0] == px[1] && px[1] == px[2]));
    }
}
