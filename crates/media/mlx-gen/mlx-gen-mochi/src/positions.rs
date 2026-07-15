//! Mochi 3-D RoPE **positions** — port of `MochiRoPE._get_positions` /
//! `MochiRoPE._centers` (diffusers `transformer_mochi.py`).
//!
//! The visual sequence is a `(num_frames, post_patch_height, post_patch_width)` grid. Each token gets
//! a continuous 3-vector position `(t, h, w)`: the temporal axis is the raw frame index, while the two
//! spatial axes are **area-normalized cell centers** — a center grid scaled so the token grid always
//! covers the same reference area (`base_height·base_width = 192·192`) regardless of the actual
//! latent resolution. The stack is row-major over `(t, h, w)` (`meshgrid(..., indexing="ij")` then
//! `view(-1, 3)`), so position `p = ((t·H) + h)·W + w`.
//!
//! Computed in **f64** on the host and cast to f32 at the edge. The reference runs `_get_positions`
//! with `dtype=torch.float32`, but the only non-integer values are the spatial cell centers, whose
//! `linspace` midpoints are exact in both f32 and f64 for the shipped geometry (`scale` is a clean
//! power for square latents); f64 host math avoids any `libm`-vs-device trig seed drift before the
//! learned-frequency `einsum`. The result is validated against the `mochi_dit_block_golden`'s
//! `image_rotary_emb` in `rope.rs`.

use mlx_rs::Array;

/// Base reference area for the spatial-axis normalization (`MochiRoPE(base_height=192, base_width=192)`).
const BASE_AREA: f64 = 192.0 * 192.0;

/// `_centers(start, stop, num)`: the `num` midpoints of an evenly-spaced `[start, stop]` partition —
/// `edges = linspace(start, stop, num + 1)`, `centers = (edges[:-1] + edges[1:]) / 2`.
fn centers(start: f64, stop: f64, num: usize) -> Vec<f64> {
    // `edges[i] = start + i·(stop − start)/num`; the midpoint of cell k is the mean of edges k, k+1.
    let step = (stop - start) / num as f64;
    (0..num)
        .map(|k| {
            let e0 = start + step * k as f64;
            let e1 = start + step * (k + 1) as f64;
            (e0 + e1) / 2.0
        })
        .collect()
}

/// The per-token `(t, h, w)` positions for a `(num_frames, height, width)` visual grid, row-major over
/// `(t, h, w)` → `[seq, 3]` f32 (`seq = num_frames·height·width`). `height`/`width` are the
/// **post-patch** latent dims (after the `patch_size` downsample).
pub fn get_positions(num_frames: usize, height: usize, width: usize) -> Array {
    // scale = sqrt(target_area / (H·W)); spatial extent is ±(dim·scale/2), centers evenly inside.
    let scale = (BASE_AREA / (height as f64 * width as f64)).sqrt();
    let h_centers = centers(
        -(height as f64) * scale / 2.0,
        height as f64 * scale / 2.0,
        height,
    );
    let w_centers = centers(
        -(width as f64) * scale / 2.0,
        width as f64 * scale / 2.0,
        width,
    );

    let seq = num_frames * height * width;
    let mut data = Vec::with_capacity(seq * 3);
    for t in 0..num_frames {
        for &hc in &h_centers {
            for &wc in &w_centers {
                data.push(t as f32); // temporal axis = raw frame index
                data.push(hc as f32);
                data.push(wc as f32);
            }
        }
    }
    Array::from_slice(&data, &[seq as i32, 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centers_are_evenly_spaced_midpoints() {
        // The golden geometry: post-patch 4×4, scale = sqrt(36864/16) = 48 → extent ±96, 4 centers.
        let c = centers(-96.0, 96.0, 4);
        assert_eq!(c, vec![-72.0, -24.0, 24.0, 72.0]);
    }

    #[test]
    fn positions_shape_and_row_major_layout() {
        // 2 frames × 4 × 4 = 32 tokens, matching the dit_block golden's visual sequence.
        let pos = get_positions(2, 4, 4);
        assert_eq!(pos.shape(), &[32, 3]);
        let v: Vec<f32> = pos.as_slice::<f32>().to_vec();
        // Row 0 = (t0, h0, w0); temporal is the fastest-changing only across frame blocks.
        assert_eq!(&v[0..3], &[0.0, -72.0, -72.0]);
        // Row 1 = (t0, h0, w1) — w advances fastest.
        assert_eq!(&v[3..6], &[0.0, -72.0, -24.0]);
        // Row 16 begins the second frame (t1, h0, w0).
        assert_eq!(&v[48..51], &[1.0, -72.0, -72.0]);
    }
}
