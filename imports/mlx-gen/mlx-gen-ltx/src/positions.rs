//! Float32 RoPE **position grid** in pixel space — port of `generate.py::create_position_grid`
//! (identical in `generate_av.py`).
//!
//! For latent dims `(num_frames, height, width)` and patch size `(1,1,1)`, each latent token gets
//! `[start, end)` bounds on three axes (frame, height, width), scaled from latent to pixel space by
//! the VAE factors (temporal 8×, spatial 32×). Two LTX-specific corrections on the **frame** axis:
//! a **causal first-frame fix** (the VAE's first-frame temporal stride is 1, not `temporal_scale`)
//! and **fps division** (frame index → time in seconds). Always f32 — the reference warns that
//! bf16 position grids degrade RoPE quality (sc-2679 keeps positions f32).
//!
//! Output shape `(batch, 3, num_frames·height·width, 2)`, C-order, where the last axis is
//! `[start, end]`. The token order is C-major over `(frame, height, width)` (`meshgrid(indexing="ij")`).

use mlx_rs::Array;

/// LTX-2 VAE factors + sampling defaults used by the T2V pipeline.
pub const TEMPORAL_SCALE: i64 = 8;
pub const SPATIAL_SCALE: i64 = 32;
pub const DEFAULT_FPS: f32 = 24.0;

/// Build the position grid with the LTX-2.3 defaults (temporal 8×, spatial 32×, 24 fps, causal fix).
pub fn create_position_grid(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
) -> Array {
    create_position_grid_with(
        batch_size,
        num_frames,
        height,
        width,
        TEMPORAL_SCALE,
        SPATIAL_SCALE,
        DEFAULT_FPS,
        true,
    )
}

/// Build the position grid with explicit VAE scale factors / fps / causal-fix toggle.
///
/// Mirrors the reference op order exactly: integer `latent · scale` (exact), cast to f32, then the
/// frame-axis causal fix `max(0, px + 1 − temporal_scale)` and `÷ fps` are applied in f32 — so the
/// only rounding is the final `÷ fps`, matching numpy under NEP 50 (f32 array ÷ python float stays f32).
#[allow(clippy::too_many_arguments)]
pub fn create_position_grid_with(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
    causal_fix: bool,
) -> Array {
    let hw = height * width;
    let num_patches = num_frames * hw;
    // C-order (batch, 3, num_patches, 2).
    let mut data = vec![0f32; batch_size * 3 * num_patches * 2];

    for p in 0..num_patches {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let h = (rem / width) as i64;
        let w = (rem % width) as i64;

        for e in 0..2i64 {
            // frame axis (d=0): pixel = (t + e) · temporal_scale, then causal fix + fps.
            let frame_pix = (t + e) * temporal_scale;
            let mut frame_f = frame_pix as f32;
            if causal_fix {
                frame_f = (frame_f + 1.0 - temporal_scale as f32).max(0.0);
            }
            frame_f /= fps;

            // height axis (d=1) and width axis (d=2): pixel = (coord + e) · spatial_scale.
            let height_f = ((h + e) * spatial_scale) as f32;
            let width_f = ((w + e) * spatial_scale) as f32;

            for b in 0..batch_size {
                let base = ((b * 3) * num_patches + p) * 2 + e as usize;
                data[base] = frame_f; // d = 0
                data[base + num_patches * 2] = height_f; // d = 1
                data[base + 2 * num_patches * 2] = width_f; // d = 2
            }
        }
    }

    Array::from_slice(&data, &[batch_size as i32, 3, num_patches as i32, 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_first_frame_causal_fix() {
        // num_frames=2, h=3, w=4 → 24 patches.
        let g = create_position_grid(1, 2, 3, 4);
        assert_eq!(g.shape(), &[1, 3, 24, 2]);
        let v: Vec<f32> = g.as_slice::<f32>().to_vec();
        // C-order index helper for (b=0, d, p, e): ((d)*24 + p)*2 + e.
        let at = |d: usize, p: usize, e: usize| v[(d * 24 + p) * 2 + e];

        // Patch p=0 → (t=0,h=0,w=0). Frame axis: start clip(0+1-8)=0 → /24 = 0;
        // end clip(8+1-8)=1 → /24.
        assert!((at(0, 0, 0) - 0.0).abs() < 1e-9);
        assert!((at(0, 0, 1) - (1.0 / 24.0)).abs() < 1e-7);
        // Height axis at p=0: start 0, end 32.
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 1), 32.0);
        // Width axis at p=0: start 0, end 32.
        assert_eq!(at(2, 0, 0), 0.0);
        assert_eq!(at(2, 0, 1), 32.0);

        // Patch p=12 → (t=1,h=0,w=0): frame start clip(8+1-8)=1 → /24, end clip(16+1-8)=9 → /24.
        assert!((at(0, 12, 0) - (1.0 / 24.0)).abs() < 1e-7);
        assert!((at(0, 12, 1) - (9.0 / 24.0)).abs() < 1e-7);
        // Patch p=5 → (t=0,h=1,w=1): height start 32, end 64; width start 32, end 64.
        assert_eq!(at(1, 5, 0), 32.0);
        assert_eq!(at(1, 5, 1), 64.0);
        assert_eq!(at(2, 5, 0), 32.0);
        assert_eq!(at(2, 5, 1), 64.0);
    }
}
