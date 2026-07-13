//! Array-level **tiled VAE decode** (sc-11747) — the MLX/tensor half of the gen-core tiling seam.
//!
//! [`crate::tiling`] (re-exported from gen-core) is the **pure** half: tiling presets, the per-axis
//! interval split, the 1-D trapezoidal blend mask, and the [`TilePlan`] for a latent — no tensor dep,
//! Linux-buildable. This module is the tensor half: given a [`TilePlan`] and a per-tile decode closure,
//! it slices each overlapping tile out of the (already-denormalized) latent, decodes it, trapezoidally
//! blends the results, and pad-and-accumulates them into the full output while keeping the peak bounded
//! by one tile's decode.
//!
//! It is **layout-agnostic** (the caller passes the `[t, h, w]` axis indices for NCTHW vs channels-last
//! and a decode closure that reaches its own VAE's decoder), so every VAE that tiles a decode shares it:
//! the Wan z16/z48 video VAEs (`mlx-gen-wan`, via a thin `vae_common` delegator preserving their call
//! sites) and the Qwen-Image still-image VAE (`mlx-gen-qwen-image`, the Krea 2 pose-control decode this
//! story bounds). Lifting it here removes the divergence hazard of a per-crate copy of this subtle
//! slice/blend/pad/accumulate loop (the Wan sc-4998/sc-5690 seam-artifact history).

use crate::tiling::TilePlan;
use crate::{CancelFlag, Error, Result};
use mlx_rs::ops::{add, divide, maximum, multiply, pad};
use mlx_rs::Array;

/// A length-1 `f32` array, used as a broadcastable scalar operand in MLX ops.
fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy. mlx-rs host reads (`as_slice`) return the *physical* buffer, so
/// an array left strided by a `transpose` is read scrambled; a reshape round-trip materializes logical
/// order. Only needed at the host-read boundary (the public decode output) — internal mlx ops are
/// stride-aware.
fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// Gather the contiguous range `[start, end)` along `axis` (mlx-rs has no slice op). Layout-agnostic.
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// `[1; 5]` with `len` placed at `axis` — a 1-D blend mask reshaped to broadcast along its own axis.
fn axis_shape(axis: i32, len: i32) -> [i32; 5] {
    let mut s = [1i32; 5];
    s[axis as usize] = len;
    s
}

/// The trapezoidally-blended tile-accumulate loop shared by every tiled `decode`. Slices each
/// overlapping tile out of `denorm` (the already-denormalized latent), decodes it via the
/// layout-specific `decode_tile` closure, trapezoidally blends along the three tiled axes, and
/// accumulates into the full output. `axes` are the `[t, h, w]` axis indices for the layout (`[2, 3, 4]`
/// for NCTHW, `[1, 2, 3]` for channels-last); the mask shapes and pad placements derive from those
/// indices, so the only per-layout input is the closure.
///
/// `plan` comes from [`TilingConfig::plan`](crate::tiling::TilingConfig::plan). The reference's per-tile
/// `mx.eval` (bounding the lazy graph + peak memory) is preserved — without it the whole tiled graph
/// would materialize at once, defeating the point of tiling.
///
/// `cancel` is the cooperative cancellation handle: the decode is a dominant fraction of a render's
/// wall-clock, so a cancel is checked between tiles and returns [`Error::Canceled`]. The per-tile `eval`
/// forces materialization, so the check observes the trip promptly.
pub fn tiled_decode(
    denorm: &Array,
    plan: &TilePlan,
    axes: [i32; 3],
    cancel: Option<&CancelFlag>,
    decode_tile: impl Fn(&Array) -> Result<Array>,
) -> Result<Array> {
    let [t_ax, h_ax, w_ax] = axes;
    let mut output: Option<Array> = None;
    let mut weights: Option<Array> = None;
    for t in &plan.t {
        for hh in &plan.h {
            for ww in &plan.w {
                if cancel.is_some_and(CancelFlag::is_cancelled) {
                    return Err(Error::Canceled);
                }
                let tile = slice_axis(denorm, t_ax, t.start, t.end)?;
                let tile = slice_axis(&tile, h_ax, hh.start, hh.end)?;
                let tile = slice_axis(&tile, w_ax, ww.start, ww.end)?;
                let dec = decode_tile(&tile)?;

                let ds = dec.shape();
                let at = ds[t_ax as usize].min(t.out_stop - t.out_start);
                let ah = ds[h_ax as usize].min(hh.out_stop - hh.out_start);
                let aw = ds[w_ax as usize].min(ww.out_stop - ww.out_start);

                // 1-D masks → outer product, each broadcasting along its own (t/h/w) axis.
                let tm = Array::from_slice(&t.mask[..at as usize], &axis_shape(t_ax, at));
                let hm = Array::from_slice(&hh.mask[..ah as usize], &axis_shape(h_ax, ah));
                let wm = Array::from_slice(&ww.mask[..aw as usize], &axis_shape(w_ax, aw));
                let blend = multiply(&multiply(&tm, &hm)?, &wm)?;

                let dec = slice_axis(&dec, t_ax, 0, at)?;
                let dec = slice_axis(&dec, h_ax, 0, ah)?;
                let dec = slice_axis(&dec, w_ax, 0, aw)?;
                let weighted = multiply(&dec, &blend)?;

                // Place at the (out_start) offsets by zero-padding to the full output shape.
                let mut pads = [(0, 0); 5];
                pads[t_ax as usize] = (t.out_start, plan.out_f - (t.out_start + at));
                pads[h_ax as usize] = (hh.out_start, plan.out_h - (hh.out_start + ah));
                pads[w_ax as usize] = (ww.out_start, plan.out_w - (ww.out_start + aw));
                let weighted_full = pad(&weighted, &pads[..], None, None)?;
                let blend_full = pad(&blend, &pads[..], None, None)?;

                output = Some(match output {
                    None => weighted_full,
                    Some(acc) => add(&acc, &weighted_full)?,
                });
                weights = Some(match weights {
                    None => blend_full,
                    Some(acc) => add(&acc, &blend_full)?,
                });
                // Bound the lazy graph + peak memory (the reference's per-tile `mx.eval`).
                output.as_ref().unwrap().eval()?;
                weights.as_ref().unwrap().eval()?;
            }
        }
    }

    let output = output.ok_or_else(|| Error::Msg("vae tiled decode: plan had no tiles".into()))?;
    let weights = weights.ok_or_else(|| Error::Msg("vae tiled decode: plan had no tiles".into()))?;
    contiguous(&divide(&output, &maximum(&weights, scalar(1e-8))?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiling::{AxisTile, SpatialTiling, TilingConfig, VaeTiling};

    /// Two non-overlapping tiles along the temporal axis with all-ones masks and an identity decode must
    /// exactly reconstruct the input — exercising slice/mask/pad placement and accumulation for a given
    /// axis layout.
    fn roundtrip(denorm: &Array, axes: [i32; 3], t_full: i32) -> Vec<f32> {
        let half = t_full / 2;
        let tile = |start, out_start| AxisTile {
            start,
            end: start + half,
            out_start,
            out_stop: out_start + half,
            mask: vec![1.0; half as usize],
        };
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![tile(0, 0), tile(half, half)],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: t_full,
            out_h: 2,
            out_w: 2,
        };
        let out = tiled_decode(denorm, &plan, axes, None, |tile| Ok(tile.clone())).unwrap();
        out.eval().unwrap();
        out.as_slice::<f32>().to_vec()
    }

    #[test]
    fn identity_roundtrip_ncthw() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        assert_eq!(roundtrip(&denorm, [2, 3, 4], 4), vals);
    }

    #[test]
    fn identity_roundtrip_channels_last() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 4, 2, 2, 1]);
        assert_eq!(roundtrip(&denorm, [1, 2, 3], 4), vals);
    }

    #[test]
    fn honors_pretripped_cancel() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        let half = 2;
        let tile = |start, out_start| AxisTile {
            start,
            end: start + half,
            out_start,
            out_stop: out_start + half,
            mask: vec![1.0; half as usize],
        };
        let unit = AxisTile {
            start: 0,
            end: 2,
            out_start: 0,
            out_stop: 2,
            mask: vec![1.0; 2],
        };
        let plan = TilePlan {
            t: vec![tile(0, 0), tile(half, half)],
            h: vec![unit.clone()],
            w: vec![unit],
            out_f: 4,
            out_h: 2,
            out_w: 2,
        };
        let cancel = CancelFlag::new();
        cancel.cancel();
        let res = tiled_decode(&denorm, &plan, [2, 3, 4], Some(&cancel), |t| Ok(t.clone()));
        assert!(matches!(res, Err(Error::Canceled)));
    }

    /// Block (nearest-neighbour) spatial upsample by `scale` along `axes[1..]` — tile-consistent, so a
    /// correct partition-of-unity blend reconstructs the full upsample exactly. Isolates the
    /// slice/mask/pad/accumulate/normalize machinery for the **image** (T=1) case this story adds.
    fn upsample_spatial(x: &Array, axes: [i32; 3], scale: i32) -> Array {
        let mut y = x.clone();
        for &ax in &axes[1..] {
            y = Array::repeat_axis::<f32>(y, scale, ax).unwrap();
        }
        y
    }

    /// sc-11747: the Qwen-Image case — a single temporal frame (T=1), spatial ×8, tiled on H and W.
    /// A tile-consistent block-upsample decode blended through the real [`TilingConfig::plan`] geometry
    /// must reconstruct the full upsample exactly (no seam), proving the image path of the shared loop.
    #[test]
    fn image_spatial_tiles_reconstruct() {
        let vae = VaeTiling::QWEN_IMAGE; // spatial ×8, temporal ×1
        let cfg = TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px: 4 * vae.spatial_scale, // 4-latent tiles
                overlap_px: 2 * vae.spatial_scale,
            }),
            temporal: None,
        };
        // NCTHW latent [1, 16→2 (tiny), 1, 13, 13]: ragged 3×3 spatial tiling, T=1.
        let (f, h, w) = (1, 13, 13);
        assert!(cfg.needs_tiling(vae, f, h, w));
        let plan = cfg.plan(vae, f, h, w);
        let shape = [1, 2, f, h, w];
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.19).sin()).collect();
        let denorm = Array::from_slice(&vals, &shape);

        let expected = upsample_spatial(&denorm, [2, 3, 4], vae.spatial_scale);
        let got = tiled_decode(&denorm, &plan, [2, 3, 4], None, |t| {
            Ok(upsample_spatial(t, [2, 3, 4], vae.spatial_scale))
        })
        .unwrap();
        got.eval().unwrap();
        assert_eq!(got.shape(), expected.shape());
        let (g, e) = (got.as_slice::<f32>(), expected.as_slice::<f32>());
        let max = g
            .iter()
            .zip(e)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-4, "image tiled blend did not reconstruct: max|Δ|={max:.3e}");
    }
}
