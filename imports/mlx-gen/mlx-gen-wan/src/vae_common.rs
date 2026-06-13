//! Layout-agnostic VAE scaffolding shared between the z16 [`vae`](crate::vae) (NCTHW) and the z48
//! [`vae22`](crate::vae22) (channels-last NTHWC) Wan VAEs. Only the pieces that are genuinely
//! byte-identical across the two layouts live here; the per-file conv/norm leaves (which carry the
//! channel axis at different positions) stay with their respective modules.

use mlx_gen::tiling::TilePlan;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{add, divide, maximum, multiply, pad};
use mlx_rs::Array;

/// A length-1 `f32` array, used as a broadcastable scalar operand in MLX ops.
pub(crate) fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy. mlx-rs host reads (`as_slice`) return the *physical* buffer,
/// so an array left strided by a `transpose` is read scrambled. A reshape round-trip materializes
/// logical order. Internal mlx ops are stride-aware, so this is only needed at the host-read
/// boundary (the public decode/encode output).
pub(crate) fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// Gather the contiguous range `[start, end)` along `axis` (mlx-rs has no slice op). Layout-agnostic:
/// the z16 VAE slices the temporal/spatial axes at 2/3/4, the z48 VAE (channels-last) at 1/2/3.
pub(crate) fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// Per-conv last-frames cache threaded through the chunked encode. `idx` resets to 0 each chunk and
/// advances once per cache-bearing conv (in the fixed traversal order), so slots stay aligned.
pub(crate) struct FeatCache {
    pub(crate) slots: Vec<Option<Array>>,
    pub(crate) idx: usize,
}
impl FeatCache {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            slots: vec![None; n],
            idx: 0,
        }
    }
}

/// `[1; 5]` with `len` placed at `axis` — a 1-D blend mask reshaped to broadcast along its own axis.
fn axis_shape(axis: i32, len: i32) -> [i32; 5] {
    let mut s = [1i32; 5];
    s[axis as usize] = len;
    s
}

/// The trapezoidally-blended tile-accumulate loop shared by both VAEs' `decode_tiled`. Slices each
/// overlapping tile out of `denorm`, decodes it via the layout-specific `decode_tile` closure
/// (conv2 → decoder → optional unpatchify → clamp), trapezoidally blends along the three tiled axes,
/// and accumulates into the full output. `axes` are the `[t, h, w]` axis indices for the layout
/// (`[2, 3, 4]` for NCTHW z16, `[1, 2, 3]` for channels-last z48); the mask shapes and pad
/// placements derive from those indices, so the only per-layout input is the closure.
///
/// `denorm` is the already-denormalized latent; `plan` comes from
/// [`TilingConfig::plan`](mlx_gen::tiling::TilingConfig::plan). The reference's per-tile `mx.eval`
/// (bounding the lazy graph + peak memory) is preserved.
pub(crate) fn tile_decode_accumulate(
    denorm: &Array,
    plan: &TilePlan,
    axes: [i32; 3],
    decode_tile: impl Fn(&Array) -> Result<Array>,
) -> Result<Array> {
    let [t_ax, h_ax, w_ax] = axes;
    let mut output: Option<Array> = None;
    let mut weights: Option<Array> = None;
    for t in &plan.t {
        for hh in &plan.h {
            for ww in &plan.w {
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

    let output =
        output.ok_or_else(|| Error::Msg("wan vae: tile-decode plan had no tiles".into()))?;
    let weights =
        weights.ok_or_else(|| Error::Msg("wan vae: tile-decode plan had no tiles".into()))?;
    contiguous(&divide(&output, &maximum(&weights, scalar(1e-8))?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::tiling::{AxisTile, TilePlan};

    /// Two non-overlapping tiles along the temporal axis with all-ones masks and an identity decode
    /// must exactly reconstruct the input — exercising slice/mask/pad placement and accumulation for
    /// a given axis layout. Returns the round-tripped values for comparison against the input.
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
        let out = tile_decode_accumulate(denorm, &plan, axes, |tile| Ok(tile.clone())).unwrap();
        out.eval().unwrap();
        out.as_slice::<f32>().to_vec()
    }

    #[test]
    fn identity_roundtrip_ncthw() {
        // [1, 1, 4, 2, 2] — channel axis at 1, tiled axes [2, 3, 4].
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 1, 4, 2, 2]);
        assert_eq!(roundtrip(&denorm, [2, 3, 4], 4), vals);
    }

    #[test]
    fn identity_roundtrip_channels_last() {
        // [1, 4, 2, 2, 1] — channel axis last, tiled axes [1, 2, 3].
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let denorm = Array::from_slice(&vals, &[1, 4, 2, 2, 1]);
        assert_eq!(roundtrip(&denorm, [1, 2, 3], 4), vals);
    }
}
