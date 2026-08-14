//! The MiniMax-H3 VAE's **spatial tiling** — the reference's `_split_tiles` / `_blend` /
//! `_stitch_tiles`, and the geometry it ships with.
//!
//! # Tiling is ON by default, and the released frames are the blended-tile ones
//!
//! `AutoencoderKLMiniMaxH3.__init__` sets `use_tiling = True` with 256 px tiles and a 64 px minimum
//! overlap. That is unusual: almost every other diffusers autoencoder leaves tiling off until
//! `enable_tiling()` is called, so it is the class of fact that gets *assumed* rather than read.
//! Upstream states the consequence in the class docstring — MiniMax-H3 was released with tiling
//! enabled for both encoding and decoding, and **the released frames are the blended-tile ones, so
//! disabling tiling changes the output.**
//!
//! So this is not a memory optimization that a port may skip. A port that runs the whole canvas
//! through the decoder in one pass does not reproduce the reference at any canvas larger than one
//! tile, and the shipped 1344×768 canvas is a 7×4 grid of them.
//!
//! # This is NOT `gen-core::tiling`
//!
//! `gen-core::tiling` (sc-15325) is the *memory-budget* seam: it picks the largest tile that fits a
//! VRAM budget and cross-fades with a trapezoidal mask. It is the right seam for "this decode does
//! not fit", and sc-18660 wants it here for exactly that.
//!
//! It is the **wrong** seam for this module, and substituting it is the specific mistake sc-18786
//! exists to prevent: a memory-driven tile size produces output that still does not match the
//! reference. The tile geometry here is a *correctness* input copied from the published model, not
//! a tunable — 256/64 with a linear cross-fade, or the frames are not the released ones.
//!
//! # Kept in step with the mlx port
//!
//! `mlx-gen-minimax-h3::spatial_tiling` is the same derivation over `mlx_rs::Array`; the two are
//! gated against the same reference numbers so they cannot drift.

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

use crate::blocks::blend;

/// Default tile edge, in pixels (`tile_sample_min_height` / `tile_sample_min_width`).
pub const TILE_SAMPLE_MIN_SIZE: usize = 256;

/// Default minimum tile overlap, in pixels (`tile_sample_min_overlap_height` /
/// `tile_sample_min_overlap_width`).
pub const TILE_SAMPLE_MIN_OVERLAP: usize = 64;

/// Whether the shipped VAE tiles spatially, for **both** encode and decode.
///
/// Pinned as a constant because it is a **default**, and defaults are the class of fact that gets
/// assumed rather than read.
pub const TILING_IS_ON_BY_DEFAULT: bool = true;

/// The reference's tiling knobs — `use_tiling` plus the four `tile_sample_min_*` fields.
///
/// [`Default`] is the shipped configuration, so a VAE built without saying anything about tiling
/// behaves like the published one. [`Self::disabled`] is the reference's `disable_tiling()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialTiling {
    /// `use_tiling`. When false the canvas decodes in one pass, which does **not** match the
    /// released frames above one tile.
    pub enabled: bool,
    /// `tile_sample_min_height`, in pixels.
    pub tile_height: usize,
    /// `tile_sample_min_width`, in pixels.
    pub tile_width: usize,
    /// `tile_sample_min_overlap_height`, in pixels.
    pub overlap_height: usize,
    /// `tile_sample_min_overlap_width`, in pixels.
    pub overlap_width: usize,
}

impl Default for SpatialTiling {
    fn default() -> Self {
        Self {
            enabled: TILING_IS_ON_BY_DEFAULT,
            tile_height: TILE_SAMPLE_MIN_SIZE,
            tile_width: TILE_SAMPLE_MIN_SIZE,
            overlap_height: TILE_SAMPLE_MIN_OVERLAP,
            overlap_width: TILE_SAMPLE_MIN_OVERLAP,
        }
    }
}

impl SpatialTiling {
    /// The reference's `disable_tiling()` — one pass over the whole canvas.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// A square tile geometry, enabled — the shape `enable_tiling(n, n, k, k)` produces.
    ///
    /// The parity fixtures use this to exercise the tiled path at a canvas small enough to commit:
    /// at the shipped 256 px tile a fixture would need a >256 px canvas, so they shrink the tile
    /// instead of shrinking the coverage.
    pub fn square(tile: usize, overlap: usize) -> Self {
        Self {
            enabled: true,
            tile_height: tile,
            tile_width: tile,
            overlap_height: overlap,
            overlap_width: overlap,
        }
    }

    /// The reference's `enable_tiling(...)`: turn tiling on, overriding only the values given.
    pub fn enable(
        &mut self,
        tile_height: Option<usize>,
        tile_width: Option<usize>,
        overlap_height: Option<usize>,
        overlap_width: Option<usize>,
    ) {
        self.enabled = true;
        self.tile_height = tile_height.unwrap_or(self.tile_height);
        self.tile_width = tile_width.unwrap_or(self.tile_width);
        self.overlap_height = overlap_height.unwrap_or(self.overlap_height);
        self.overlap_width = overlap_width.unwrap_or(self.overlap_width);
    }

    /// The reference's `disable_tiling()`, in place.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

/// Where one axis's tiles start, how long each is, and the overlap between consecutive tiles.
///
/// This is the reference's `_split_tiles`. The tile count is the smallest whose union covers the
/// axis at the minimum overlap; the slack is then distributed round-robin over the overlaps in
/// whole `spatial_compression_ratio` steps, so every tile boundary stays latent-aligned.
///
/// **All three vectors are in pixels**, on both the encode and the decode side — the decode side
/// divides `starts` and `lengths` by the ratio itself when it indexes the latent, and stitches with
/// `overlaps` undivided because its tiles come back out in pixel space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    /// Pixel index each tile starts at.
    pub starts: Vec<usize>,
    /// Pixel length of each tile.
    pub lengths: Vec<usize>,
    /// Pixel overlap between tile `i` and tile `i + 1`; `starts.len() - 1` entries.
    pub overlaps: Vec<usize>,
}

impl TilePlan {
    /// Lay `tile_size`-wide tiles over `length` pixels at at least `min_overlap` overlap.
    ///
    /// `tile_size` and `min_overlap` must both be whole multiples of
    /// `spatial_compression_ratio`. The plan is in pixels but the decode side indexes the *latent*
    /// with `start / ratio` and `length / ratio`, so a misaligned geometry truncates every tile —
    /// `enable_tiling(Some(100), ..)` at ratio 16 would decode 96 px tiles while stitching with
    /// 100 px-derived overlaps and return a wrong-sized video with no error. The shipped 256/64 is
    /// aligned; this is a guard on the public [`SpatialTiling::enable`] surface.
    pub fn split(
        length: usize,
        tile_size: usize,
        min_overlap: usize,
        spatial_compression_ratio: usize,
    ) -> Result<Self> {
        if length == 0 || tile_size == 0 || spatial_compression_ratio == 0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: tile plan needs positive length/tile/ratio, got \
                 {length}/{tile_size}/{spatial_compression_ratio}"
            )));
        }
        if min_overlap >= tile_size {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: tile overlap {min_overlap} must be within [0, {tile_size})"
            )));
        }
        if !tile_size.is_multiple_of(spatial_compression_ratio)
            || !min_overlap.is_multiple_of(spatial_compression_ratio)
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: tile {tile_size} / overlap {min_overlap} must both be whole \
                 multiples of the {spatial_compression_ratio}x spatial compression ratio — the \
                 decode side indexes the latent as start/ratio and length/ratio, so a misaligned \
                 tile is silently truncated and stitched with overlaps it no longer matches"
            )));
        }
        // The reference's `tile_size >= length` short circuit returns ONE tile spanning the whole
        // axis — `[length]`, not `[tile_size]`. That is what makes tiling inert (bit-identical to
        // an untiled pass) below one tile, rather than padding the canvas out to a full tile.
        if tile_size >= length {
            return Ok(Self {
                starts: vec![0],
                lengths: vec![length],
                overlaps: Vec::new(),
            });
        }
        // The tile-count search is signed in the reference and genuinely goes negative on its first
        // iterations; doing it in `usize` would underflow rather than loop.
        let (len_i, tile_i, ov_i) = (length as i64, tile_size as i64, min_overlap as i64);
        let mut num_tiles = (len_i + tile_i - 1) / tile_i;
        while tile_i * num_tiles - ov_i * (num_tiles - 1) - len_i < 0 {
            num_tiles += 1;
        }
        let mut overlaps = vec![min_overlap; (num_tiles - 1) as usize];
        let remaining =
            tile_i * num_tiles - overlaps.iter().map(|&o| o as i64).sum::<i64>() - len_i;
        for i in 0..(remaining / spatial_compression_ratio as i64) {
            let slot = (i as usize) % overlaps.len();
            overlaps[slot] += spatial_compression_ratio;
        }
        let mut starts = vec![0usize];
        for i in 0..(num_tiles - 1) as usize {
            starts.push(starts[i] + tile_size - overlaps[i]);
        }
        Ok(Self {
            starts,
            lengths: vec![tile_size; num_tiles as usize],
            overlaps,
        })
    }

    /// Tiles on this axis.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    /// Whether the axis is a single untiled span.
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// Whether this plan is a single span covering the axis — i.e. tiling is inert here.
    pub fn is_single_span(&self) -> bool {
        self.starts.len() == 1
    }

    /// The pixel index one past the last tile. Equals the axis length whenever the slack divided
    /// evenly into `spatial_compression_ratio` steps, which it does for every canvas whose sides
    /// are multiples of the ratio — and a decode plan's length is `latent · ratio`, so always.
    pub fn coverage(&self) -> usize {
        match (self.starts.last(), self.lengths.last()) {
            (Some(&s), Some(&l)) => s + l,
            _ => 0,
        }
    }
}

/// Blend and concatenate a grid of tiles back into one tensor — the reference's `_stitch_tiles`.
///
/// The overlaps are in **the units of the tiles themselves**: latent for an encode grid, pixel for
/// a decode grid. Axes are `-2` (height) and `-1` (width).
///
/// Each tile is cross-faded against its up/left neighbour and then has its own trailing overlap
/// trimmed, so every output position is written exactly once. The trim happens *after* the blend
/// and uses this tile's own overlap, not the neighbour's — the two differ whenever the round-robin
/// slack distribution landed unevenly, which it does on any axis with three or more tiles.
pub fn stitch_tiles(
    tiles: &[Vec<Tensor>],
    height_overlaps: &[usize],
    width_overlaps: &[usize],
) -> Result<Tensor> {
    if tiles.is_empty() || tiles[0].is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 tiling: cannot stitch an empty tile grid".into(),
        ));
    }
    if tiles.len() > 1 && height_overlaps.len() + 1 != tiles.len() {
        return Err(CandleError::Msg(format!(
            "minimax-h3 tiling: {} tile rows need {} height overlaps, got {}",
            tiles.len(),
            tiles.len() - 1,
            height_overlaps.len()
        )));
    }
    if tiles[0].len() > 1 && width_overlaps.len() + 1 != tiles[0].len() {
        return Err(CandleError::Msg(format!(
            "minimax-h3 tiling: {} tile columns need {} width overlaps, got {}",
            tiles[0].len(),
            tiles[0].len() - 1,
            width_overlaps.len()
        )));
    }
    let rank = tiles[0][0].dims().len();
    if rank < 2 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 tiling: a tile needs height and width axes, got rank {rank}"
        )));
    }
    let (h_axis, w_axis) = (rank - 2, rank - 1);
    let extent = |o: usize| -> Result<i32> {
        i32::try_from(o).map_err(|_| {
            CandleError::Msg(format!(
                "minimax-h3 tiling: overlap {o} does not fit an i32"
            ))
        })
    };
    let mut result_rows = Vec::with_capacity(tiles.len());
    for (i, row) in tiles.iter().enumerate() {
        let mut result_row = Vec::with_capacity(row.len());
        for (j, tile) in row.iter().enumerate() {
            let mut t = tile.clone();
            if i > 0 {
                t = blend(
                    &tiles[i - 1][j],
                    &t,
                    extent(height_overlaps[i - 1])?,
                    h_axis,
                )?;
            }
            if j > 0 {
                t = blend(&row[j - 1], &t, extent(width_overlaps[j - 1])?, w_axis)?;
            }
            if i < tiles.len() - 1 {
                let n = t.dims()[h_axis];
                t = t.narrow(h_axis, 0, n - height_overlaps[i])?;
            }
            if j < row.len() - 1 {
                let n = t.dims()[w_axis];
                t = t.narrow(w_axis, 0, n - width_overlaps[j])?;
            }
            result_row.push(t.contiguous()?);
        }
        result_rows.push(Tensor::cat(&result_row, w_axis)?);
    }
    Ok(Tensor::cat(&result_rows, h_axis)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The shipped defaults, pinned against the reference's `__init__`.
    #[test]
    fn shipped_defaults_match_the_reference() {
        let t = SpatialTiling::default();
        assert!(t.enabled, "AutoencoderKLMiniMaxH3 sets use_tiling = True");
        assert_eq!((t.tile_height, t.tile_width), (256, 256));
        assert_eq!((t.overlap_height, t.overlap_width), (64, 64));
        // NOTE: these assert the WIRING (that `Default` is the shipped configuration), not the
        // FACT — they are constants comparing against themselves. What pins the fact is
        // `real_weight_tiled_decode_matches_the_official_diffusers_vae`, which reads
        // `use_tiling` / `tile_sample_min_*` back off the loaded reference instance and compares
        // THOSE with these constants.
    }

    #[test]
    fn disable_and_enable_round_trip() {
        let mut t = SpatialTiling::default();
        t.disable();
        assert!(!t.enabled);
        t.enable(None, None, None, None);
        assert_eq!(t, SpatialTiling::default());
        t.enable(Some(64), None, Some(16), None);
        assert_eq!(
            (
                t.tile_height,
                t.tile_width,
                t.overlap_height,
                t.overlap_width
            ),
            (64, 256, 16, 64)
        );
    }

    /// The shipped 1344×768 canvas is a **7×4 = 28-tile** grid — the geometry the defect is live
    /// at. Every number below was produced by running the reference's own `_split_tiles`, not by
    /// this implementation and not by hand.
    ///
    /// The tile count is *derived*, and the naive `ceil(length / tile)` is wrong: 1344 px looks
    /// like six 256 px tiles, but six tiles at the 64 px minimum overlap span only
    /// `6·256 − 5·64 = 1216 < 1344`, so the loop takes a seventh.
    #[test]
    fn shipped_canvas_splits_into_the_reference_grid() {
        let cols = TilePlan::split(1344, 256, 64, 16).unwrap();
        assert_eq!(
            cols.len(),
            7,
            "six tiles cannot span 1344 px at a 64 px overlap"
        );
        assert_eq!(cols.starts, vec![0, 176, 352, 528, 704, 896, 1088]);
        assert_eq!(cols.lengths, vec![256; 7]);
        assert_eq!(cols.overlaps, vec![80, 80, 80, 80, 64, 64]);
        assert_eq!(cols.coverage(), 1344, "the tiles cover the axis exactly");

        let rows = TilePlan::split(768, 256, 64, 16).unwrap();
        assert_eq!(
            rows.len(),
            4,
            "three tiles cannot span 768 px at a 64 px overlap"
        );
        assert_eq!(rows.starts, vec![0, 160, 336, 512]);
        assert_eq!(rows.overlaps, vec![96, 80, 80]);
        assert_eq!(rows.coverage(), 768);

        assert_eq!(
            rows.len() * cols.len(),
            28,
            "the shipped canvas is 28 tiles"
        );
    }

    /// The slack is distributed **round-robin**, so on an axis whose slack is not a whole multiple
    /// of the tile count the overlaps differ from each other. A uniform-overlap simplification
    /// passes every even case and fails here.
    #[test]
    fn round_robin_slack_leaves_uneven_overlaps() {
        let plan = TilePlan::split(672, 256, 64, 16).unwrap();
        assert_eq!(
            plan.overlaps,
            vec![128, 112, 112],
            "slot 0 takes the extra step"
        );
        assert_eq!(plan.starts, vec![0, 128, 272, 416]);
        assert_eq!(plan.coverage(), 672);
        assert_eq!(
            TilePlan::split(512, 256, 64, 16).unwrap().overlaps,
            vec![128, 128]
        );
        assert_eq!(
            TilePlan::split(320, 256, 64, 16).unwrap().overlaps,
            vec![192]
        );
    }

    #[test]
    fn every_overlap_is_latent_aligned() {
        for length in (256..=2048).step_by(32) {
            let plan = TilePlan::split(length, 256, 64, 16).unwrap();
            assert_eq!(plan.coverage(), length, "axis {length} must be covered");
            for (i, &o) in plan.overlaps.iter().enumerate() {
                assert_eq!(
                    o % 16,
                    0,
                    "axis {length} overlap {i} = {o} is not a multiple of 16"
                );
                assert!(
                    o < 256,
                    "axis {length} overlap {i} = {o} is not smaller than the tile"
                );
            }
            for (i, &s) in plan.starts.iter().enumerate() {
                assert_eq!(
                    s % 16,
                    0,
                    "axis {length} start {i} = {s} is not latent-aligned"
                );
            }
        }
    }

    /// Below one tile the plan is a single full-length span, which is what makes tiling inert at
    /// the sub-tile geometries the committed fixtures use.
    #[test]
    fn a_sub_tile_axis_is_one_full_length_span() {
        for length in [16, 64, 240, 255, 256] {
            let plan = TilePlan::split(length, 256, 64, 16).unwrap();
            assert!(plan.is_single_span(), "{length} px must not tile");
            assert_eq!(plan.starts, vec![0]);
            assert_eq!(plan.lengths, vec![length], "the single tile spans the axis");
            assert!(plan.overlaps.is_empty());
        }
        assert!(!TilePlan::split(257, 256, 64, 16).unwrap().is_single_span());
    }

    /// **A tile geometry the latent grid cannot express is refused, not truncated.**
    ///
    /// The plan is in pixels, but `decode_clip_tiled` indexes the latent with `start / ratio` and
    /// `length / ratio`. `enable_tiling` is public API, so a caller can ask for a tile that is not
    /// a whole number of latent cells; before this guard `enable_tiling(Some(100), ..)` at the
    /// shipped ratio 16 decoded 96 px tiles while stitching with 100 px-derived overlaps and
    /// returned a wrong-sized video with no error at all.
    ///
    /// The two clauses are probed **separately** — a case only the tile check catches and a case
    /// only the overlap check catches — so removing either one alone fails this test.
    #[test]
    fn split_rejects_a_tile_geometry_the_latent_grid_cannot_express() {
        // Only the TILE is misaligned (64 is a clean multiple of 16).
        assert!(
            TilePlan::split(768, 100, 64, 16).is_err(),
            "a 100 px tile is 6.25 latent cells"
        );
        // Only the OVERLAP is misaligned (256 is a clean multiple of 16).
        assert!(
            TilePlan::split(768, 256, 72, 16).is_err(),
            "a 72 px overlap is 4.5 latent cells"
        );
        // Aligning both is accepted, so the guard rejects the misalignment and not the canvas.
        assert!(TilePlan::split(768, 96, 64, 16).is_ok());
        assert!(TilePlan::split(768, 256, 80, 16).is_ok());
        // …and the error names the ratio, so the remedy is legible from the message alone.
        let msg = TilePlan::split(768, 100, 64, 16).unwrap_err().to_string();
        assert!(msg.contains("16x spatial compression ratio"), "{msg}");

        // The guard binds BEFORE the sub-tile short circuit: a geometry is valid or not on its own
        // terms, not according to whether this particular canvas happens to reach a second tile.
        assert!(
            TilePlan::split(64, 100, 64, 16).is_err(),
            "a misaligned tile must not be excused by a sub-tile canvas"
        );
    }

    #[test]
    fn split_rejects_degenerate_geometry() {
        assert!(TilePlan::split(0, 256, 64, 16).is_err(), "zero length");
        assert!(TilePlan::split(768, 0, 64, 16).is_err(), "zero tile");
        assert!(TilePlan::split(768, 256, 64, 0).is_err(), "zero ratio");
        assert!(
            TilePlan::split(768, 256, 256, 16).is_err(),
            "an overlap as wide as the tile never advances"
        );
    }

    fn tile(v: f32, h: usize, w: usize) -> Tensor {
        Tensor::from_vec(vec![v; h * w], (1, h, w), &Device::Cpu).unwrap()
    }

    #[test]
    fn stitch_rejects_a_mismatched_overlap_count() {
        let grid = vec![
            vec![tile(0.0, 2, 2), tile(1.0, 2, 2)],
            vec![tile(2.0, 2, 2), tile(3.0, 2, 2)],
        ];
        assert!(stitch_tiles(&grid, &[1], &[1]).is_ok());
        assert!(
            stitch_tiles(&grid, &[], &[1]).is_err(),
            "missing height overlap"
        );
        assert!(
            stitch_tiles(&grid, &[1], &[1, 1]).is_err(),
            "too many width overlaps"
        );
        assert!(stitch_tiles(&[], &[], &[]).is_err(), "empty grid");
    }

    /// A one-tile grid stitches to exactly that tile — no blend, no trim.
    #[test]
    fn a_single_tile_stitches_to_itself() {
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::from_vec(vals.clone(), (1, 3, 4), &Device::Cpu).unwrap();
        let out = stitch_tiles(&[vec![t]], &[], &[]).unwrap();
        assert_eq!(out.dims(), &[1, 3, 4]);
        assert_eq!(out.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vals);
    }

    /// Two identical tiles must stitch back to the same values: a linear cross-fade of a constant
    /// is that constant, so any weighting or trimming error shows up as a non-constant seam.
    #[test]
    fn stitching_identical_tiles_preserves_the_constant() {
        let out = stitch_tiles(&[vec![tile(7.0, 4, 8), tile(7.0, 4, 8)]], &[], &[4]).unwrap();
        assert_eq!(out.dims(), &[1, 4, 12], "8 + 8 - 4");
        for v in out.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert!((v - 7.0).abs() < 1e-6, "seam is not constant: {v}");
        }
    }
}
