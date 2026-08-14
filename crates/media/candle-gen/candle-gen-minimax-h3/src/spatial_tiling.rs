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

/// The streaming, scratch-bounded equivalent of [`stitch_tiles`] — ladder rung 2 (sc-18660).
///
/// The Candle twin of `mlx_gen_minimax_h3::spatial_tiling::BoundedStitch`, here for the same
/// reason: the tile **geometry** is pinned by output correctness (sc-18786) and cannot be a memory
/// lever, so what rung 2 bounds at this seam is the number of decoded tiles held live.
/// [`stitch_tiles`] takes the whole `rows x cols` grid by reference; this consumes tiles row-major
/// and retains only the trailing windows [`blend`] will actually read — `O(cols)` strips.
///
/// # Bit-identical, and why
///
/// [`blend`] reads exactly `a.narrow(axis, a_len - extent, extent)` of its first argument. A strip
/// that *is* that window yields the same `a_ov`, and its shorter `a_len` cannot change `extent`,
/// which is already clamped to the strip's own length by construction.
///
/// # Candle storage sharing
///
/// Every retained strip is `contiguous()`, so it owns its bytes and the parent tile's storage is
/// released when the tile drops. A bare `narrow` is a view over the parent's `Arc` storage, which
/// would leave the bound structural only — the same class of mistake as `released_bytes` reporting
/// frees that did not happen.
pub struct BoundedStitch {
    rows: usize,
    cols: usize,
    height_overlaps: Vec<usize>,
    width_overlaps: Vec<usize>,
    /// Trailing height strips of the previous row's original tiles, indexed by column.
    carry: Vec<Option<Tensor>>,
    /// Trailing width strip of the current row's previous original tile.
    left: Option<Tensor>,
    result_row: Vec<Tensor>,
    result_rows: Vec<Tensor>,
    seen: usize,
    rank: Option<usize>,
}

impl BoundedStitch {
    /// Start a stitch over a `rows` x `cols` grid at the given per-seam overlaps.
    pub fn new(
        rows: usize,
        cols: usize,
        height_overlaps: &[usize],
        width_overlaps: &[usize],
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 tiling: cannot stitch an empty tile grid".into(),
            ));
        }
        if rows > 1 && height_overlaps.len() + 1 != rows {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: {rows} tile rows need {} height overlaps, got {}",
                rows - 1,
                height_overlaps.len()
            )));
        }
        if cols > 1 && width_overlaps.len() + 1 != cols {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: {cols} tile columns need {} width overlaps, got {}",
                cols - 1,
                width_overlaps.len()
            )));
        }
        Ok(Self {
            rows,
            cols,
            height_overlaps: height_overlaps.to_vec(),
            width_overlaps: width_overlaps.to_vec(),
            carry: (0..cols).map(|_| None).collect(),
            left: None,
            result_row: Vec::with_capacity(cols),
            result_rows: Vec::with_capacity(rows),
            seen: 0,
            rank: None,
        })
    }

    /// Elements retained as **blend scratch** — the quantity rung 2 bounds. Exact and weights-free;
    /// under [`stitch_tiles`] the equivalent is the whole undecoded-yet-unconsumed grid.
    pub fn scratch_elements(&self) -> usize {
        let count = |t: &Tensor| t.dims().iter().product::<usize>();
        self.carry.iter().flatten().map(count).sum::<usize>() + self.left.as_ref().map_or(0, count)
    }

    /// Feed the next tile, in row-major order.
    pub fn push(&mut self, tile: Tensor) -> Result<()> {
        if self.seen >= self.rows * self.cols {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: pushed more than the {} tiles this {}x{} grid declared",
                self.rows * self.cols,
                self.rows,
                self.cols
            )));
        }
        let rank = tile.dims().len();
        if rank < 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: a tile needs height and width axes, got rank {rank}"
            )));
        }
        match self.rank {
            None => self.rank = Some(rank),
            Some(first) if first != rank => {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 tiling: tile rank {rank} does not match the grid's {first}"
                )));
            }
            Some(_) => {}
        }
        let (h_axis, w_axis) = (rank - 2, rank - 1);
        let (i, j) = (self.seen / self.cols, self.seen % self.cols);
        let extent = |o: usize| -> Result<i32> {
            i32::try_from(o).map_err(|_| {
                CandleError::Msg(format!(
                    "minimax-h3 tiling: overlap {o} does not fit an i32"
                ))
            })
        };

        // Capture the trailing windows from the ORIGINAL tile, before either blend rewrites it.
        let next_left = if j + 1 < self.cols {
            Some(Self::trailing(&tile, w_axis, self.width_overlaps[j])?)
        } else {
            None
        };
        let next_carry = if i + 1 < self.rows {
            Some(Self::trailing(&tile, h_axis, self.height_overlaps[i])?)
        } else {
            None
        };

        let mut t = tile;
        if i > 0 {
            let above = self.carry[j].as_ref().ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 tiling: row {i} column {j} has no retained strip from the row above"
                ))
            })?;
            t = blend(above, &t, extent(self.height_overlaps[i - 1])?, h_axis)?;
        }
        if j > 0 {
            let left = self.left.as_ref().ok_or_else(|| {
                CandleError::Msg(format!(
                    "minimax-h3 tiling: row {i} column {j} has no retained strip from its left \
                     neighbour"
                ))
            })?;
            t = blend(left, &t, extent(self.width_overlaps[j - 1])?, w_axis)?;
        }
        // The trim uses THIS tile's own overlap, not the neighbour's — the two differ whenever the
        // round-robin slack distribution landed unevenly.
        if i + 1 < self.rows {
            let n = t.dims()[h_axis];
            t = t.narrow(h_axis, 0, n - self.height_overlaps[i])?;
        }
        if j + 1 < self.cols {
            let n = t.dims()[w_axis];
            t = t.narrow(w_axis, 0, n - self.width_overlaps[j])?;
        }

        self.left = next_left;
        if i + 1 < self.rows {
            self.carry[j] = next_carry;
        }
        self.result_row.push(t.contiguous()?);
        self.seen += 1;

        if j + 1 == self.cols {
            let row = std::mem::take(&mut self.result_row);
            self.result_rows.push(Tensor::cat(&row, w_axis)?);
            self.left = None;
        }
        Ok(())
    }

    /// The trailing `extent` slice along `axis`, materialized so the parent can be released.
    fn trailing(tile: &Tensor, axis: usize, extent: usize) -> Result<Tensor> {
        let len = tile.dims()[axis];
        let extent = extent.min(len);
        Ok(tile.narrow(axis, len - extent, extent)?.contiguous()?)
    }

    /// Concatenate the completed rows. Fails if fewer tiles were pushed than declared.
    pub fn finish(self) -> Result<Tensor> {
        let expected = self.rows * self.cols;
        if self.seen != expected {
            return Err(CandleError::Msg(format!(
                "minimax-h3 tiling: {}x{} grid expected {expected} tiles, got {}",
                self.rows, self.cols, self.seen
            )));
        }
        let rank = self.rank.expect("a complete grid pushed at least one tile");
        Ok(Tensor::cat(&self.result_rows, rank - 2)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// A deterministic non-constant tile grid: a constant grid cannot see a weighting error, and a
    /// separable one cannot see a transposed axis.
    fn grid(rows: usize, cols: usize, shape: &[usize]) -> Vec<Vec<Tensor>> {
        let n: usize = shape.iter().product();
        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| {
                        let base = (i * 31 + j * 7) as f32;
                        let vals: Vec<f32> = (0..n)
                            .map(|k| base + (k as f32) * 0.5 + ((k * k) % 13) as f32)
                            .collect();
                        Tensor::from_vec(vals, shape, &Device::Cpu).unwrap()
                    })
                    .collect()
            })
            .collect()
    }

    /// **The bounded stitch is BIT-identical to the full-grid stitch** — the guard that rung 2's
    /// scratch bound did not move the output.
    #[test]
    fn bounded_stitch_matches_the_full_grid_stitch() {
        for (rows, cols, ho, wo) in [
            (1usize, 1usize, vec![], vec![]),
            (1, 3, vec![], vec![2usize, 1]),
            (3, 1, vec![2usize, 1], vec![]),
            (2, 2, vec![1usize], vec![1usize]),
            (4, 7, vec![3usize, 2, 2], vec![2usize, 2, 1, 2, 1, 2]),
        ] {
            let tiles = grid(rows, cols, &[1, 2, 6, 5]);
            let full = stitch_tiles(&tiles, &ho, &wo).unwrap();
            let mut s = BoundedStitch::new(rows, cols, &ho, &wo).unwrap();
            for row in &tiles {
                for tile in row {
                    s.push(tile.clone()).unwrap();
                }
            }
            let bounded = s.finish().unwrap();
            assert_eq!(bounded.dims(), full.dims(), "{rows}x{cols}: shape moved");
            let a = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = bounded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let peak = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                peak, 0.0,
                "{rows}x{cols}: bounded stitch differs by {peak:.3e}; it must be bit-identical"
            );
        }
    }

    /// **The bound is `O(cols)`**: retained blend scratch must not grow with the row count.
    #[test]
    fn retained_scratch_is_bounded_by_one_row_of_strips() {
        let shape = [1usize, 2, 16, 16];
        let tile_elems: usize = shape.iter().product();
        let (rows, cols) = (4usize, 7usize);
        let (ho, wo) = (vec![6usize, 5, 5], vec![5usize, 5, 5, 5, 4, 4]);

        let worst = |rows: usize, ho: &[usize]| {
            let tiles = grid(rows, cols, &shape);
            let mut s = BoundedStitch::new(rows, cols, ho, &wo).unwrap();
            let mut worst = 0usize;
            for row in &tiles {
                for tile in row {
                    s.push(tile.clone()).unwrap();
                    worst = worst.max(s.scratch_elements());
                }
            }
            s.finish().unwrap();
            worst
        };

        // The derived ceiling contains **no `rows` term** — that absence is the property under test.
        let strip = |overlap: usize| tile_elems / 16 * overlap;
        let ceiling = cols * strip(*ho.iter().max().unwrap()) + strip(*wo.iter().max().unwrap());
        let short = worst(rows, &ho);
        assert!(
            short <= ceiling,
            "retained scratch {short} exceeded the O(cols) ceiling {ceiling}"
        );
        assert!(
            short * 4 < rows * cols * tile_elems,
            "retained scratch {short} is not a material bound on the grid `stitch_tiles` holds"
        );
        let tall_ho = vec![6usize, 5, 5, 6, 5, 5, 6];
        assert_eq!(tall_ho.iter().max(), ho.iter().max());
        assert!(
            worst(8, &tall_ho) <= ceiling,
            "doubling the rows pushed scratch over the row-independent ceiling {ceiling}"
        );
    }

    /// The arity checks are the full-grid stitcher's, moved to construction time.
    #[test]
    fn bounded_stitch_rejects_a_mismatched_overlap_count() {
        assert!(BoundedStitch::new(2, 2, &[1], &[1]).is_ok());
        assert!(BoundedStitch::new(2, 2, &[], &[1]).is_err());
        assert!(BoundedStitch::new(2, 2, &[1], &[1, 1]).is_err());
        assert!(BoundedStitch::new(0, 2, &[], &[1]).is_err());
        // An incomplete grid is a typed error, not a wrong-sized tensor.
        assert!(BoundedStitch::new(2, 2, &[1], &[1])
            .unwrap()
            .finish()
            .is_err());
    }

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
