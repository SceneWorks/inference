//! S2b — VAE decode **tiling**: split a large latent into overlapping spatial/temporal tiles,
//! decode each independently, and trapezoidally blend the results into the full video. The memory
//! layer over the single-pass [`crate::vae::LtxVideoVae::decode`]; required for production-res /
//! long-video decode. Port of the `mlx_video` reference `models/ltx/video_vae/tiling.py`.
//!
//! This module is the **pure** half: tiling presets, the per-axis interval split (causal-adjusted
//! for time), and the 1-D trapezoidal blend mask. The Array blend loop (decode each tile,
//! pad-and-accumulate into the output/weight buffers, normalize) lives in `vae.rs` so it can reach
//! the decoder. The reference itself allocates full-size `output`+`weights` accumulators and
//! processes one tile at a time, so the pad-and-accumulate form keeps the same bounded peak memory.

/// Spatial scale (VAE: 8× learned upsample × 4× unpatchify) and temporal scale (8× learned).
pub const SPATIAL_SCALE: i32 = 32;
pub const TEMPORAL_SCALE: i32 = 8;

/// Per-frame spatial tiling (tile + overlap in **pixels**; both divisible by 32).
#[derive(Clone, Copy, Debug)]
pub struct SpatialTiling {
    pub tile_px: i32,
    pub overlap_px: i32,
}

/// Temporal tiling (tile + overlap in **frames**; both divisible by 8).
#[derive(Clone, Copy, Debug)]
pub struct TemporalTiling {
    pub tile_frames: i32,
    pub overlap_frames: i32,
}

/// Which axes to tile. `None` on either axis disables tiling there.
#[derive(Clone, Copy, Debug, Default)]
pub struct TilingConfig {
    pub spatial: Option<SpatialTiling>,
    pub temporal: Option<TemporalTiling>,
}

impl TilingConfig {
    /// Reference default: 512 px / 64 px spatial, 64 / 24 frame temporal.
    pub fn default_preset() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 512,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 64,
                overlap_frames: 24,
            }),
        }
    }

    /// Aggressive (smaller tiles, lowest memory): 256/64 px, 32/8 frame.
    pub fn aggressive() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 256,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 32,
                overlap_frames: 8,
            }),
        }
    }

    /// Conservative (larger tiles, faster, less saving): 768/64 px, 96/24 frame.
    pub fn conservative() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 768,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 96,
                overlap_frames: 24,
            }),
        }
    }

    pub fn spatial_only(tile_px: i32, overlap_px: i32) -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px,
                overlap_px,
            }),
            temporal: None,
        }
    }

    pub fn temporal_only(tile_frames: i32, overlap_frames: i32) -> Self {
        Self {
            spatial: None,
            temporal: Some(TemporalTiling {
                tile_frames,
                overlap_frames,
            }),
        }
    }

    /// Auto-select a config from output dimensions (reference `TilingConfig.auto`), or `None` when
    /// no tiling is needed. Mirrors the reference thresholds (spatial > 512, temporal > 65).
    pub fn auto(height: i32, width: i32, num_frames: i32) -> Option<Self> {
        let needs_spatial = height > 512 || width > 512;
        let needs_temporal = num_frames > 65;
        if !needs_spatial && !needs_temporal {
            return None;
        }
        let est_gb = (3.0 * num_frames as f64 * height as f64 * width as f64 * 4.0)
            / (1024.0 * 1024.0 * 1024.0);
        if est_gb > 2.0 || (height * width > 768 * 1024 && num_frames > 100) {
            return Some(Self::aggressive());
        }
        let spatial = needs_spatial.then(|| {
            let max_dim = height.max(width);
            let tile_px = if max_dim > 1024 {
                384
            } else if max_dim > 768 {
                512
            } else {
                384
            };
            SpatialTiling {
                tile_px,
                overlap_px: 64,
            }
        });
        let temporal = needs_temporal.then(|| {
            let (tile_frames, overlap_frames) = if num_frames > 200 {
                (32, 8)
            } else if num_frames > 100 {
                (48, 16)
            } else {
                (64, 24)
            };
            TemporalTiling {
                tile_frames,
                overlap_frames,
            }
        });
        Some(Self { spatial, temporal })
    }

    /// Whether tiling actually fires for a latent of shape `[_, _, f, h, w]`.
    pub fn needs_tiling(&self, f: i32, h: i32, w: i32) -> bool {
        let s = self.spatial.is_some_and(|s| {
            let t = s.tile_px / SPATIAL_SCALE;
            h > t || w > t
        });
        let t = self
            .temporal
            .is_some_and(|tc| f > tc.tile_frames / TEMPORAL_SCALE);
        s || t
    }
}

/// One tile along one axis: latent `[start, end)`, the output `[out_start, out_stop)` it maps to,
/// and the 1-D blend `mask` (length `out_stop − out_start`).
#[derive(Clone, Debug)]
pub struct AxisTile {
    pub start: i32,
    pub end: i32,
    pub out_start: i32,
    pub out_stop: i32,
    pub mask: Vec<f32>,
}

/// `compute_trapezoidal_mask_1d`: ones with a left fade-in (`ramp_left`) and right fade-out
/// (`ramp_right`). `left_from_0` chooses the linspace convention (temporal tiles fade from 0).
pub fn trapezoidal_mask(
    length: i32,
    ramp_left: i32,
    ramp_right: i32,
    left_from_0: bool,
) -> Vec<f32> {
    assert!(length > 0, "mask length must be positive");
    let length = length as usize;
    let ramp_left = ramp_left.clamp(0, length as i32) as usize;
    let ramp_right = ramp_right.clamp(0, length as i32) as usize;
    let mut mask = vec![1.0f32; length];

    if ramp_left > 0 {
        let interval = if left_from_0 {
            ramp_left + 1
        } else {
            ramp_left + 2
        };
        // linspace(0, 1, interval), drop last; if !left_from_0 also drop first.
        let full: Vec<f32> = (0..interval)
            .map(|i| i as f32 / (interval as f32 - 1.0))
            .collect();
        let fade_in: &[f32] = if left_from_0 {
            &full[..interval - 1]
        } else {
            &full[1..interval - 1]
        };
        for i in 0..ramp_left.min(fade_in.len()) {
            mask[i] *= fade_in[i];
        }
    }

    if ramp_right > 0 {
        // fade_out = linspace(1, 0, ramp_right+2)[1:-1] = (ramp_right+1-i)/(ramp_right+1), i=1..ramp_right
        for i in 0..ramp_right {
            let v = (ramp_right as f32 + 1.0 - (i as f32 + 1.0)) / (ramp_right as f32 + 1.0);
            mask[length - ramp_right + i] *= v;
        }
    }

    for v in &mut mask {
        *v = v.clamp(0.0, 1.0);
    }
    mask
}

/// Raw per-axis interval split (`split_in_spatial`): `(starts, ends, left_ramps, right_ramps)`.
fn split_spatial(size: i32, overlap: i32, dim: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    if dim <= size {
        return (vec![0], vec![dim], vec![0], vec![0]);
    }
    let amount = (dim + size - 2 * overlap - 1) / (size - overlap);
    let starts: Vec<i32> = (0..amount).map(|i| i * (size - overlap)).collect();
    let mut ends: Vec<i32> = starts.iter().map(|s| s + size).collect();
    *ends.last_mut().unwrap() = dim;
    let mut left = vec![overlap; amount as usize];
    left[0] = 0;
    let mut right = vec![overlap; amount as usize];
    *right.last_mut().unwrap() = 0;
    (starts, ends, left, right)
}

/// `split_in_temporal`: spatial split, then `starts[1:] -= 1`, `left_ramps[1:] += 1` (causal).
fn split_temporal(size: i32, overlap: i32, dim: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut starts, ends, mut left, right) = split_spatial(size, overlap, dim);
    for i in 1..starts.len() {
        starts[i] -= 1;
        left[i] += 1;
    }
    (starts, ends, left, right)
}

/// Build the spatial-axis tiles (`map_spatial_slice`: out = latent·32, mask via `left_from_0=false`).
fn spatial_tiles(tile_latent: i32, overlap_latent: i32, dim: i32) -> Vec<AxisTile> {
    let (starts, ends, left, right) = split_spatial(tile_latent, overlap_latent, dim);
    starts
        .iter()
        .enumerate()
        .map(|(i, &begin)| {
            let end = ends[i];
            let out_start = begin * SPATIAL_SCALE;
            let out_stop = end * SPATIAL_SCALE;
            let mask = trapezoidal_mask(
                out_stop - out_start,
                left[i] * SPATIAL_SCALE,
                right[i] * SPATIAL_SCALE,
                false,
            );
            AxisTile {
                start: begin,
                end,
                out_start,
                out_stop,
                mask,
            }
        })
        .collect()
}

/// Build the temporal-axis tiles (`map_temporal_slice`: out = 1+(latent−1)·8, mask `left_from_0`).
fn temporal_tiles(tile_latent: i32, overlap_latent: i32, dim: i32) -> Vec<AxisTile> {
    let (starts, ends, left, right) = split_temporal(tile_latent, overlap_latent, dim);
    starts
        .iter()
        .enumerate()
        .map(|(i, &begin)| {
            let end = ends[i];
            let out_start = begin * TEMPORAL_SCALE;
            let out_stop = 1 + (end - 1) * TEMPORAL_SCALE;
            let left_scaled = if left[i] > 0 {
                1 + (left[i] - 1) * TEMPORAL_SCALE
            } else {
                0
            };
            let mask = trapezoidal_mask(
                out_stop - out_start,
                left_scaled,
                right[i] * TEMPORAL_SCALE,
                true,
            );
            AxisTile {
                start: begin,
                end,
                out_start,
                out_stop,
                mask,
            }
        })
        .collect()
}

/// The full tiling plan for a latent `[_, _, f, h, w]`: per-axis tile lists + the output dims.
pub struct TilePlan {
    pub t: Vec<AxisTile>,
    pub h: Vec<AxisTile>,
    pub w: Vec<AxisTile>,
    pub out_f: i32,
    pub out_h: i32,
    pub out_w: i32,
}

impl TilingConfig {
    /// Build the [`TilePlan`] for a latent of shape `[_, _, f, h, w]`.
    pub fn plan(&self, f: i32, h: i32, w: i32) -> TilePlan {
        let (t_tile, t_over) = match self.temporal {
            Some(tc) => (
                tc.tile_frames / TEMPORAL_SCALE,
                tc.overlap_frames / TEMPORAL_SCALE,
            ),
            None => (f, 0),
        };
        let (s_tile, s_over) = match self.spatial {
            Some(sc) => (sc.tile_px / SPATIAL_SCALE, sc.overlap_px / SPATIAL_SCALE),
            None => (h.max(w), 0),
        };
        TilePlan {
            t: temporal_tiles(t_tile, t_over, f),
            h: spatial_tiles(s_tile, s_over, h),
            w: spatial_tiles(s_tile, s_over, w),
            out_f: 1 + (f - 1) * TEMPORAL_SCALE,
            out_h: h * SPATIAL_SCALE,
            out_w: w * SPATIAL_SCALE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoid_no_ramp_is_all_ones() {
        assert_eq!(trapezoidal_mask(4, 0, 0, false), vec![1.0; 4]);
    }

    #[test]
    fn trapezoid_right_fade_out() {
        // ramp_right=2: last two = (3-1)/3, (3-2)/3 = 2/3, 1/3.
        let m = trapezoidal_mask(5, 0, 2, false);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[2], 1.0);
        assert!((m[3] - 2.0 / 3.0).abs() < 1e-6);
        assert!((m[4] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn trapezoid_left_from_0_fade_in() {
        // ramp_left=2, left_from_0: linspace(0,1,3)[:-1] = [0, 0.5].
        let m = trapezoidal_mask(5, 2, 0, true);
        assert!((m[0] - 0.0).abs() < 1e-6);
        assert!((m[1] - 0.5).abs() < 1e-6);
        assert_eq!(m[2], 1.0);
    }

    #[test]
    fn spatial_split_three_tiles() {
        // tile=2, overlap=1, dim=4 → amount=(4+2-2-1)/1=3.
        let (starts, ends, left, right) = split_spatial(2, 1, 4);
        assert_eq!(starts, vec![0, 1, 2]);
        assert_eq!(ends, vec![2, 3, 4]);
        assert_eq!(left, vec![0, 1, 1]);
        assert_eq!(right, vec![1, 1, 0]);
    }

    #[test]
    fn temporal_split_causal_adjust() {
        // tile=2, overlap=1, dim=3 → spatial(2,1,3): amount=(3+2-2-1)/1=2, starts=[0,1].
        // temporal: starts[1]-=1 → [0,0], left[1]+=1.
        let (starts, _ends, left, _right) = split_temporal(2, 1, 3);
        assert_eq!(starts, vec![0, 0]);
        assert_eq!(left, vec![0, 2]);
    }

    #[test]
    fn needs_tiling_thresholds() {
        let cfg = TilingConfig::spatial_only(64, 32); // tile = 2 latent
        assert!(cfg.needs_tiling(1, 4, 4)); // h=4 > 2
        assert!(!cfg.needs_tiling(10, 2, 2)); // h=w=2 not > 2, no temporal cfg
        let tc = TilingConfig::temporal_only(16, 8); // tile = 2 latent
        assert!(tc.needs_tiling(3, 2, 2)); // f=3 > 2
        assert!(!tc.needs_tiling(2, 99, 99)); // f=2 not > 2, no spatial cfg
    }
}
