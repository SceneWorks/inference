//! The Wan2.1 (z16) VAE decode / encode peak models and its budgeted tiling plan.

use super::*;

// --- z16 Wan 2.1 VAE decode budgeting (sc-6894 F-009) ---------------------------------------------
//
// The 14B T2V/I2V + VACE decode paths previously used the unbudgeted px-threshold `TilingConfig::auto`
// on the largest-resident models — the same OOM-prone selector the z48 path replaced (sc-4998). These
// route the z16 decode through the shared `budgeted_plan` selector with a z16-specific cost model fit
// from the real `vae16_decode_sweep.rs` anchors (the z16 decoder is non-causal time ×4, spatial ×8).

/// Per-output-voxel cost of the z16 decode's full-output f32 accumulators (`output` [1,3,F,H,W] +
/// `weights` [1,1,F,H,W]) — paid by every tiled plan. Isolated from the `vae16_decode_sweep.rs`
/// anchors (128 GB M-series, f32): the 768²×16 single-pass peak (56.35 GB) minus the same output tiled
/// @384 px (14.46 GB) pins this term at ~57 B/voxel; rounded **up** to 64 for headroom (the model must
/// never under-predict — an under-shoot is an OOM, an over-shoot only tiles slightly more).
pub(super) const VAE16_ACCUM_BYTES_PER_VOXEL: u64 = 64;

/// Per-tile-output-voxel cost of the z16 decoder working set (conv stack + ×8 spatial / ×4 temporal
/// upsample). Fit from the same anchors at ~6355 B/voxel (≈1.7× the z48 `vae22`'s 3800 — the bigger
/// spatial upsample); rounded **up** to 6500. z16 decodes f32 in production, so there is no bf16
/// coefficient (unlike `vae22`, sc-5039).
pub(super) const VAE16_TILE_BYTES_PER_OUT_VOXEL: u64 = 6_500;

/// Candidate spatial tile sizes (output px, multiples of the z16 ×8 spatial scale, overlap 64).
pub(super) const VAE16_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];

/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
pub(super) const VAE16_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];

/// Estimated concurrent GPU peak (GiB) of a z16 decode whose largest tile spans `tile_*` output voxels
/// while assembling an `out_*` video. Pure (no global state) → unit-testable against the
/// `vae16_decode_sweep.rs` anchors. Single-pass is the special case `tile_* == out_*`; a zero tile is
/// the accumulator-only floor.
pub(super) fn estimated_z16_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (VAE16_ACCUM_BYTES_PER_VOXEL as f64 * out_voxels
        + VAE16_TILE_BYTES_PER_OUT_VOXEL as f64 * tile_voxels)
        / GIB
}

/// Evaluate the calibrated Wan VAE decode cost for one output and optional tiling configuration.
///
/// Tile extents are capped at the output on every axis, exactly as the budget planner sizes a
/// candidate. All coefficients are non-negative, so passing `None` is a conservative upper bound for
/// every valid tiled configuration. The z48 path uses its shipped bf16 coefficient.
#[doc(hidden)]
pub fn video_decode_peak_bytes_for_vae_and_tiling(
    vae: VaeTiling,
    width: u32,
    height: u32,
    frames: u32,
    tiling: Option<&TilingConfig>,
) -> Option<u64> {
    let width = u64::from(width);
    let height = u64::from(height);
    let frames = u64::from(frames);
    let output_voxels = width.checked_mul(height)?.checked_mul(frames)?;
    if output_voxels == 0 {
        return None;
    }
    if vae != WanVae::VAE_TILING && vae != Wan22Vae::VAE_TILING {
        return None;
    }

    let positive_extent = |configured: i32, full: u64| {
        let configured = u64::try_from(configured).ok()?;
        (configured > 0).then_some(configured.min(full))
    };
    let tile_width = tiling
        .and_then(|cfg| cfg.spatial)
        .map_or(Some(width), |spatial| {
            positive_extent(spatial.tile_px, width)
        })?;
    let tile_height = tiling
        .and_then(|cfg| cfg.spatial)
        .map_or(Some(height), |spatial| {
            positive_extent(spatial.tile_px, height)
        })?;
    let tile_frames = tiling
        .and_then(|cfg| cfg.temporal)
        .map_or(Some(frames), |temporal| {
            positive_extent(temporal.tile_frames, frames)
        })?;
    if vae == Wan22Vae::VAE_TILING {
        return vae22_decode_peak_bytes(
            i64::try_from(frames).ok()?,
            i64::try_from(height).ok()?,
            i64::try_from(width).ok()?,
            i64::try_from(tile_frames).ok()?,
            i64::try_from(tile_height).ok()?,
            i64::try_from(tile_width).ok()?,
            true,
        );
    }

    let tile_voxels = tile_width
        .checked_mul(tile_height)?
        .checked_mul(tile_frames)?;
    VAE16_ACCUM_BYTES_PER_VOXEL
        .checked_mul(output_voxels)?
        .checked_add(VAE16_TILE_BYTES_PER_OUT_VOXEL.checked_mul(tile_voxels)?)
}

/// Conservative single-pass Wan VAE decode working-set peak in bytes for a concrete VAE geometry.
///
/// This is the full-output case of the calibrated cost function used by the actual z16/z48 budget
/// planners. Because [`video_decode_peak_bytes_for_vae_and_tiling`] caps every configured tile at the
/// output extent and the cost is monotone, this also bounds providers that use a different valid
/// tiling selector over the same concrete VAE. It excludes DiT/text-encoder weights.
pub fn conservative_video_decode_peak_bytes_for_vae(
    vae: VaeTiling,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<u64> {
    video_decode_peak_bytes_for_vae_and_tiling(vae, width, height, frames, None)
}

// --- sc-20686 (epic E8): Wan VAE encode working set -------------------------------------------
//
// Both Wan encoders are chunked causal encoders (a 1-frame first chunk, then 4-frame chunks) that
// release their dead buffers at every block, so the live encode peak is one chunk's working set plus
// a small per-input-frame term (the input clip and the growing latent), and MLX's cache stays empty
// (live + cache == live). There is no tiled encode: the chunking already bounds time, and space is
// priced whole. Fit from `encode_footprint_harness` (production-width synthetic weights, f32 — the
// dtype production encodes in for both VAEs) at 64x128 and 128x128, T = 1/5/9/13; every coefficient
// is rounded up so each measured point is covered:
//   z16: T=1 6,284/6,246 B/px; T>=5 (peak - 700*T*HW) / 4HW <= 4,518.
//   z48: T=1 2,843/2,828 B/px; T>=5 (peak - 50*T*HW) / 4HW <= 2,330.
/// z16 encode: bytes per input pixel of a single-frame encode.
pub(super) const VAE16_ENCODE_SINGLE_FRAME_BYTES_PER_PIXEL: u64 = 6_300;

/// z16 encode: bytes per pixel of one 4-frame chunk.
pub(super) const VAE16_ENCODE_CHUNK_BYTES_PER_VOXEL: u64 = 4_600;

/// z16 encode: bytes per input voxel of the whole clip (input + accumulated latent).
pub(super) const VAE16_ENCODE_CLIP_BYTES_PER_VOXEL: u64 = 700;

/// z48 encode: bytes per input pixel of a single-frame encode.
pub(super) const VAE22_ENCODE_SINGLE_FRAME_BYTES_PER_PIXEL: u64 = 2_900;

/// z48 encode: bytes per voxel of one 4-frame chunk.
pub(super) const VAE22_ENCODE_CHUNK_BYTES_PER_VOXEL: u64 = 2_400;

/// z48 encode: bytes per input voxel of the whole clip.
pub(super) const VAE22_ENCODE_CLIP_BYTES_PER_VOXEL: u64 = 50;

/// Frames in every chunk after the first (the encoders' temporal stride).
pub(super) const VAE_ENCODE_CHUNK_FRAMES: u64 = 4;

/// Conservative live working set (bytes) of one Wan VAE encode of a `frames x height x width` clip
/// (`frames = 1` for a still). Excludes the VAE weights. `None` for a non-Wan VAE or overflow.
pub fn video_encode_peak_bytes_for_vae(
    vae: VaeTiling,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<u64> {
    let (single, chunk, clip) = if vae == WanVae::VAE_TILING {
        (
            VAE16_ENCODE_SINGLE_FRAME_BYTES_PER_PIXEL,
            VAE16_ENCODE_CHUNK_BYTES_PER_VOXEL,
            VAE16_ENCODE_CLIP_BYTES_PER_VOXEL,
        )
    } else if vae == Wan22Vae::VAE_TILING {
        (
            VAE22_ENCODE_SINGLE_FRAME_BYTES_PER_PIXEL,
            VAE22_ENCODE_CHUNK_BYTES_PER_VOXEL,
            VAE22_ENCODE_CLIP_BYTES_PER_VOXEL,
        )
    } else {
        return None;
    };
    let pixels = u64::from(width).checked_mul(u64::from(height))?;
    match frames {
        0 => None,
        1 => single.checked_mul(pixels),
        frames => chunk
            .checked_mul(VAE_ENCODE_CHUNK_FRAMES)?
            .checked_mul(pixels)?
            .max(single.checked_mul(pixels)?)
            .checked_add(clip.checked_mul(u64::from(frames))?.checked_mul(pixels)?),
    }
}

/// **Memory-budgeted** tiling for the z16 Wan 2.1 VAE decode (sc-6894 F-009): the z16 analogue of
/// [`auto_tiling_budgeted`], routing the shared [`budgeted_plan`] selector through the z16 cost model.
/// Replaces the unbudgeted [`TilingConfig::auto`] on the 14B T2V/I2V + VACE decode paths. Uses the same
/// **free-aware** budget as the z48 path (`wan_vae_safe_budget_gib`, sc-12737). Caller passes the
/// **output** dims (the decoded video size).
pub fn auto_tiling_budgeted_z16(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_z16_tiling(height, width, out_frames, wan_vae_safe_budget_gib())
}

/// The z16 selector used by Krea Realtime and SCAIL-2. It shares Wan's candidate grid, peak model,
/// and free-aware budget, but deliberately retains sc-15325's quality-oriented half-tile overlap.
/// Wan's own product providers use [`auto_tiling_budgeted_z16`], which restores the measured
/// candidate overlap after sc-15445 found the half-tile policy materially slower for a marginal gain.
pub fn auto_tiling_budgeted_z16_quality_overlap(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    plan_z16_tiling_with_overlap(
        height,
        width,
        out_frames,
        wan_vae_safe_budget_gib(),
        TemporalOverlapPolicy::HalfTile,
    )
}

/// Pure z16 tile selector behind [`auto_tiling_budgeted_z16`] (the `safe_gib` ceiling is injected so it
/// is unit-testable without touching the global memory limit). Supplies the z16 cost model + candidate
/// grid to the shared [`budgeted_plan`]; same `Ok(None)` / `Ok(Some)` / catchable-`Err` contract as
/// [`plan_vae22_tiling`].
pub(super) fn plan_z16_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    plan_z16_tiling_with_overlap(
        height,
        width,
        out_frames,
        safe_gib,
        TemporalOverlapPolicy::Candidate,
    )
}

pub(super) fn plan_z16_tiling_with_overlap(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
    temporal_overlap_policy: TemporalOverlapPolicy,
) -> Result<Option<TilingConfig>> {
    let candidates = TileCandidates {
        spatial_px: &VAE16_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &VAE16_TEMPORAL_FR,
        temporal_overlap_policy,
    };
    budgeted_plan(
        WanVae::VAE_TILING,
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_z16_decode_peak_gib,
    )
    .map_err(|e| wan_budget_error("z16 vae", width, height, out_frames, e))
}
