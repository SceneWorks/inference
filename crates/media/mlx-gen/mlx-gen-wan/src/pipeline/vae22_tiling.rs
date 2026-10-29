//! The Wan2.2 (z48) VAE decode peak model and its budgeted tiling plan.

use super::*;

// ===========================================================================================
// sc-4998 — memory-budgeted z48 vae22 decode tiling
// ===========================================================================================
//
// The dense TI2V-5B is welded to the z48 `vae22` decode (it cannot use the lighter 2.1 z16 VAE),
// and once Lightning makes the DiT trivial that decode is ~95 % of wall-clock. The px-threshold
// [`TilingConfig::auto`] picked **512 px** tiles just below its aggressive cutover — peaking at
// **60 GB** on a routine 1024×576×97 video (OOMs a 64 GB Mac) while the *larger* 1280×704×145
// decode peaked at only 12.6 GB with 256 px tiles: it traded memory the wrong way (non-monotonic).
//
// [`auto_tiling_budgeted`] replaces that with a **free-aware** peak-GB target ([`wan_vae_safe_budget_gib`],
// sc-12737): it picks the *largest* tile whose estimated decode peak stays under the safe budget, so the
// peak is bounded and monotonic in output size, and — being the largest fitting tile — it minimizes the
// overlap-recompute that dominates the aggressive path's wall-clock.
//
// sc-12737 — the budget was `get_memory_limit() × 0.85` (the **total** MLX working-set ceiling), which
// ignored whatever was already resident when the render starts. That mirrors the total-based bug the
// candle Wan tiler fixed (sc-12734): the candle decode over-budgeted on top of the resident denoise
// weights + pool. On unified memory the analogue is `free = get_memory_limit() − get_active_memory()`
// (the soft ceiling minus bytes held by live arrays — e.g. a co-resident model in the video-lane
// residency worker, epic 10975, or a prior render's live tensors). The budget is now `free × 0.85`,
// with the same `WAN_VAE_BUDGET_GIB` env override + accumulator-floor semantics as candle.

/// Bytes of GPU working-set per **output voxel** (`out_f·out_h·out_w`) for the two terms of a z48
/// `vae22` tiled decode, fit from the real-weight `wedge_sweep.rs` anchors (M5 Max):
///   • 1024×576×97 video, 512 px / 64-frame tiles → **60 GB** peak,
///   • 1280×704×145 video, 256 px / 32-frame tiles → **12.6 GB** peak.
/// The peak splits cleanly into a *fixed* full-output term (the output + blend-weight accumulators
/// plus the per-tile pad/add transients, ≈40 B/voxel) and a *per-tile* term that scales with the
/// largest tile's output volume (≈3800 B/voxel through the decoder's 1024-channel stack). With these
/// two constants both anchors reproduce within ~10 % on the conservative side (the model
/// over-estimates slightly — what a guard wants).
pub(super) const VAE22_ACCUM_BYTES_PER_VOXEL: u64 = 40;

pub(super) const VAE22_TILE_BYTES_PER_OUT_VOXEL: u64 = 3_800;

/// Per-tile coefficient for a **bf16** decode (sc-5039). Measured on the same real-weight rig at two
/// tiles of the 1024×576×97 video (cosine 0.99995, no NaN both): 768 px / 64-frame → **79.7 GB**
/// (vs 97.7 GB f32), 640 px / 48-frame → **55.1 GB**. The per-tile term only drops to ~85 % of f32
/// — *not* 50 % — because the `RMS_norm` channel-L2 reduction stays f32 and materializes a full-size
/// f32 temporary of each activation. Calibrated to the **higher** of the two implied coefficients
/// (the 640/48 point) so the estimate never under-shoots a real peak — the 3100 first guess let the
/// selector pick a tile that measured 55.1 GB, just over the 54.4 GB safe line at the 64 GB tier.
/// The fixed accumulator term is unchanged (the blend buffers are f32 either way).
pub(super) const VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16: u64 = 3_400;

/// Estimated concurrent GPU peak (GiB) of a z48 `vae22` decode whose **largest tile** spans
/// `tile_f·tile_h·tile_w` output voxels while assembling a `out_f·out_h·out_w` video. `bf16` selects
/// the lighter per-tile coefficient (sc-5039). Pure (no global state) so it is unit-testable against
/// the `wedge_sweep.rs` anchors. A single-pass decode is the special case `tile_* == out_*`; passing
/// a zero tile yields the accumulator-only floor (the unavoidable cost of holding the output).
///
/// No explicit overflow guard: the voxel products are `i64`→`f64`, and the inputs are bounded upstream
/// by the descriptor's `max_size` (1280 px long edge) and the generated frame count, so
/// `out_f·out_h·out_w` stays ~10¹⁰ — many orders below `i64`/`f64` overflow. The model depends on those
/// upstream caps rather than guarding here (sc-6894 Info).
pub(super) fn estimated_vae22_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
    bf16: bool,
) -> f64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    vae22_decode_peak_bytes(out_f, out_h, out_w, tile_f, tile_h, tile_w, bf16)
        .map_or(f64::INFINITY, |bytes| bytes as f64 / GIB)
}

/// Candidate spatial tile sizes (output px, multiples of the vae22 ×16 spatial scale, overlap 64).
pub(crate) const VAE22_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];

/// Fixed spatial overlap paired with every selected MLX Wan z48 tile edge.
pub const VAE22_SELECTED_OVERLAP: u32 = 64;

/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames (matching the preset
/// overlaps: 24 for the longer tiles, 16/8 for the shorter).
pub(crate) const VAE22_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];

/// Checked byte form of the z48 decode model. Both the live planner and the worker-facing selected
/// profile call this one function, so admission cannot price a different tile than generation runs.
pub(super) fn vae22_decode_peak_bytes(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
    bf16: bool,
) -> Option<u64> {
    let to_u64 = |value: i64| u64::try_from(value).ok();
    let output_voxels = to_u64(out_f)?
        .checked_mul(to_u64(out_h)?)?
        .checked_mul(to_u64(out_w)?)?;
    if output_voxels == 0 {
        return None;
    }
    let tile_voxels = to_u64(tile_f)?
        .checked_mul(to_u64(tile_h)?)?
        .checked_mul(to_u64(tile_w)?)?;
    let tile_coefficient = if bf16 {
        VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16
    } else {
        VAE22_TILE_BYTES_PER_OUT_VOXEL
    };
    VAE22_ACCUM_BYTES_PER_VOXEL
        .checked_mul(output_voxels)?
        .checked_add(tile_coefficient.checked_mul(tile_voxels)?)
}

/// **Memory-budgeted** tiling for the z48 `vae22` decode (sc-4998). Resolves a **free-aware** safe
/// peak-GB ceiling via `wan_vae_safe_budget_gib` (`free × 0.85`, `free = limit − resident`; sc-12737)
/// and returns the *largest* tile that fits — see `plan_vae22_tiling` for the cases and the catchable
/// over-budget error. Caller passes the **output** dimensions (the decoded video size).
pub fn auto_tiling_budgeted(
    height: i32,
    width: i32,
    out_frames: i32,
    bf16: bool,
) -> Result<Option<TilingConfig>> {
    plan_vae22_tiling(height, width, out_frames, wan_vae_safe_budget_gib(), bf16)
}

/// Pure tile selector behind [`auto_tiling_budgeted`] (the `safe_gib` ceiling is injected so this is
/// unit-testable without touching the global memory limit). Returns:
///   • `Ok(None)`    — a single-pass decode already fits `safe_gib` (small/short video); the
///                     existing `decode` path runs, so single-pass is reached **only** when safe.
///   • `Ok(Some(c))` — tiling is required; `c` is the largest tile whose estimated peak ≤ `safe_gib`
///                     (largest ⇒ fewest tiles ⇒ least overlap-recompute ⇒ fastest within budget).
///   • `Err(..)`     — even the smallest candidate tile (or the unavoidable full-output
///                     accumulators) exceeds `safe_gib`: a **catchable** error returned before the
///                     decode, so the caller surfaces it instead of the OS hard-killing the worker
///                     (SIGKILL) or the Metal command buffer aborting (`kIOGPUCommandBufferError…`).
pub(super) fn plan_vae22_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
    bf16: bool,
) -> Result<Option<TilingConfig>> {
    // The selector algorithm now lives in gen-core ([`budgeted_plan`], sc-6894) so the LTX and Wan
    // z16 decodes share it. This wrapper supplies only the **vae22-specific** pieces: the candidate
    // tile grid and the [`estimated_vae22_decode_peak_gib`] cost model (its constants were fit to the
    // z48 `wedge_sweep.rs` anchors and are meaningless for any other VAE/backend, so they stay here),
    // then maps the neutral over-budget signal back to the wan-specific message.
    let candidates = TileCandidates {
        spatial_px: &VAE22_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &VAE22_TEMPORAL_FR,
        temporal_overlap_policy: TemporalOverlapPolicy::Candidate,
    };
    budgeted_plan(
        Wan22Vae::VAE_TILING,
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        |of, oh, ow, tf, th, tw| estimated_vae22_decode_peak_gib(of, oh, ow, tf, th, tw, bf16),
    )
    .map_err(|e| wan_budget_error("z48 vae22", width, height, out_frames, e))
}

/// Resolve a decode plan for one worker-selected spatial cap. The selected edge is the only spatial
/// candidate: the planner may shorten the temporal tile to stay within the same live safe budget as
/// the historical automatic route, but it may neither silently widen nor shrink the admitted edge.
pub(super) fn plan_vae22_tiling_for_selected_spatial(
    height: i32,
    width: i32,
    out_frames: i32,
    tile_edge: i32,
    overlap: i32,
    safe_gib: f64,
) -> Result<TilingConfig> {
    let spatial = [tile_edge];
    let candidates = TileCandidates {
        spatial_px: &spatial,
        spatial_overlap_px: overlap,
        temporal: &VAE22_TEMPORAL_FR,
        temporal_overlap_policy: TemporalOverlapPolicy::Candidate,
    };
    let max_spatial = i64::from(height.max(width));
    let tile_edge_i64 = i64::from(tile_edge);
    let mut plan = budgeted_plan(
        Wan22Vae::VAE_TILING,
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        |of, oh, ow, tf, th, tw| {
            // `budgeted_plan` adds a full-spatial candidate. Above the admitted cap that candidate
            // must never bypass the worker selection, even when the host could afford it.
            if max_spatial > tile_edge_i64 && th == oh && tw == ow {
                f64::INFINITY
            } else {
                estimated_vae22_decode_peak_gib(of, oh, ow, tf, th, tw, true)
            }
        },
    )
    .map_err(|error| wan_budget_error("z48 vae22", width, height, out_frames, error))?
    .unwrap_or_default();

    // Keep the selected carrier observable even when the output is smaller than the cap. Decode
    // clamps it to the output extent, while diagnostics still see the exact admitted parameters.
    plan.spatial = Some(SpatialTiling {
        tile_px: tile_edge,
        overlap_px: overlap,
    });
    Ok(plan)
}

pub(super) fn selected_vae22_plan_with_budget(
    width: u32,
    height: u32,
    frames: u32,
    tile_edge: u32,
    overlap: u32,
    safe_gib: f64,
) -> Result<(TilingConfig, mlx_gen::VideoDecodeMemoryProfile)> {
    let width = i32::try_from(width)
        .map_err(|_| Error::Msg("wan z48 vae22 decode width exceeds i32".into()))?;
    let height = i32::try_from(height)
        .map_err(|_| Error::Msg("wan z48 vae22 decode height exceeds i32".into()))?;
    let frames = i32::try_from(frames)
        .map_err(|_| Error::Msg("wan z48 vae22 decode frame count exceeds i32".into()))?;
    let tile_edge = i32::try_from(tile_edge)
        .map_err(|_| Error::Msg("wan z48 vae22 decode tile edge exceeds i32".into()))?;
    let overlap = i32::try_from(overlap)
        .map_err(|_| Error::Msg("wan z48 vae22 decode overlap exceeds i32".into()))?;
    if width <= 0 || height <= 0 || frames <= 0 || tile_edge <= overlap || overlap < 0 {
        return Err(Error::Msg(
            "wan z48 vae22 decode requires positive geometry and tile_edge > overlap".into(),
        ));
    }

    let tiling = plan_vae22_tiling_for_selected_spatial(
        height, width, frames, tile_edge, overlap, safe_gib,
    )?;
    let bytes = video_decode_peak_bytes_for_vae_and_tiling(
        Wan22Vae::VAE_TILING,
        width as u32,
        height as u32,
        frames as u32,
        Some(&tiling),
    )
    .ok_or_else(|| Error::Msg("wan z48 vae22 decode byte projection overflows u64".into()))?;
    let profile = mlx_gen::VideoDecodeMemoryProfile::new(bytes, 0)
        .expect("zero included decoder bytes always form a valid decode profile");
    Ok((tiling, profile))
}

/// Resolve the exact provider-owned z48 decode carrier and working-set profile for a selected
/// bounded-decode edge. Admission and generation share this live-budgeted entry point.
pub fn selected_vae22_plan(
    width: u32,
    height: u32,
    frames: u32,
    tile_edge: u32,
    overlap: u32,
) -> Result<(TilingConfig, mlx_gen::VideoDecodeMemoryProfile)> {
    selected_vae22_plan_with_budget(
        width,
        height,
        frames,
        tile_edge,
        overlap,
        wan_vae_safe_budget_gib(),
    )
}
