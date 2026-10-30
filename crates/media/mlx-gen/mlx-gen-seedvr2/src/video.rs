//! SeedVR2 video-mode orchestration (sc-4814): the MLX budget source over the shared host planner.
//!
//! Chunk planning, the overlap cross-fade, the budget-model selectors and spatial tiling are
//! backend-neutral and live in [`gen_core_family::seedvr2_video`] (sc-27256), shared with the
//! Candle twin; this module re-exports them so `video::*` paths stay stable and adds the MLX budget.

pub use gen_core_family::seedvr2_video::*;

/// The safe peak-GB budget — [`mlx_gen::memory::safe_budget_gib`], which clamps the MLX limit × 0.85
/// by the Metal per-single-buffer cap (`maxBufferLength`, sc-8135) so a budget-sized tile never admits
/// a single allocation that would trip MLX's per-buffer OOM.
pub fn safe_budget_gib() -> f64 {
    mlx_gen::memory::safe_budget_gib()
}

/// Size the temporal chunk against this machine's MLX memory limit (× 0.85). See [`plan_chunk_size_with`].
pub fn plan_chunk_size(weights_bytes: usize, out_h: i32, out_w: i32) -> ChunkPlan {
    plan_chunk_size_with(weights_bytes, out_h, out_w, safe_budget_gib())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_cap_clamp_forces_fitting_tiling_at_4x() {
        // sc-8135: a high-RAM Mac with no user GPU cap — the MLX memory limit defaults to ≈ 1.5× the
        // device working set, so `limit × 0.85` (≈ 127.5 GiB) sits ABOVE the Metal per-single-buffer cap
        // `maxBufferLength` (≈ 80 GiB). Without the clamp that oversized budget sizes a tile whose peak
        // (and thus its single largest allocation) exceeds the cap → MLX `metal::malloc` OOM. The clamp
        // drops the budget to the cap, forcing spatial tiling with tiles that fit under it.
        let weights = (7.3 * GIB) as usize;
        let raw = 150.0 * mlx_gen::memory::SAFE_FRAC; // 1.5×WSS limit × 0.85 on a 128 GB Mac
        let cap = 80.0; // maxBufferLength
        let safe = mlx_gen::memory::clamp_budget_to_cap(raw, Some(cap));
        assert_eq!(safe, cap, "the cap clamps the oversized raw budget");

        // 4096² single frame now exceeds the clamped budget → OverBudget → spatial tiling kicks in.
        assert!(matches!(
            plan_chunk_size_with(weights, 4096, 4096, safe),
            ChunkPlan::OverBudget { .. }
        ));

        // The budget-sized tile's modeled per-frame peak (weights + 8·mpx) fits under the cap.
        let tile = plan_spatial_tile_px(weights, safe);
        let mpx = (tile * tile) as f64 / 1e6;
        let peak = (weights as f64 / GIB) + GB_PER_MPX_FRAME * mpx;
        assert!(
            peak <= cap + 1e-6,
            "tile peak {peak} GiB exceeds cap {cap} GiB"
        );

        // …and a 4096² frame is actually split into multiple tiles (not one full-size pass).
        let overlap = SPATIAL_OVERLAP.min(tile / 2);
        assert!(
            plan_spatial_tiles(4096, 4096, tile, overlap).len() > 1,
            "expected tiling, got a single tile"
        );

        // The sc-8135 *budget* clamp is asserted above (`safe == cap`). At SeedVR2 image scales the
        // sc-8261 VAE-correctness cap (`VAE_SAFE_DECODE_EDGE_PX`) is tighter than the memory-sized tile,
        // so it now bounds the tile edge regardless of the budget — i.e. even the UNCLAMPED raw budget
        // yields a VAE-safe tile (belt-and-suspenders; the budget clamp still governs the OverBudget
        // decision and larger-footprint engines). Verify the cap dominates here.
        let raw_tile = plan_spatial_tile_px(weights, raw);
        assert_eq!(
            raw_tile, VAE_SAFE_DECODE_EDGE_PX,
            "VAE cap should bound the tile even under the unclamped budget"
        );
    }
}
