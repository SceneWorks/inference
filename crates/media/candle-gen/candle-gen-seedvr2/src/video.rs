//! SeedVR2 video-mode orchestration (sc-5926): the CUDA/Candle budget source over the shared host
//! planner.
//!
//! Chunk planning, the overlap cross-fade, the budget-model selectors and spatial tiling are
//! backend-neutral and live in [`candle_gen::gen_core::seedvr2_video`] (sc-27256), shared with the
//! MLX twin; this module re-exports them so `video::*` paths stay stable and adds the device-VRAM
//! budget. On CUDA the budget is the device's total VRAM (read from `nvidia-smi`, the SceneWorks
//! worker convention), overridable via `SEEDVR2_BUDGET_GIB`.

pub use candle_gen::gen_core::seedvr2_video::*;

/// Fraction of the device VRAM treated as safe (matches the wan/ltx guard convention).
const SAFE_FRAC: f64 = 0.85;
/// Conservative budget (GiB) when no device VRAM can be read (CPU / sandboxed / no `nvidia-smi`) — small
/// enough to keep the planner on safe small chunks or the per-frame path.
const DEFAULT_BUDGET_GIB: f64 = 16.0;

/// The safe peak-GB budget for the chunk/tile sizers. Resolved in order:
///   1. `SEEDVR2_BUDGET_GIB` env override (a positive float) — the deterministic injection point the
///      worker and tests use to pin a per-GPU budget;
///   2. the device's total VRAM × `SAFE_FRAC` (read via the shared trusted-path `nvidia-smi`
///      probe [`candle_gen::gpu::nvidia_smi_min_total_gib`] — an absolute System32/CUDA_PATH binary,
///      never a bare `PATH` lookup; sc-9014 / F-030);
///   3. `DEFAULT_BUDGET_GIB` when neither is available.
pub fn safe_budget_gib() -> f64 {
    if let Ok(raw) = std::env::var("SEEDVR2_BUDGET_GIB") {
        if let Ok(gib) = raw.trim().parse::<f64>() {
            if gib > 0.0 {
                return gib;
            }
        }
    }
    match candle_gen::gpu::nvidia_smi_min_total_gib() {
        Some(total) => total * SAFE_FRAC,
        None => DEFAULT_BUDGET_GIB,
    }
}

/// Size the temporal chunk against the device VRAM budget. See [`plan_chunk_size_with`].
pub fn plan_chunk_size(weights_bytes: usize, out_h: i32, out_w: i32) -> ChunkPlan {
    plan_chunk_size_with(weights_bytes, out_h, out_w, safe_budget_gib())
}
