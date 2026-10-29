//! The free-memory-aware VAE decode budget shared by both VAE tiling planners.

use super::*;

// --- sc-12737: free-aware VAE-decode budget (contract-aligned with candle sc-12734) --------------
//
// Both Wan VAE-decode tilers (`auto_tiling_budgeted` z48 + `auto_tiling_budgeted_z16`) resolve their
// safe peak-GB ceiling through [`wan_vae_safe_budget_gib`], the MLX analogue of candle-gen's
// `vae_tiling::free_aware_safe_budget_gib`. Same three-step precedence, the same `WAN_VAE_BUDGET_GIB`
// env knob, and the same `free × 0.85` framing — so one operator contract governs the Wan VAE decode
// budget on both backends. The accumulator floor stays in the shared gen-core [`budgeted_plan`]
// selector (its step-2 `AccumulatorsExceedBudget`): only the *budget value* fed to it changed, so
// decode geometry / blend / output parity are untouched.

/// Env override for the Wan VAE-decode budget (GiB, positive float). The deterministic injection point
/// for the worker/tests, and **the same knob the candle Wan tiler honors** (`candle-gen-wan`'s
/// `WAN_VAE_BUDGET_GIB`, sc-12734): set `WAN_VAE_BUDGET_GIB=N` to pin the Wan VAE-decode budget to N GiB
/// regardless of backend. Wins over the live free-memory probe.
pub(super) const WAN_VAE_BUDGET_ENV: &str = "WAN_VAE_BUDGET_GIB";

/// Fraction of **free** unified memory a decode may target (headroom for allocator churn + the OS).
/// Matches the 0.85 the mlx denoise guard and candle's free-aware Wan tiler both use.
pub(super) const WAN_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;

/// Conservative fallback budget, used only when the MLX memory limit is disabled (`0`) so no free
/// figure can be derived. Mirrors candle's `WAN22_VAE_DEFAULT_BUDGET_GIB` (16 GiB, the smallest shipped
/// Apple-silicon tier). In practice unreachable — MLX's default limit is 1.5× the recommended working
/// set, never 0 unless a caller explicitly disabled it via `set_memory_limit(0)`.
pub(super) const WAN_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

/// Live **free** unified memory in GiB — the MLX analogue of candle's `nvidia-smi memory.free`
/// (`total − used`). On unified memory "free" is the MLX memory limit (the soft working-set ceiling,
/// [`get_memory_limit`]) MINUS the bytes held by live arrays ([`get_active_memory`]) — i.e.
/// `limit − currently_resident`. The reclaimable buffer **cache** (`get_cache_memory`) is deliberately
/// NOT subtracted: MLX auto-reclaims it under allocation pressure, so it is genuinely available to the
/// decode (subtracting it would over-tile with no safety gain). `None` when the limit is disabled (`0`)
/// — there is no ceiling to subtract from — so the resolver falls back to the conservative default.
pub(super) fn live_free_gib() -> Option<f64> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let limit = get_memory_limit();
    if limit == 0 {
        return None;
    }
    let active = get_active_memory();
    Some(limit.saturating_sub(active) as f64 / GIB)
}

/// Pure `free × safe_frac` (clamped at 0). Split out so the free-aware budget arithmetic is
/// unit-testable with an injected `free_gib` (`= limit − resident`) and no live probe — the
/// "N GiB artificially resident ⇒ smaller budget" seam the acceptance criteria call for. Mirrors
/// candle-gen's `vae_tiling::free_aware_budget_gib`.
pub(super) fn free_aware_budget_gib(free_gib: f64, safe_frac: f64) -> f64 {
    free_gib.max(0.0) * safe_frac
}

/// Core of [`wan_vae_safe_budget_gib`] with the free-memory probe injected, so the env-override
/// precedence + the probe→default fallback are unit-testable without touching the global memory limit.
/// The live entry point passes [`live_free_gib`]; tests pass a stub closure. Mirrors candle-gen's
/// `vae_tiling::resolve_free_aware_budget`.
pub(super) fn resolve_free_aware_budget(
    env_var: &str,
    safe_frac: f64,
    default_gib: f64,
    free_probe: impl Fn() -> Option<f64>,
) -> f64 {
    if let Ok(raw) = std::env::var(env_var) {
        if let Ok(gib) = raw.trim().parse::<f64>() {
            if gib > 0.0 {
                return gib;
            }
        }
    }
    match free_probe() {
        Some(free) => free_aware_budget_gib(free, safe_frac),
        None => default_gib,
    }
}

/// The **free-aware** safe peak-GiB budget shared by both Wan VAE-decode tilers (sc-12737), aligning
/// the mlx contract with candle's sc-12734 free-aware resolver. Resolved in order (mirrors candle-gen's
/// `vae_tiling::free_aware_safe_budget_gib`):
///  1. the `WAN_VAE_BUDGET_GIB` env override (a positive float — the deterministic worker/test knob,
///     shared with candle);
///  2. `free × WAN_VAE_BUDGET_SAFE_FRAC` via the live [`live_free_gib`] probe (`limit − resident`);
///  3. `WAN_VAE_DEFAULT_BUDGET_GIB` when the limit is disabled.
///
/// This replaces the previous `get_memory_limit() × 0.85`, which budgeted against **total** and so
/// ignored whatever was already resident when the render starts (a co-resident model in the video-lane
/// residency worker, epic 10975, or a prior render's live tensors) — the same over-budgeting the candle
/// side fixed.
pub(super) fn wan_vae_safe_budget_gib() -> f64 {
    resolve_free_aware_budget(
        WAN_VAE_BUDGET_ENV,
        WAN_VAE_BUDGET_SAFE_FRAC,
        WAN_VAE_DEFAULT_BUDGET_GIB,
        live_free_gib,
    )
}

/// Map gen-core's neutral [`TilingBudgetError`] to a wan-facing message tagged with the VAE `label`
/// (e.g. `"z48 vae22"`, `"z16 vae"`). Shared by the per-VAE budgeted-tiling wrappers so the catchable
/// over-budget wording stays identical across them.
pub(super) fn wan_budget_error(
    label: &str,
    width: i32,
    height: i32,
    out_frames: i32,
    e: TilingBudgetError,
) -> Error {
    match e {
        TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "wan {label} decode: assembling a {width}×{height}×{out_frames} video needs \
             ~{projected_gib:.0} GB just for the output buffers, over this machine's ~{safe_gib:.0} GB \
             safe budget. Reduce the resolution or frame count."
        )),
        TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "wan {label} decode: a {width}×{height}×{out_frames} video peaks at ~{projected_gib:.0} GB \
             even with the smallest tile, over this machine's ~{safe_gib:.0} GB safe budget. Reduce \
             the resolution or frame count."
        )),
    }
}
