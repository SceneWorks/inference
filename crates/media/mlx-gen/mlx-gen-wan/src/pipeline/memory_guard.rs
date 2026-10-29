//! The pre-flight denoise memory guard and the denoise activation estimate.

use super::*;

/// sc-4986 — **pre-flight denoise memory guard.** Estimate the concurrent GPU peak of the
/// DiT-denoise stage (the resident transformer weights + the per-token activation working set of one
/// forward) and return a **catchable** error *before* the expensive text-encode / weight-load when it
/// exceeds this machine's MLX memory budget — instead of letting the OS hard-kill the worker (SIGKILL)
/// or the Metal command buffer abort it (uncaught `kIOGPUCommandBufferCallbackError…` → `terminate`),
/// the two non-recoverable deaths seen in production. Mirrors the z-image sc-4874 `preflight_memory_guard`.
///
/// The staged generate (TE → DiT → VAE each loaded then dropped, see [`crate::model`]) means the DiT
/// stage is the transformer peak; the **14B MoE keeps both experts resident**, so pass the summed
/// expert bytes. `activation_bytes ≈ 72 · batch · tokens · dim` is fit from real Wan2.2 TI2V-5B
/// measurements (peak − weights across L = 1 760 … 32 560, batched B=2; sc-4986). `batch` is 2 with CFG.
///
/// Scope: this guards the **DiT-denoise** stage's memory (OOM / command-buffer abort). It
/// deliberately does *not* encode a wall-time/step-count policy (a long-but-fitting run is the
/// worker's call — sc-4997 / the forward-progress watchdog sc-4984). The z48 VAE-decode peak is a
/// *separate, later* stage (the DiT is freed before the VAE loads), so it has its own budgeted guard
/// in [`auto_tiling_budgeted`] (sc-4998) rather than being summed into this one.
pub fn preflight_denoise_memory_guard(
    model_id: &str,
    dit_resident_bytes: u64,
    tokens: usize,
    dim: usize,
    cfg_enabled: bool,
) -> Result<()> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let weights_gb = dit_resident_bytes as f64 / GIB;
    let peak_gb = estimated_denoise_peak_gib(dit_resident_bytes, tokens, dim, cfg_enabled);
    let act_gb = peak_gb - weights_gb;
    let budget_gb = get_memory_limit() as f64 / GIB;
    let safe = budget_gb * 0.85;
    if peak_gb > safe {
        return Err(Error::Msg(format!(
            "{model_id}: a denoise step at this resolution/frame-count needs ~{peak_gb:.0} GB \
             (transformer ~{weights_gb:.0} GB resident + ~{act_gb:.0} GB activations for {tokens} \
             attention tokens{}), exceeding this machine's ~{safe:.0} GB safe budget ({budget_gb:.0} \
             GB MLX limit × 0.85). Unmitigated, the OS hard-kills the worker (SIGKILL) or the Metal \
             command buffer aborts (sc-4986). Reduce the resolution or frame count, or load a Q8/Q4 \
             snapshot.",
            if cfg_enabled { ", ×2 for CFG" } else { "" }
        )));
    }
    Ok(())
}

/// Estimated concurrent GPU peak (GiB) of one denoise stage: resident transformer weights + the
/// activation working set of a single forward. `activation ≈ 72 B · batch · tokens · dim`, fit from
/// real Wan2.2 TI2V-5B measurements (sc-4986: peak − weights tracked 0.7→14.4 GiB across
/// L = 1 760…32 560 at batch 2). Pure (no global state) so it is unit-testable against those anchors.
pub(super) fn estimated_denoise_peak_gib(
    dit_resident_bytes: u64,
    tokens: usize,
    dim: usize,
    cfg_enabled: bool,
) -> f64 {
    const ACT_BYTES_PER_ELEM: f64 = 72.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let batch = if cfg_enabled { 2.0 } else { 1.0 };
    let act = ACT_BYTES_PER_ELEM * batch * tokens as f64 * dim as f64;
    (dit_resident_bytes as f64 + act) / GIB
}

/// The activation working set (bytes) [`preflight_denoise_memory_guard`] prices for one denoise
/// forward -- the same `72 B · batch · tokens · dim` as [`estimated_denoise_peak_gib`], batch 2 when
/// CFG runs batched -- in checked integer bytes for the SC-20686 admission estimate. `None` on
/// overflow.
pub(crate) fn denoise_activation_bytes(
    tokens: usize,
    dim: usize,
    cfg_enabled: bool,
) -> Option<u64> {
    72_u64
        .checked_mul(if cfg_enabled { 2 } else { 1 })?
        .checked_mul(u64::try_from(tokens).ok()?)?
        .checked_mul(u64::try_from(dim).ok()?)
}
