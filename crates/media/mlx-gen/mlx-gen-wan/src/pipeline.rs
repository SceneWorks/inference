//! S4/S5 — the **T2V generation pipeline**: the denoise loop + CFG + VAE decode + frame assembly
//! that turns latents into video. Port of `generate_wan.py`'s `generate_video` — both the
//! single-model dense path ([`denoise`], S4) and the dual-expert MoE path ([`denoise_moe`], S5).
//!
//! This is **reusable machinery**, not a model: the dense loop is exactly what each Wan2.2-A14B MoE
//! expert runs (the MoE adds only the per-step boundary swap) and what the 5B runs (sc-2680, with
//! its z48 VAE). The concrete `Generator::generate` wiring lands in `model.rs`.
//!
//! Shapes are channels-first **`[C, F, H, W]`** (no batch dim) for the latents + scheduler. CFG runs
//! cond + uncond as a **single batched B=2 forward** ([`WanTransformer::forward_cached`]) — the shared
//! latent is patchified once and broadcast across the batch, so each per-step GPU kernel launches once
//! instead of twice (the small-seq win, sc-2853); it stays bit-identical to two B=1 forwards since
//! attention never mixes batch elements. The per-block cross-attention K/V and the RoPE cos/sin are
//! **precomputed once per expert** before the loop (the reference's `prepare_cross_kv` / `prepare_rope`)
//! and reused across all steps, instead of recomputed every forward.

use mlx_rs::memory::{get_active_memory, get_memory_limit};
use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::tiling::{
    budgeted_plan, SpatialTiling, TemporalOverlapPolicy, TileCandidates, TilingBudgetError,
    TilingConfig, VaeTiling,
};
use mlx_gen::{default_seed, CancelFlag, Error, GenerationRequest, Image, Progress, Result};

use crate::scheduler::{compute_sigmas, make_scheduler, SolverKind, WanScheduler};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Align a pixel dimension **down** to a multiple of `patch · vae_stride` (the reference rounds the
/// requested size to the nearest valid grid; sub-tile requests are rejected by `validate`).
pub fn align_dim(value: u32, patch: usize, stride: usize) -> u32 {
    let align = (patch * stride) as u32;
    (value / align) * align
}

/// Reject a request whose **grid-aligned** geometry exceeds `max_area` (`0` = uncapped). `dw`/`dh`
/// are the model's `patch · vae_stride` grid. Measures the geometry the pipeline would actually
/// render — i.e. the same dims `resolve_capped_dims` returns — so an off-grid request is judged on
/// what it becomes, not on what was typed.
///
/// **sc-12308 — this replaces a silent aspect-preserving refit** (`best_output_size`). SceneWorks
/// always supplies an explicit `width`/`height`, so refitting overrode a geometry the user chose and
/// delivered one on no advertised bucket (`1280×720` → `1264×704`, off every menu entry). Upstream's
/// i2v derives its geometry from the source image + `max_area` and so has no stated geometry to
/// violate; ours does. candle rejects the same request, so rejecting here is what makes one manifest
/// entry mean one thing on both backends — the sc-6983 precedent (surface an adjustment, never apply
/// it silently), taken to its conclusion.
pub fn reject_over_area(
    id: &str,
    req: &GenerationRequest,
    dw: u32,
    dh: u32,
    max_area: usize,
) -> Result<()> {
    reject_over_area_dims(id, req.width, req.height, dw, dh, max_area)
}

/// [`reject_over_area`] over an explicit `width`/`height` instead of `req`'s, for a provider whose
/// rendered geometry is **not** the one on the request.
///
/// SCAIL-2 resolves `width`/`height == 0` from its driving-video frames (sc-16167), so `req.width`
/// is the sentinel `0` and reading it would measure the wrong geometry — `0` area passes every cap.
/// The two entry points share this body so the cap means the same thing whether the geometry was
/// typed or resolved; there is no second copy of the alignment or the message to drift.
pub fn reject_over_area_dims(
    id: &str,
    width: u32,
    height: u32,
    dw: u32,
    dh: u32,
    max_area: usize,
) -> Result<()> {
    if max_area == 0 {
        return Ok(());
    }
    let (w, h) = (
        align_dim(width, 1, dw as usize),
        align_dim(height, 1, dh as usize),
    );
    let area = w as usize * h as usize;
    if area > max_area {
        return Err(Error::Msg(format!(
            "{id}: width×height ({w}×{h} = {area} px) exceeds the max area {max_area} px; \
             reduce the resolution"
        )));
    }
    Ok(())
}

/// Reject a request whose `width`/`height` is **not a multiple of** the model's `patch · vae_stride`
/// grid — the lattice `dw`/`dh` that [`align_dim`] rounds *down* to. An off-grid request is refused
/// here rather than silently snapped to the nearest tile, so the caller gets the geometry it asked for
/// or a clear error, never one it never chose. Every wan grid is square (`patch_size.1 == .2`,
/// symmetric spatial `vae_stride`), so `dw == dh` and the message reports the single stride.
///
/// **sc-12607 — this closes the last "reject (candle) vs silently refit (mlx)" gap on the wan family.**
/// candle hard-errors an off-grid `width`/`height` (`candle_gen_wan`'s `is_multiple_of(SIZE_MULTIPLE)` /
/// `SIZE_MULTIPLE_14B`); mlx used to only `align_dim` it down inside `resolve_capped_dims`, so the same
/// request produced an error on one backend and a snapped render on the other. Rejecting here makes one
/// manifest entry mean one thing on both backends — the sibling of [`reject_over_area`]'s sc-12308 fix,
/// one axis over (the spatial stride instead of the area cap). `dw`/`dh` come from the caller's
/// `grid(&config)`, the same source `align_dim`/`resolve_capped_dims` use, so `validate` can never
/// accept a size the pipeline would then refit.
pub fn reject_off_grid(id: &str, req: &GenerationRequest, dw: u32, dh: u32) -> Result<()> {
    if !req.width.is_multiple_of(dw) || !req.height.is_multiple_of(dh) {
        return Err(Error::Msg(format!(
            "{id}: width/height must be multiples of {dw} (got {}×{})",
            req.width, req.height
        )));
    }
    Ok(())
}

/// Resolve the sampler-loop knobs shared **byte-identically** by every Wan generate path (dense 5B,
/// A14B MoE, single- and dual-expert VACE): the step count, scheduler shift, solver kind, and seed
/// (F-010). Each falls back to the config default when the request leaves it unset; an unset sampler
/// maps to UniPC (the `generate_wan.py` default — `validate` has already rejected any unadvertised
/// name), and an unset seed draws a fresh [`default_seed`] so repeated calls vary. The four return
/// types are distinct, so a mis-ordered destructure at a call site is a compile error.
pub fn resolve_sampler_knobs(
    req: &GenerationRequest,
    steps_default: usize,
    shift_default: f32,
) -> (usize, f32, SolverKind, u64) {
    let steps = req.steps.map(|s| s as usize).unwrap_or(steps_default);
    let shift = req.scheduler_shift.unwrap_or(shift_default);
    let kind = SolverKind::from_name(req.sampler.as_deref().unwrap_or("uni_pc"));
    let seed = req.seed.unwrap_or_else(default_seed);
    (steps, shift, kind, seed)
}

/// Latent shape `[z_dim, t_lat, h_lat, w_lat]` for a `frames × H × W` request.
/// `t_lat = (frames − 1) / vae_stride_t + 1`; spatial divide by the vae stride.
pub fn latent_shape(
    frames: usize,
    height: u32,
    width: u32,
    z_dim: usize,
    vae_stride: (usize, usize, usize),
) -> Result<[i32; 4]> {
    // `frames == 0` would underflow `frames - 1` (usize) into a massive `t_lat`; reject it here,
    // co-located with the subtraction, so a config/parse path that bypasses the upstream frame
    // validation gets a clear error rather than a silent wrong latent shape (F-007).
    let frames_minus_1 = frames
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("wan latent_shape: frames must be >= 1".to_string()))?;
    let t_lat = frames_minus_1 / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    Ok([z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32])
}

/// Transformer sequence length: `ceil(h_lat · w_lat / (patch_h · patch_w) · t_lat)`.
pub fn seq_len(latent: [i32; 4], patch_size: (usize, usize, usize)) -> usize {
    let (_z, t_lat, h_lat, w_lat) = (latent[0], latent[1], latent[2], latent[3]);
    // Exact integer `ceil(h_lat·w_lat·t_lat / (patch_h·patch_w))` — the old f64 ceil was exact only
    // up to 2^24 and could go off-by-one beyond it (F-089).
    let tokens = h_lat as usize * w_lat as usize * t_lat as usize;
    tokens.div_ceil(patch_size.1 * patch_size.2)
}

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
fn estimated_denoise_peak_gib(
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
const VAE22_ACCUM_BYTES_PER_VOXEL: u64 = 40;
const VAE22_TILE_BYTES_PER_OUT_VOXEL: u64 = 3_800;
/// Per-tile coefficient for a **bf16** decode (sc-5039). Measured on the same real-weight rig at two
/// tiles of the 1024×576×97 video (cosine 0.99995, no NaN both): 768 px / 64-frame → **79.7 GB**
/// (vs 97.7 GB f32), 640 px / 48-frame → **55.1 GB**. The per-tile term only drops to ~85 % of f32
/// — *not* 50 % — because the `RMS_norm` channel-L2 reduction stays f32 and materializes a full-size
/// f32 temporary of each activation. Calibrated to the **higher** of the two implied coefficients
/// (the 640/48 point) so the estimate never under-shoots a real peak — the 3100 first guess let the
/// selector pick a tile that measured 55.1 GB, just over the 54.4 GB safe line at the 64 GB tier.
/// The fixed accumulator term is unchanged (the blend buffers are f32 either way).
const VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16: u64 = 3_400;

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
fn estimated_vae22_decode_peak_gib(
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
fn vae22_decode_peak_bytes(
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
const WAN_VAE_BUDGET_ENV: &str = "WAN_VAE_BUDGET_GIB";
/// Fraction of **free** unified memory a decode may target (headroom for allocator churn + the OS).
/// Matches the 0.85 the mlx denoise guard and candle's free-aware Wan tiler both use.
const WAN_VAE_BUDGET_SAFE_FRAC: f64 = 0.85;
/// Conservative fallback budget, used only when the MLX memory limit is disabled (`0`) so no free
/// figure can be derived. Mirrors candle's `WAN22_VAE_DEFAULT_BUDGET_GIB` (16 GiB, the smallest shipped
/// Apple-silicon tier). In practice unreachable — MLX's default limit is 1.5× the recommended working
/// set, never 0 unless a caller explicitly disabled it via `set_memory_limit(0)`.
const WAN_VAE_DEFAULT_BUDGET_GIB: f64 = 16.0;

/// Live **free** unified memory in GiB — the MLX analogue of candle's `nvidia-smi memory.free`
/// (`total − used`). On unified memory "free" is the MLX memory limit (the soft working-set ceiling,
/// [`get_memory_limit`]) MINUS the bytes held by live arrays ([`get_active_memory`]) — i.e.
/// `limit − currently_resident`. The reclaimable buffer **cache** (`get_cache_memory`) is deliberately
/// NOT subtracted: MLX auto-reclaims it under allocation pressure, so it is genuinely available to the
/// decode (subtracting it would over-tile with no safety gain). `None` when the limit is disabled (`0`)
/// — there is no ceiling to subtract from — so the resolver falls back to the conservative default.
fn live_free_gib() -> Option<f64> {
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
fn free_aware_budget_gib(free_gib: f64, safe_frac: f64) -> f64 {
    free_gib.max(0.0) * safe_frac
}

/// Core of [`wan_vae_safe_budget_gib`] with the free-memory probe injected, so the env-override
/// precedence + the probe→default fallback are unit-testable without touching the global memory limit.
/// The live entry point passes [`live_free_gib`]; tests pass a stub closure. Mirrors candle-gen's
/// `vae_tiling::resolve_free_aware_budget`.
fn resolve_free_aware_budget(
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
fn wan_vae_safe_budget_gib() -> f64 {
    resolve_free_aware_budget(
        WAN_VAE_BUDGET_ENV,
        WAN_VAE_BUDGET_SAFE_FRAC,
        WAN_VAE_DEFAULT_BUDGET_GIB,
        live_free_gib,
    )
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
fn plan_vae22_tiling(
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
fn plan_vae22_tiling_for_selected_spatial(
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

fn selected_vae22_plan_with_budget(
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

/// Map gen-core's neutral [`TilingBudgetError`] to a wan-facing message tagged with the VAE `label`
/// (e.g. `"z48 vae22"`, `"z16 vae"`). Shared by the per-VAE budgeted-tiling wrappers so the catchable
/// over-budget wording stays identical across them.
fn wan_budget_error(
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
const VAE16_ACCUM_BYTES_PER_VOXEL: u64 = 64;
/// Per-tile-output-voxel cost of the z16 decoder working set (conv stack + ×8 spatial / ×4 temporal
/// upsample). Fit from the same anchors at ~6355 B/voxel (≈1.7× the z48 `vae22`'s 3800 — the bigger
/// spatial upsample); rounded **up** to 6500. z16 decodes f32 in production, so there is no bf16
/// coefficient (unlike `vae22`, sc-5039).
const VAE16_TILE_BYTES_PER_OUT_VOXEL: u64 = 6_500;

/// Candidate spatial tile sizes (output px, multiples of the z16 ×8 spatial scale, overlap 64).
const VAE16_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
const VAE16_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];

/// Estimated concurrent GPU peak (GiB) of a z16 decode whose largest tile spans `tile_*` output voxels
/// while assembling an `out_*` video. Pure (no global state) → unit-testable against the
/// `vae16_decode_sweep.rs` anchors. Single-pass is the special case `tile_* == out_*`; a zero tile is
/// the accumulator-only floor.
fn estimated_z16_decode_peak_gib(
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
const VAE16_ENCODE_SINGLE_FRAME_BYTES_PER_PIXEL: u64 = 6_300;
/// z16 encode: bytes per pixel of one 4-frame chunk.
const VAE16_ENCODE_CHUNK_BYTES_PER_VOXEL: u64 = 4_600;
/// z16 encode: bytes per input voxel of the whole clip (input + accumulated latent).
const VAE16_ENCODE_CLIP_BYTES_PER_VOXEL: u64 = 700;
/// z48 encode: bytes per input pixel of a single-frame encode.
const VAE22_ENCODE_SINGLE_FRAME_BYTES_PER_PIXEL: u64 = 2_900;
/// z48 encode: bytes per voxel of one 4-frame chunk.
const VAE22_ENCODE_CHUNK_BYTES_PER_VOXEL: u64 = 2_400;
/// z48 encode: bytes per input voxel of the whole clip.
const VAE22_ENCODE_CLIP_BYTES_PER_VOXEL: u64 = 50;
/// Frames in every chunk after the first (the encoders' temporal stride).
const VAE_ENCODE_CHUNK_FRAMES: u64 = 4;

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
fn plan_z16_tiling(
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

fn plan_z16_tiling_with_overlap(
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

/// Classifier-free guidance combine: `uncond + gs·(cond − uncond)`.
fn cfg_combine(cond: &Array, uncond: &Array, gs: f32) -> Result<Array> {
    Ok(add(
        uncond,
        &multiply(&subtract(cond, uncond)?, scalar(gs))?,
    )?)
}

/// Per-generate caches for one transformer/expert, constant across every denoise step: the bf16 RoPE
/// `(cos, sin)` for the (fixed) grid + each block's cross-attention K/V for the (CFG-batched) context.
/// Mirrors the reference's `prepare_rope` / `prepare_cross_kv`, computed once before the loop.
struct StepCache {
    /// Per-block cross-attention `(k, v)`, each `[batch, n, text_len, d]` (bf16).
    cross_kv: Vec<(Array, Array)>,
    cos: Array,
    sin: Array,
    /// Forward batch width: 2 when CFG is on (cond+uncond stacked), else 1.
    batch: usize,
    /// SC-20686 Metal-lane ownership of `cross_kv` (`None` unless a campaign is armed). Declared
    /// last so the K/V arrays are freed before the post-release allocator remnant is sampled.
    _sc20686: Option<mlx_gen::sc20686::CacheSetGuard>,
}

/// SC-20686 operation names for the Wan cross-K/V cache lifecycle (Metal lane).
pub(crate) const SC20686_CROSS_KV_CREATE: &str = "WanTransformer::prepare_cross_kv";
pub(crate) const SC20686_CROSS_KV_RELEASE: &str = "StepCache::drop";

/// Build the per-expert [`StepCache`] from the embedded contexts + the (constant) RoPE grid. When CFG
/// is on (`ctx_uncond = Some`) the cond/uncond contexts are stacked on the batch axis so the cross-K/V
/// is `B=2`; otherwise `B=1`. The caches are evaluated once here (the reference's `mx.eval(cross_kv,
/// rope_cos_sin)`) so each per-step graph reuses them instead of recomputing.
fn build_cache(
    transformer: &WanTransformer,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    grid: (usize, usize, usize),
) -> Result<StepCache> {
    let (context_batch, batch) = match ctx_uncond {
        Some(uncond) => (concatenate_axis(&[ctx_cond, uncond], 0)?, 2),
        None => (ctx_cond.clone(), 1),
    };
    mlx_gen::sc20686::mark_phase("prepare-cache");
    let cross_kv = transformer.prepare_cross_kv(&context_batch)?;
    let (cos, sin) = transformer.prepare_rope(grid)?;
    let mut to_eval: Vec<&Array> = vec![&cos, &sin];
    for (k, v) in &cross_kv {
        to_eval.push(k);
        to_eval.push(v);
    }
    mlx_rs::transforms::eval(to_eval)?;
    // Inert (`None`, no work) unless an SC-20686 campaign is armed on this thread.
    let sc20686 = mlx_gen::sc20686::register_cross_kv_set(
        &cross_kv,
        (grid.0 * grid.1 * grid.2) as u64,
        SC20686_CROSS_KV_CREATE,
        SC20686_CROSS_KV_RELEASE,
    )?;
    mlx_gen::sc20686::mark_denoise();
    Ok(StepCache {
        cross_kv,
        cos,
        sin,
        batch,
        _sc20686: sc20686,
    })
}

/// One denoise prediction reusing the precomputed [`StepCache`]: a single batched forward yielding
/// `[cond, uncond]`, combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only
/// forward.
///
/// `y` is the optional I2V channel-concat conditioning `[20, F, H, W]` (mirrors `WanModel.__call__`'s
/// `y`): when `Some`, it is concatenated **onto the channel axis after** the `[16, …]` noise latent —
/// `[noise(16), mask(4), z_video(16)]` → `[36, F, H, W]` — before patchify, exactly the channel order
/// the I2V-14B `patch_embedding` (in_dim 36) was trained on. The DiT prediction stays `out_dim = 16`,
/// so the scheduler step still consumes/produces the 16-channel latent.
fn predict(
    transformer: &WanTransformer,
    latents: &Array,
    t: f32,
    cache: &StepCache,
    guidance: f32,
    y: Option<&Array>,
) -> Result<Array> {
    predict_approx(transformer, latents, t, cache, guidance, y, None, None)
}

/// [`predict`] carrying the sc-18322 denoise feature cache. `trunk: None` is exactly [`predict`].
#[allow(clippy::too_many_arguments)]
fn predict_approx(
    transformer: &WanTransformer,
    latents: &Array,
    t: f32,
    cache: &StepCache,
    guidance: f32,
    y: Option<&Array>,
    trunk: Option<&mut crate::feature_cache::TrunkCache>,
    prune: Option<&crate::token_pruning::TokenPruner>,
) -> Result<Array> {
    let x = match y {
        Some(y) => concatenate_axis(&[latents, y], 0)?,
        None => latents.clone(),
    };
    let preds = transformer.forward_cached_approx(
        &x,
        t,
        &cache.cross_kv,
        &cache.cos,
        &cache.sin,
        cache.batch,
        trunk,
        prune,
    )?;
    if cache.batch == 2 {
        // preds[0] = cond (context row 0), preds[1] = uncond (row 1).
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        preds
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("wan: B=1 forward produced no output".into()))
    }
}

/// The dense denoise loop (single model). `ctx_cond`/`ctx_uncond` are
/// [`WanTransformer::embed_text`] outputs; pass `ctx_uncond = None` for the CFG-disabled B=1 fast
/// path. `init_noise` is `[C, F, H, W]` f32. Returns the denoised latents `[out_dim, F, H, W]`
/// (f32). `on_step(i)` is called after each completed step.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    denoise_approx(
        transformer,
        kind,
        num_train_timesteps,
        steps,
        shift,
        guidance,
        ctx_cond,
        ctx_uncond,
        init_noise,
        cancel,
        on_step,
        None,
        None,
    )
}

/// Refuse an approximate plan on a denoise route that does not implement it (sc-18322).
///
/// The routes that are **not** wired for the denoise feature cache — the TI2V mask-blend, the curated
/// unified solvers, and the MoE expert-swap paths — must refuse rather than run exactly, because
/// silently ignoring a selection is the defect the typed approximate surface exists to remove. Each
/// has a structural reason it is unwired, named in the message so a caller knows it is not a gap
/// waiting to be filled by a retry:
///
/// * **TI2V mask-blend** re-blends the conditioning latent into the trajectory after every step, so a
///   trunk residual captured at one step is not the quantity a later step needs.
/// * **Curated solvers** evaluate the model 1..N times per solver step (Heun twice, DPM++ SDE at a
///   midpoint), so "the previous step's residual" is ambiguous — and the step index the cache keys on
///   never reaches the predict closure.
/// * **MoE expert swap** changes transformers mid-trajectory, so a residual captured under the
///   high-noise expert is meaningless under the low-noise one, a rotation phase captured under one is
///   not the other's, and retaining either across the swap would keep the outgoing expert's weights
///   alive.
///
/// Each of those three has a live call site: `Wan::generate_impl`'s TI2V and curated arms, and
/// `Wan14b::generate_impl`, which covers every A14B route because every A14B route swaps experts. The
/// refusal names the plan's **actually selected** mechanisms rather than a fixed string, so a
/// pruning-only plan is not refused with the cache's name.
///
/// Unreachable today — the shared floor refuses every approximate selection before a provider is
/// called, so no plan reaching here can be anything but `Exact`. It exists so that the day a
/// characterization artifact makes selection possible, these routes fail closed by construction
/// instead of needing to be remembered.
pub fn refuse_unwired_approximation(
    id: &str,
    route: &str,
    plan: &mlx_gen::gen_core::ApproximationPlan,
) -> Result<()> {
    if plan.is_exact() {
        return Ok(());
    }
    Err(Error::Unsupported(format!(
        "{id}: the {route} denoise route does not implement {}. Leave approximation unset for this \
         route's exact denoise.",
        plan.describe_selection()
    )))
}

/// [`denoise`] carrying the sc-18322 **denoise feature cache**.
///
/// `trunk: None` is exactly [`denoise`]. When present, the cache decides per step whether the block
/// stack runs or its retained trunk residual is reapplied — and note where that decision is *not*: the
/// cancel check, the per-step `eval` and the progress callback below are straight-line statements in
/// this loop, outside the branch, so a reused step performs all three exactly as a recomputed one
/// does. That is the whole of the cancel-responsiveness discipline; see [`crate::feature_cache`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_approx(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
    mut trunk: Option<&mut crate::feature_cache::TrunkCache>,
    mut prune: Option<&mut crate::token_pruning::TokenPruner>,
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: run the DiT's fusable elementwise glue (adaLN affine, gated residual, gated-GELU FFN,
    // RoPE rotation) through `mx.compile` — bit-exact (proven `max|Δ|=0` real + tiny, perf.rs /
    // compile_parity.rs) and ~14% faster/step at production geometry. Scoped + restored on drop by the
    // RAII guard (F-006/F-007) instead of leaking the render thread's setting into later work.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute the RoPE + cross-K/V caches once (grid + context are constant across steps).
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract (sc-5551, the video sibling of chroma's sc-5514):
        // a video render runs minutes, so check before each step. The per-step `eval` below makes
        // this effective — without it MLX's lazy graph defers all compute to VAE decode and this
        // check would pass for every step.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Declare the step to the feature cache before the forward: the step index lives here, in the
        // only loop that has one, so the cache never has to derive it from a timestep (sc-18322).
        if let Some(cache) = trunk.as_deref_mut() {
            cache.begin_step(i);
        }
        // Pruning declares the step for the same reason the cache does — it keys its per-block rotation
        // on it — and its warmup decision comes from the contract's own predicate, so a warmup step
        // passes `None` and runs the untouched block loop.
        if let Some(pruner) = prune.as_deref_mut() {
            pruner.begin_step(i);
        }
        let prune_this_step = match prune.as_deref() {
            Some(pruner) if pruner.prunes_step(i) => Some(pruner),
            _ => None,
        };
        let pred = predict_approx(
            transformer,
            &latents,
            t,
            &cache,
            guidance,
            None,
            trunk.as_deref_mut(),
            prune_this_step,
        )?;
        latents = sched.step(&pred, &latents)?;
        // Force evaluation each step to bound the lazy graph's peak memory (the reference's
        // per-step `mx.eval(latents)`).
        //
        // Unconditional, and outside the feature-cache branch three call levels down: a reused step is
        // materialized and progress-reported exactly like a recomputed one, so skipping the block stack
        // can never make the loop less cancel-responsive (sc-18322).
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// The dense denoise loop driven by a **curated unified solver** (epic 7114, sc-7121) — the additive
/// fold onto the shared gen-core solver library, alongside the native [`denoise`]. Routes any curated
/// [`mlx_gen::Solver`] (`euler` / `euler_ancestral` / `heun` / `dpmpp_sde` / `ddim` / …) through
/// `mlx_gen::run_flow_sampler` over Wan's own shifted flow-σ schedule ([`compute_sigmas`]).
///
/// Wan's native `unipc`/`dpmpp2m` are a diffusers `FlowDPMSolver`/`FlowUniPC` in flow-SNR space
/// (`λ = log((1−σ)/σ)`), which the gen-core VE-space solvers (`λ = −ln σ`) do NOT reproduce — so the
/// native default stays on [`denoise`] (the N1 default-parity gate), and this path is selected only for
/// the gen-core-only curated solvers. The model is velocity-prediction over the FLOW
/// [`mlx_gen::TimestepConvention::Sigma`] convention, and Wan feeds the DiT the integer-valued timestep
/// `(σ·num_train).trunc()` (the predict closure maps σ → that timestep). `seed` drives the stochastic
/// solvers' per-step noise. Progress / cancel route through `run_flow_sampler`'s per-eval hook.
#[allow(clippy::too_many_arguments)]
pub fn denoise_curated(
    transformer: &WanTransformer,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    let nt = num_train_timesteps as f32;
    mlx_gen::run_flow_sampler(
        Some(sampler_name),
        mlx_gen::TimestepConvention::Sigma,
        &sigmas,
        init_noise.clone(),
        seed,
        cancel,
        on_progress,
        |x, sigma| {
            // Wan feeds the DiT the integer-valued timestep `(σ·num_train).trunc()`, not raw σ.
            let t = (sigma * nt).trunc();
            predict(transformer, x, t, &cache, guidance, None)
        },
    )
}

/// One TI2V prediction with **per-token timesteps**, reusing the precomputed [`StepCache`]: a single
/// batched forward over the per-token timestep vector `t_tokens` `[1, L]` (mask-blend, sc-2680),
/// combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only forward. Mirrors
/// [`predict`] but routes through [`WanTransformer::forward_tokens_cached`].
fn predict_tokens(
    transformer: &WanTransformer,
    latents: &Array,
    t_tokens: &Array,
    cache: &StepCache,
    guidance: f32,
) -> Result<Array> {
    let preds = transformer.forward_tokens_cached(
        latents,
        t_tokens,
        &cache.cross_kv,
        &cache.cos,
        &cache.sin,
        cache.batch,
    )?;
    if cache.batch == 2 {
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        preds
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("wan: B=1 forward produced no output".into()))
    }
}

/// The image-conditioned TI2V-5B **mask-blend** denoise loop (port of `generate_wan.py`'s
/// `is_i2v_mask_blend` path, sc-2680). The first latent temporal frame is pinned to the encoded
/// image `z_img` and *frozen*: every step (1) builds the per-token timestep vector `t_tokens =
/// mask_tokens · t` (`0` for the first-frame tokens, so they carry timestep 0), (2) predicts the
/// noise with `predict_tokens`, (3) scheduler-steps, then (4) re-blends `latents = (1−mask)·z_img +
/// mask·latents` so the first frame stays the conditioning image while the rest denoise.
///
/// `init_latents` is the pre-blended `[C,F,H,W]` start `(1−mask)·z_img + mask·noise`; `z_img` is the
/// VAE-encoded image `[C,1,H,W]` (broadcasts over `F`); `mask` is `[C,F,H,W]` (`0` first frame, `1`
/// rest); `mask_tokens` is `[1,L]` (`0` first-frame tokens, `1` rest), `L` = the patch-token count.
#[allow(clippy::too_many_arguments)]
pub fn denoise_ti2v(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_latents: &Array,
    z_img: &Array,
    mask: &Array,
    mask_tokens: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: compile the DiT's fusable elementwise glue (bit-exact, ~14% faster/step). The per-token
    // modulation shapes differ from T2V's, so `mx.compile` simply re-traces them once. Scoped +
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking into later work.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute the RoPE + cross-K/V caches once (grid + context constant across steps), exactly like
    // [`denoise`]. The per-token timesteps change each step (the only per-step DiT input besides the
    // latent), so the time embedding is recomputed inside `forward_tokens_cached`.
    let grid = transformer.patch_grid(init_latents);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    // `(1−mask)·z_img` — the frozen first-frame content (z_img broadcasts over F); precomputed once.
    let one_minus_mask = subtract(scalar(1.0), mask)?;
    let frozen = multiply(&one_minus_mask, z_img)?;

    let mut latents = init_latents.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Per-token timesteps: 0 for the first-frame tokens (frozen), `t` for the rest. (The 5B's
        // seq_len equals the patch count, so no padding is needed — matches the reference.)
        let t_tokens = multiply(mask_tokens, scalar(t))?;
        let pred = predict_tokens(transformer, &latents, &t_tokens, &cache, guidance)?;
        latents = sched.step(&pred, &latents)?;
        // Re-apply the mask so the first frame stays pinned to the conditioning image.
        latents = add(&frozen, &multiply(mask, &latents)?)?;
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// One MoE expert: a full transformer + its own (per-model) embedded contexts + guidance scale.
/// Wan2.2-A14B's "MoE" is two complete checkpoints, not token routing — each carries its own
/// `text_embedding`, so contexts are embedded per expert.
pub struct Expert<'a> {
    pub transformer: &'a WanTransformer,
    /// `embed_text` output for this expert (cond).
    pub ctx_cond: Array,
    /// `embed_text` output for this expert (uncond); `None` ⇒ CFG disabled for this expert.
    pub ctx_uncond: Option<Array>,
    /// This expert's guidance scale (the `low`/`high` of the dual `sample_guide_scale`).
    pub guidance: f32,
}

/// First step index whose integer timestep drops **below** `boundary_timestep` — the single high→low
/// MoE crossing (sc-12736, mirroring candle's `crossing_index`). Steps `0..k` run the high-noise
/// expert (`t ≥ boundary_timestep`), `k..steps` the low-noise expert. The flow-match integer timesteps
/// are monotonically non-increasing (built from a decreasing σ schedule), so this prefix/suffix split
/// is **exactly** the per-step `t ≥ boundary_timestep` choice the resident loop makes — returns
/// `timesteps.len()` if the boundary is never crossed (all high), `0` if it is crossed at the first
/// step (all low). Pure + GPU-free, so both the resident [`denoise_moe`] and the sequential
/// expert-swap share it and a unit test can pin the split against the per-step rule.
pub fn crossing_index(timesteps: &[f32], boundary_timestep: f32) -> usize {
    timesteps
        .iter()
        .position(|&t| t < boundary_timestep)
        .unwrap_or(timesteps.len())
}

/// Run the denoise steps `range` on a **single** expert, advancing the (shared, continuous) scheduler
/// and mutating `latents` in place. Extracted (sc-12736) so the resident MoE loop ([`denoise_moe`]) and
/// the sequential expert-swap drive the **identical** per-step math over their respective step ranges:
/// the prefix/suffix split is bit-exact to the resident per-step `t ≥ boundary` choice because the same
/// `sched` advances through every step in order (a multistep solver's history therefore carries across
/// the crossing unchanged). The per-expert `StepCache` (RoPE + cross-K/V) is built once here, at the
/// start of the range.
///
/// `y` is the optional I2V channel-concat conditioning (see `predict`). `grid` is the shared patch
/// grid (both experts see the same F/H/W). `on_step(i)` is called with the **global** step index after
/// each completed step.
#[allow(clippy::too_many_arguments)]
pub fn denoise_range(
    sched: &mut dyn WanScheduler,
    e: &Expert,
    grid: (usize, usize, usize),
    y: Option<&Array>,
    latents: &mut Array,
    timesteps: &[f32],
    range: std::ops::Range<usize>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<()> {
    let cache = build_cache(e.transformer, &e.ctx_cond, e.ctx_uncond.as_ref(), grid)?;
    for i in range {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let t = timesteps[i];
        let pred = predict(e.transformer, latents, t, &cache, e.guidance, y)?;
        *latents = sched.step(&pred, latents)?;
        mlx_rs::transforms::eval([&*latents])?;
        on_step(i + 1);
    }
    Ok(())
}

/// The dual-expert MoE denoise loop (Wan2.2-A14B). The **high-noise** expert runs while the integer
/// timestep is `≥ boundary_timestep` (`config.boundary · num_train_timesteps`, e.g. `0.875 · 1000 =
/// 875`) and the **low-noise** expert below it — switching the transformer, the per-expert contexts,
/// and the per-expert guidance together. Reduces to [`denoise`] when both experts are the same model.
///
/// This is the **resident** path: BOTH experts are held resident by the caller for the whole loop. The
/// single high→low boundary crossing is a prefix/suffix split ([`crossing_index`] + two
/// [`denoise_range`] calls over one continuous scheduler), so it is bit-exact to the old per-step
/// `t ≥ boundary` choice — and drives the *identical* per-step math the sequential expert-swap
/// (sc-12736) drives, so the residency change stays numerics-preserving.
///
/// `y` is the optional I2V-14B channel-concat conditioning `[20, F, H, W]` ([`build_i2v_y`]),
/// concatenated onto each forward's noise latent (see `predict`); `None` for T2V. It is constant
/// across steps and shared by both experts (the conditioning doesn't change with the noise level).
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: compiled elementwise glue (bit-exact, ~14% faster/step) — see `denoise`. Scoped +
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking into later work.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // The grid is shared — the channel-concat `y` doesn't change F/H/W and each expert's contexts are
    // constant across steps. The single high→low crossing: high runs `0..k` (`t ≥ boundary`), low runs
    // `k..steps`. One continuous `sched` advances across the split, so a multistep solver's history is
    // unbroken (bit-exact to the old per-step choice).
    let grid = low.transformer.patch_grid(init_noise);
    let k = crossing_index(&timesteps, boundary_timestep);
    let mut latents = init_noise.clone();
    denoise_range(
        &mut *sched,
        high,
        grid,
        y,
        &mut latents,
        &timesteps,
        0..k,
        cancel,
        on_step,
    )?;
    denoise_range(
        &mut *sched,
        low,
        grid,
        y,
        &mut latents,
        &timesteps,
        k..timesteps.len(),
        cancel,
        on_step,
    )?;
    Ok(latents)
}

/// The dual-expert MoE denoise loop driven by a **curated unified solver** (epic 7114, sc-7121) — the
/// additive fold onto the shared gen-core solver library, alongside the native [`denoise_moe`]. Same
/// rationale as [`denoise_curated`]: the native `unipc`/`dpmpp2m` (flow-SNR) stay native (N1), and this
/// path serves the gen-core-only curated solvers. The boundary expert swap is applied inside the
/// predict closure (the integer timestep `(σ·num_train).trunc()` is compared to `boundary_timestep`),
/// so a multi-eval solver re-evaluates the correct expert at its intermediate σ.
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe_curated(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let grid = low.transformer.patch_grid(init_noise);
    let low_cache = build_cache(
        low.transformer,
        &low.ctx_cond,
        low.ctx_uncond.as_ref(),
        grid,
    )?;
    let high_cache = build_cache(
        high.transformer,
        &high.ctx_cond,
        high.ctx_uncond.as_ref(),
        grid,
    )?;
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    let nt = num_train_timesteps as f32;
    mlx_gen::run_flow_sampler(
        Some(sampler_name),
        mlx_gen::TimestepConvention::Sigma,
        &sigmas,
        init_noise.clone(),
        seed,
        cancel,
        on_progress,
        |x, sigma| {
            let t = (sigma * nt).trunc();
            let (e, cache) = if t >= boundary_timestep {
                (high, &high_cache)
            } else {
                (low, &low_cache)
            };
            predict(e.transformer, x, t, cache, e.guidance, y)
        },
    )
}

/// Sequential-residency twin of [`denoise_moe_curated`]. The curated solver remains in control of
/// every model evaluation (including Heun / DPM++ SDE sub-evaluations); this wrapper changes only
/// which expert is resident for that evaluation. Curated sigma trajectories are monotonically
/// descending, so there is at most one high→low transition. On that transition the outgoing expert
/// is dropped and its MLX buffers are flushed before the incoming expert is loaded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_moe_curated_swapped(
    boundary_timestep: f32,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut load: impl FnMut(bool) -> Result<(WanTransformer, Array, Option<Array>, f32)>,
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let (transformer, cond, uncond, guidance) = load(true)?;
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(&transformer, &cond, uncond.as_ref(), grid)?;
    // The first shifted Wan sigma is always above the configured A14B boundary. Keeping the
    // initially loaded high expert here avoids a second load before the first evaluation.
    let mut active = Some((transformer, cache, guidance, true));
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    let nt = num_train_timesteps as f32;
    let result = mlx_gen::run_flow_sampler(
        Some(sampler_name),
        mlx_gen::TimestepConvention::Sigma,
        &sigmas,
        init_noise.clone(),
        seed,
        cancel,
        on_progress,
        |x, sigma| {
            let t = (sigma * nt).trunc();
            let wants_high = t >= boundary_timestep;
            let has_high = active.as_ref().is_some_and(|(_, _, _, high)| *high);
            if wants_high != has_high {
                // A descending sigma schedule may cross high→low once, never low→high.
                if wants_high {
                    return Err(Error::Msg(
                        "wan: curated sampler timestep increased across the MoE boundary".into(),
                    ));
                }
                // SC-20686 Metal lane: the swap (drop, evict, load) is its own phase window.
                mlx_gen::sc20686::mark_phase("load");
                drop(active.take());
                mlx_rs::memory::clear_cache();
                let (transformer, cond, uncond, guidance) = load(false)?;
                let cache = build_cache(&transformer, &cond, uncond.as_ref(), grid)?;
                active = Some((transformer, cache, guidance, false));
            }
            let (transformer, cache, guidance, _) =
                active.as_ref().expect("curated expert is loaded");
            predict(transformer, x, t, cache, *guidance, y)
        },
    );
    drop(active);
    mlx_rs::memory::clear_cache();
    result
}

/// Drive the A14B high→low MoE expert swap so the two ~8-9 GB experts are **never co-resident**
/// (sc-12736, epic 12732 — the mlx Pillar-1 win, the residency twin of candle's `staged_expert_swap`):
/// load the high expert, `use_high` it over steps `0..k`, DROP it + `evict`, then load the low expert
/// and `use_low` it over `k..steps`. Each expert is loaded, used, then **explicitly dropped before the
/// next loads** — the evict-then-load discipline: MLX frees the dropped expert's arrays to its buffer
/// cache and `evict` (`mlx_rs::memory::clear_cache`) returns them to the OS *before* the incoming
/// expert materializes, so the peak is one expert, not both. A naive "load low, then drop high" would
/// momentarily co-reside both and the footprint would NOT drop; the ordering here is the whole point.
///
/// There is exactly ONE boundary crossing per denoise (`t ≥ boundary` ⇒ high, below ⇒ low; flow-match
/// timesteps decrease monotonically), so at most one swap. An expert whose step range is empty
/// (`k == 0` all-low, or `k == steps` all-high) is skipped entirely — it never loads.
///
/// Callers must `eval` any output that must survive the drop (the latents) *before* the expert is
/// dropped, or MLX's lazy graph would keep the expert's weights referenced and freeing it would reclaim
/// nothing — [`denoise_range`] already `eval`s the latents after every step, so by the time `use_high`
/// returns they are materialized and independent of the expert.
///
/// Generic over the expert type `E` and threaded state `St` so a weight-free unit test can pin the
/// never-co-resident property with a lightweight liveness witness — no GPU, no real weights — exactly as
/// the resident/sequential parity rests on [`denoise_range`]. The load closures receive `&mut St` so
/// they can emit their [`Progress::Loading`] before the (heavy) load; the use closures receive `&mut St`
/// to advance the shared scheduler/latents.
#[allow(clippy::too_many_arguments)]
pub fn staged_expert_swap<E, St>(
    k: usize,
    steps: usize,
    state: &mut St,
    load_high: impl FnOnce(&mut St) -> Result<E>,
    use_high: impl FnOnce(&E, &mut St) -> Result<()>,
    load_low: impl FnOnce(&mut St) -> Result<E>,
    use_low: impl FnOnce(&E, &mut St) -> Result<()>,
    mut evict: impl FnMut() -> Result<()>,
) -> Result<()> {
    if k > 0 {
        let high = load_high(state)?;
        use_high(&high, state)?;
        // Evict-then-load: drop the high expert (its arrays return to MLX's buffer cache) BEFORE the
        // low expert is ever loaded, then flush the cache to the OS — the never-co-resident invariant.
        drop(high);
        evict()?;
    }
    if k < steps {
        let low = load_low(state)?;
        use_low(&low, state)?;
        drop(low);
        evict()?;
    }
    Ok(())
}

/// Decode denoised latents `[C, F, H, W]` → an RGB video tensor `[F_out, H_out, W_out, 3]` of
/// `uint8` (the reference's `(video + 1)/2 · 255`, clamped). Uses the Wan 2.1 z16 VAE (S2). When
/// `tiling` is `Some`, decodes via [`WanVae::decode_tiled`] (memory-bounded for large/long video;
/// it falls back to a single pass when the config doesn't fire); `None` is always single-pass.
pub fn decode_to_frames(
    vae: &WanVae,
    latents: &Array,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Array> {
    // Honor cancellation before the (dominant-cost, sc-4998) decode stage (F-014).
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(Error::Canceled);
    }
    // WanVae::decode[_tiled] expect/return a leading batch dim: [1, 3, F, H, W] in [-1, 1].
    let z = latents.reshape(&prepend1(latents.shape()))?;
    let video = match tiling {
        Some(cfg) => vae.decode_tiled(&z, cfg, cancel)?,
        None => vae.decode(&z)?,
    };
    // [1,3,F,H,W] → [F,H,W,3]
    let sh = video.shape(); // [1,3,F,H,W]
    let (f, h, w) = (sh[2], sh[3], sh[4]);
    let chw = video
        .reshape(&[3, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // [F,H,W,3]
                                         // [-1,1] → [0,255] uint8
    let scaled = multiply(&add(&chw, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(mlx_rs::ops::round(&clamped, None)?.as_dtype(mlx_rs::Dtype::Uint8)?)
}

/// Decode denoised z48 latents `[C, F, H, W]` → an RGB video tensor `[F_out, H_out, W_out, 3]` of
/// `uint8` via the Wan **2.2** z48 [`Wan22Vae`] (sc-2680). The vae22 decoder is **channels-last** and
/// emits `[1, F', 16H, 16W, 3]` in `[-1, 1]` directly (no `[1,3,F,H,W]` transpose, unlike the z16
/// [`decode_to_frames`]); this drops the batch axis and maps `(v+1)/2·255` clamped. `tiling` →
/// [`Wan22Vae::decode_tiled`] (memory-bounded); `None` is single-pass.
pub fn decode_to_frames_22(
    vae: &Wan22Vae,
    latents: &Array,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Array> {
    // Honor cancellation before the (dominant-cost, sc-4998) decode stage (F-014).
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(Error::Canceled);
    }
    let video = match tiling {
        Some(cfg) => vae.decode_tiled(latents, cfg, cancel)?,
        None => vae.decode(latents)?,
    };
    // [1, F', H', W', 3] → [F', H', W', 3]; [-1,1] → [0,255] uint8.
    let sh = video.shape();
    let (f, h, w) = (sh[1], sh[2], sh[3]);
    let frames = video.reshape(&[f, h, w, 3])?;
    let scaled = multiply(&add(&frames, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(mlx_rs::ops::round(&clamped, None)?.as_dtype(mlx_rs::Dtype::Uint8)?)
}

/// Split a `[F, H, W, 3]` `uint8` video tensor (the [`decode_to_frames`] output) into one
/// [`Image`] per frame. The tensor is transpose-strided, so a raw `as_slice` would read the
/// physical (pre-transpose) buffer — `reshape` first re-materializes it in logical C-order, then we
/// chunk the contiguous bytes `H·W·3` at a time (see `mlx_rs_as_slice_physical_buffer`).
pub fn frames_to_images(frames_u8: &Array) -> Result<Vec<Image>> {
    let sh = frames_u8.shape(); // [F, H, W, 3]
    let (f, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    // sc-12748: reshape to the frame-major `[F, H·W·C]` — the natural per-frame layout. This collapses
    // the (transpose-strided) H/W/C axes, which forces a contiguous logical-order copy (`as_slice`
    // returns the physical buffer), and it never forms the full `f·h·w·c` product — `f` stays its own
    // dim and the inner `h·w·c` is a single frame (≤ ~1e8 even at 8K, well within i32). This retires
    // the F-070 `reshape(-1)` workaround: on the pinned MLX 0.32.0 fork, a multi-dimensional reshape
    // whose individual dimensions fit i32 is exact past an `i32::MAX` total (verified in
    // `mlx-gen/tests/mlx_write_bound_probe.rs`; a single `reshape(-1)` dimension still raises). Thus
    // `f·h·w·3 > i32::MAX` (1920×1088@349f ≈ 2.19e9; 4K@89f) now renders rather than overflowing.
    // Byte-identical to the old reshape below-bound.
    let per = (h as i64 * w as i64 * c as i64) as usize;
    let flat = frames_u8.reshape(&[f, h * w * c])?;
    let bytes = flat.as_slice::<u8>();
    let mut out = Vec::with_capacity(f as usize);
    for i in 0..f as usize {
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels: bytes[i * per..(i + 1) * per].to_vec(),
        });
    }
    Ok(out)
}

/// `[d0, d1, ...]` → `[1, d0, d1, ...]` (prepend a batch axis).
fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

// ===========================================================================================
// I2V-14B channel-concat conditioning (port of `generate_wan.py`'s `is_i2v_channel_concat` setup)
// ===========================================================================================

/// Python `round()` — round half to **even** (banker's rounding), matching `round(img.width * scale)`
/// in the reference's image preprocessing. (Rust `f64::round` rounds half away from zero, which would
/// differ on exact `.5` derived sizes.)
fn py_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    // Round up on frac > 0.5, or on an exact tie (frac == 0.5) when `floor` is odd (→ even).
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    (if round_up { floor + 1.0 } else { floor }) as usize
}

/// Preprocess an I2V conditioning image to `[3, height, width]` f32 in `[-1, 1]` (CHW), matching the
/// reference's inline pipeline: **cover-fit** LANCZOS resize (`scale = max(W/iw, H/ih)`, new dims
/// `round(·)`), **center-crop** to the target, then `px/255·2 − 1`. The resize is the core PIL-exact
/// fixed-point integer LANCZOS ([`resize_lanczos_u8`]), so it's bit-identical to PIL's `Image.LANCZOS`.
pub fn preprocess_i2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len()
        != mlx_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(Error::Msg(format!(
            "i2v image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit: scale so the image covers the target, then round to integer dims (PIL `round`).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = py_round(iw as f64 * scale).max(tw);
    let nh = py_round(ih as f64 * scale).max(th);
    let resized: Vec<f32> = if (nh, nw) == (ih, iw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, nh, nw)?
    };
    // Center-crop the (integer-valued) resized HWC buffer to (th, tw), then normalize → CHW [-1,1].
    let x1 = (nw - tw) / 2;
    let y1 = (nh - th) / 2;
    let mut chw = vec![0f32; 3 * th * tw];
    let plane = th * tw;
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Array::from_slice(&chw, &[3, th as i32, tw as i32]))
}

/// The I2V-14B 4-channel temporal mask `[4, T_lat, h_lat, w_lat]` (f32): `1.0` for the first latent
/// temporal frame (all 4 channels, all spatial), `0.0` elsewhere. The reference builds this via a
/// `ones`/`zeros` → `repeat(first,4)` → `reshape(·,T_lat,4,·,·)` → `transpose` dance over the
/// `[1, F, h_lat, w_lat]` per-frame mask (first frame 1, rest 0); the result is exactly this pattern
/// (the per-frame mask collapses to "the first 4 of `F+3` temporal slots", which is latent frame 0).
fn build_i2v_mask(t_lat: usize, h_lat: usize, w_lat: usize) -> Array {
    let plane = h_lat * w_lat;
    let mut data = vec![0f32; 4 * t_lat * plane];
    for c in 0..4 {
        let base = c * t_lat * plane; // temporal index 0 of channel c
        for p in 0..plane {
            data[base + p] = 1.0;
        }
    }
    Array::from_slice(&data, &[4, t_lat as i32, h_lat as i32, w_lat as i32])
}

/// Build the I2V-14B channel-concat conditioning `y = [mask(4), z_video(16)]` → `[20, T_lat, h_lat,
/// w_lat]` (f32). Port of `generate_wan.py`'s `is_i2v_channel_concat` branch: a conditioning video
/// (first frame = the preprocessed image, the remaining `frames−1` zero) is encoded by the 2.1 z16
/// `WanVae` → `z_video [16, T_lat, …]`, and concatenated under the temporal mask. `vae` must carry
/// encoder weights. The result is `Some(y)` fed to [`denoise_moe`].
pub fn build_i2v_y(
    vae: &WanVae,
    image: &Image,
    frames: usize,
    height: u32,
    width: u32,
    vae_stride: (usize, usize, usize),
) -> Result<Array> {
    let (h, w) = (height as i32, width as i32);
    // `frames == 0` would make both the `frames − 1` zero-pad count (negative i32) and the `t_lat`
    // subtraction (usize underflow) bogus; reject it up front (F-007).
    let frames_minus_1 = frames
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("wan build_i2v_y: frames must be >= 1".to_string()))?;
    // Conditioning video [3, F, H, W]: first frame = image, rest zeros.
    let first = preprocess_i2v_image(image, width, height)?.reshape(&[3, 1, h, w])?;
    let rest = Array::zeros::<f32>(&[3, frames_minus_1 as i32, h, w])?;
    let video = concatenate_axis(&[&first, &rest], 1)?; // [3, F, H, W]

    // VAE-encode → [1, 16, T_lat, h_lat, w_lat], drop the batch axis → [16, T_lat, h_lat, w_lat].
    let z_video = vae.encode(&video.reshape(&[1, 3, frames as i32, h, w])?)?;
    let z_video = z_video.reshape(&z_video.shape()[1..])?;

    let t_lat = frames_minus_1 / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    let mask = build_i2v_mask(t_lat, h_lat, w_lat);

    Ok(concatenate_axis(&[&mask, &z_video], 0)?)
}

// ===========================================================================================
// TI2V-5B mask-blend conditioning (port of `generate_wan.py`'s `is_i2v_mask_blend` setup + i2v_utils)
// ===========================================================================================

/// Preprocess a TI2V conditioning image to **channels-last** `[1, 1, height, width, 3]` f32 in
/// `[-1, 1]` (batch + temporal dims), the layout the z48 [`Wan22Vae::encode`] consumes. Reuses the
/// PIL-exact cover-fit LANCZOS + center-crop pipeline of [`preprocess_i2v_image`] (which returns CHW),
/// then moves channels last + adds the batch/temporal axes. Mirrors `i2v_utils.preprocess_image`.
pub fn preprocess_ti2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let chw = preprocess_i2v_image(image, width, height)?; // [3, H, W]
    Ok(chw
        .transpose_axes(&[1, 2, 0])?
        .expand_dims(0)?
        .expand_dims(0)?) // [1, 1, H, W, 3]
}

/// Build the TI2V-5B mask-blend tensors (port of `i2v_utils.build_i2v_mask`, generalized to the
/// Wan-native multi-keyframe pins of epic 3040 and to per-pin **strength**, sc-19571):
///  - `mask` `[z, T_lat, h_lat, w_lat]` (f32): `1 − strength` at each pinned latent temporal frame
///    (all channels/spatial), `1.0` elsewhere. `strength = 1.0` fully freezes the frame (mask `0`,
///    the historical hard pin); `strength = 0.0` leaves it free to denoise; anything between is a
///    partial pin, which is the whole point of the knob.
///  - `mask_tokens` `[1, L]` (f32): the channel-0 mask subsampled to the patch grid (`1 − strength`
///    for each pinned frame's tokens, `1.0` for the rest), `L` = the DiT patch-token count
///    `(T_lat/pt)·(h_lat/ph)·(w_lat/pw)`. Token order is temporal-slowest (matching
///    [`crate::patchify::patchify`]).
///
/// `pins` are `(latent_frame, strength)` pairs: first_last_frame is `[(0, s0), (t_lat-1, s1)]`, a
/// single `Reference` image is `[(0, s)]`. Latent indices `>= t_lat` are ignored (the caller
/// validates); the same weight drives both the clean-latent blend and the per-token diffusion
/// timestep, so one number controls the whole pin. **Byte-for-byte the same construction as
/// candle's `candle_gen_wan::pipeline::build_ti2v_mask`** — the two lanes must not diverge here.
pub fn build_ti2v_mask(
    pins: &[(usize, f32)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch_size: (usize, usize, usize),
) -> (Array, Array) {
    let plane = h_lat * w_lat;
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        for &(t, strength) in pins.iter().filter(|&&(t, _)| t < t_lat) {
            let base = (c * t_lat + t) * plane;
            mask[base..base + plane].fill(1.0 - strength);
        }
    }
    let mask = Array::from_slice(
        &mask,
        &[z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32],
    );

    let (pt, ph, pw) = patch_size;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tok = vec![1f32; tg * hg * wg];
    for &(t, strength) in pins {
        let tg_idx = t / pt;
        if tg_idx < tg {
            tok[(tg_idx * hg * wg)..((tg_idx + 1) * hg * wg)].fill(1.0 - strength);
        }
    }
    let mask_tokens = Array::from_slice(&tok, &[1, (tg * hg * wg) as i32]);
    (mask, mask_tokens)
}

/// Scatter per-keyframe latents into a single `[z, T_lat, h, w]` clean latent for the multi-keyframe
/// mask-blend (epic 3040): each `(z_k, idx)` (with `z_k` shaped `[z, 1, h, w]`) is placed at temporal
/// frame `idx`; every other frame is zeros (those frames have mask `1`, so the zero is never read). The
/// resulting latent feeds [`ti2v_blend_init`] + [`denoise_ti2v`] as the `z_img` per-frame conditioning.
pub fn build_ti2v_keyframe_z(
    frames: &[(Array, usize)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
) -> Result<Array> {
    let zero = Array::zeros::<f32>(&[z_dim as i32, 1, h_lat as i32, w_lat as i32])?;
    let mut slices: Vec<Array> = (0..t_lat).map(|_| zero.clone()).collect();
    for (z_k, idx) in frames {
        if *idx < t_lat {
            slices[*idx] = z_k.clone();
        }
    }
    let refs: Vec<&Array> = slices.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Blend the encoded image latent with the initial noise for the TI2V start: `latents = (1−mask)·
/// z_img + mask·noise` (port of `generate_wan.py`'s `is_i2v_mask_blend` init). `z_img` is `[z,1,h,w]`
/// (broadcasts over the noise's `T_lat`), `mask`/`noise` are `[z,T_lat,h,w]`.
pub fn ti2v_blend_init(z_img: &Array, mask: &Array, noise: &Array) -> Result<Array> {
    let one_minus_mask = subtract(scalar(1.0), mask)?;
    Ok(add(
        &multiply(&one_minus_mask, z_img)?,
        &multiply(mask, noise)?,
    )?)
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

/// The decode mode the product's automatic Wan VAE planner selects at one free-memory budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannedDecodeMode {
    /// The single-pass decode fits the safe budget (the planner returns no tiling).
    SinglePass,
    /// The largest candidate tile that fits the safe budget.
    Tiled,
    /// Not even the smallest tile fits: the run refuses catchably before its denoise.
    OverBudget,
}

impl PlannedDecodeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SinglePass => "single-pass",
            Self::Tiled => "tiled",
            Self::OverBudget => "over-budget",
        }
    }
}

/// The product's automatic decode decision at `free_bytes` of free memory, priced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedDecode {
    pub mode: PlannedDecodeMode,
    /// The safe budget the planner chose against: `free × 0.85` (`wan_vae_safe_budget_gib`'s
    /// free-aware arm). A run given `WAN_VAE_BUDGET_GIB` = this value makes the same decision.
    pub safe_budget_gib: f64,
    /// The priced decode working set: the chosen plan's cost, or the conservative single-pass cost
    /// when no plan fits (`OverBudget`).
    pub working_set_bytes: u64,
}

/// SC-20686 / E8: price a Wan VAE decode at the decision the product's automatic planner
/// ([`auto_tiling_budgeted`] for z48 -- bf16, as the dense 5B decodes -- and
/// [`auto_tiling_budgeted_z16`] for z16) makes when `free_bytes` are free at decode time, instead of
/// the conservative single-pass bound. `frames` are the decoded output frames the planner sees
/// (z16: the full non-causal `4 · T_lat`). `None` for a non-Wan VAE or invalid geometry.
pub fn planned_video_decode(
    vae: VaeTiling,
    width: u32,
    height: u32,
    frames: u32,
    free_bytes: u64,
) -> Option<PlannedDecode> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let safe_budget_gib = free_aware_budget_gib(free_bytes as f64 / GIB, WAN_VAE_BUDGET_SAFE_FRAC);
    let (h, w, f) = (
        i32::try_from(height).ok()?,
        i32::try_from(width).ok()?,
        i32::try_from(frames).ok()?,
    );
    let plan = if vae == Wan22Vae::VAE_TILING {
        plan_vae22_tiling(h, w, f, safe_budget_gib, true)
    } else if vae == WanVae::VAE_TILING {
        plan_z16_tiling(h, w, f, safe_budget_gib)
    } else {
        return None;
    };
    let single_pass = video_decode_peak_bytes_for_vae_and_tiling(vae, width, height, frames, None)?;
    let (mode, working_set_bytes) = match plan {
        Ok(None) => (PlannedDecodeMode::SinglePass, single_pass),
        Ok(Some(tiling)) => (
            PlannedDecodeMode::Tiled,
            video_decode_peak_bytes_for_vae_and_tiling(vae, width, height, frames, Some(&tiling))?,
        ),
        Err(_) => (PlannedDecodeMode::OverBudget, single_pass),
    };
    Some(PlannedDecode {
        mode,
        safe_budget_gib,
        working_set_bytes,
    })
}

#[cfg(test)]
mod planned_decode_tests;

#[cfg(test)]
#[path = "sc20686_hooks_tests.rs"]
mod sc20686_hooks_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod denoise_activation_tests;
