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
use mlx_gen::tiling::{budgeted_plan, TileCandidates, TilingBudgetError, TilingConfig, VaeTiling};
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
    if max_area == 0 {
        return Ok(());
    }
    let (w, h) = (
        align_dim(req.width, 1, dw as usize),
        align_dim(req.height, 1, dh as usize),
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
const VAE22_ACCUM_BYTES_PER_VOXEL: f64 = 40.0;
const VAE22_TILE_BYTES_PER_OUT_VOXEL: f64 = 3800.0;
/// Per-tile coefficient for a **bf16** decode (sc-5039). Measured on the same real-weight rig at two
/// tiles of the 1024×576×97 video (cosine 0.99995, no NaN both): 768 px / 64-frame → **79.7 GB**
/// (vs 97.7 GB f32), 640 px / 48-frame → **55.1 GB**. The per-tile term only drops to ~85 % of f32
/// — *not* 50 % — because the `RMS_norm` channel-L2 reduction stays f32 and materializes a full-size
/// f32 temporary of each activation. Calibrated to the **higher** of the two implied coefficients
/// (the 640/48 point) so the estimate never under-shoots a real peak — the 3100 first guess let the
/// selector pick a tile that measured 55.1 GB, just over the 54.4 GB safe line at the 64 GB tier.
/// The fixed accumulator term is unchanged (the blend buffers are f32 either way).
const VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16: f64 = 3400.0;

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
    let tile_coeff = if bf16 {
        VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16
    } else {
        VAE22_TILE_BYTES_PER_OUT_VOXEL
    };
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (VAE22_ACCUM_BYTES_PER_VOXEL * out_voxels + tile_coeff * tile_voxels) / GIB
}

/// Candidate spatial tile sizes (output px, multiples of the vae22 ×16 spatial scale, overlap 64).
const VAE22_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames (matching the preset
/// overlaps: 24 for the longer tiles, 16/8 for the shorter).
const VAE22_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];

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
    };
    budgeted_plan(
        VaeTiling::WAN22,
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        |of, oh, ow, tf, th, tw| estimated_vae22_decode_peak_gib(of, oh, ow, tf, th, tw, bf16),
    )
    .map_err(|e| wan_budget_error("z48 vae22", width, height, out_frames, e))
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
const VAE16_ACCUM_BYTES_PER_VOXEL: f64 = 64.0;
/// Per-tile-output-voxel cost of the z16 decoder working set (conv stack + ×8 spatial / ×4 temporal
/// upsample). Fit from the same anchors at ~6355 B/voxel (≈1.7× the z48 `vae22`'s 3800 — the bigger
/// spatial upsample); rounded **up** to 6500. z16 decodes f32 in production, so there is no bf16
/// coefficient (unlike `vae22`, sc-5039).
const VAE16_TILE_BYTES_PER_OUT_VOXEL: f64 = 6500.0;

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
    (VAE16_ACCUM_BYTES_PER_VOXEL * out_voxels + VAE16_TILE_BYTES_PER_OUT_VOXEL * tile_voxels) / GIB
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
    let candidates = TileCandidates {
        spatial_px: &VAE16_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &VAE16_TEMPORAL_FR,
    };
    budgeted_plan(
        VaeTiling::WAN,
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
}

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
    let cross_kv = transformer.prepare_cross_kv(&context_batch)?;
    let (cos, sin) = transformer.prepare_rope(grid)?;
    let mut to_eval: Vec<&Array> = vec![&cos, &sin];
    for (k, v) in &cross_kv {
        to_eval.push(k);
        to_eval.push(v);
    }
    mlx_rs::transforms::eval(to_eval)?;
    Ok(StepCache {
        cross_kv,
        cos,
        sin,
        batch,
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
    let x = match y {
        Some(y) => concatenate_axis(&[latents, y], 0)?,
        None => latents.clone(),
    };
    let preds =
        transformer.forward_cached(&x, t, &cache.cross_kv, &cache.cos, &cache.sin, cache.batch)?;
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
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: run the DiT's fusable elementwise glue (adaLN affine, gated residual, gated-GELU FFN,
    // RoPE rotation) through `mx.compile` — bit-exact (proven `max|Δ|=0` real + tiny, perf.rs /
    // compile_parity.rs) and ~14% faster/step at production geometry. Scoped + restored on drop by the
    // RAII guard (F-006/F-007) instead of leaking the process-global toggle on.
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
        let pred = predict(transformer, &latents, t, &cache, guidance, None)?;
        latents = sched.step(&pred, &latents)?;
        // Force evaluation each step to bound the lazy graph's peak memory (the reference's
        // per-step `mx.eval(latents)`).
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
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking the process-global toggle on.
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
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking the process-global on.
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
    // the F-070 `reshape(-1)` workaround: on MLX 0.32.0 reshape is int64-safe past `i32::MAX`
    // (verified in `mlx-gen/tests/mlx_write_bound_probe.rs`), so `f·h·w·3 > i32::MAX` (1920×1088@349f
    // ≈ 2.19e9; 4K@89f) now renders rather than overflowing. Byte-identical to the old reshape below-bound.
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

/// Build the TI2V-5B mask-blend tensors (port of `i2v_utils.build_i2v_mask`):
///  - `mask` `[z, T_lat, h_lat, w_lat]` (f32): `0.0` for the first latent temporal frame (all
///    channels/spatial), `1.0` elsewhere — the latent the first frame is frozen, the rest denoise.
///  - `mask_tokens` `[1, L]` (f32): the channel-0 mask subsampled to the patch grid (`0.0` for the
///    first-frame tokens, `1.0` for the rest), `L` = the DiT patch-token count `(T_lat/pt)·(h_lat/ph)·
///    (w_lat/pw)`. Token order is temporal-slowest (matching [`crate::patchify::patchify`]).
pub fn build_ti2v_mask(
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch_size: (usize, usize, usize),
) -> (Array, Array) {
    let plane = h_lat * w_lat;
    // mask: 1.0 everywhere except temporal index 0 (= 0.0).
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        let base = c * t_lat * plane; // temporal index 0 of channel c
        for p in 0..plane {
            mask[base + p] = 0.0;
        }
    }
    let mask = Array::from_slice(
        &mask,
        &[z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32],
    );

    // mask_tokens: subsample channel 0 by the patch grid. mask is 0 only at temporal index 0, so a
    // token is 0 iff its source temporal index `t'·pt == 0` (i.e. `t' == 0`) → the first `hg·wg`
    // tokens (temporal-slowest order) are 0, the rest 1.
    let (pt, ph, pw) = patch_size;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tok = vec![1f32; tg * hg * wg];
    for v in tok.iter_mut().take(hg * wg) {
        *v = 0.0;
    }
    let mask_tokens = Array::from_slice(&tok, &[1, (tg * hg * wg) as i32]);
    (mask, mask_tokens)
}

/// Multi-keyframe generalization of [`build_ti2v_mask`] (epic 3040, Wan-native first_last_frame):
/// pin the latent temporal frames in `indices` (mask `0.0` there, `1.0` elsewhere) instead of only
/// frame 0. first_last_frame = `indices = [0, t_lat-1]`. `mask` `[z, T_lat, h, w]` + `mask_tokens`
/// `[1, L]` (the `hg·wg` tokens of each pinned frame are `0`). Indices must be `< t_lat`; out-of-range
/// indices are ignored (the caller validates). With `indices = [0]` this is exactly `build_ti2v_mask`.
pub fn build_ti2v_multi_mask(
    indices: &[usize],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch_size: (usize, usize, usize),
) -> (Array, Array) {
    let plane = h_lat * w_lat;
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        for &t in indices {
            if t >= t_lat {
                continue;
            }
            let base = (c * t_lat + t) * plane;
            for p in 0..plane {
                mask[base + p] = 0.0;
            }
        }
    }
    let mask = Array::from_slice(
        &mask,
        &[z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32],
    );

    let (pt, ph, pw) = patch_size;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tok = vec![1f32; tg * hg * wg];
    for &t in indices {
        let tg_idx = t / pt;
        if tg_idx >= tg {
            continue;
        }
        for k in 0..(hg * wg) {
            tok[tg_idx * hg * wg + k] = 0.0;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the actual callback order for all four curated samplers. Heun and DPM++ SDE add
    /// intermediate evaluations, but their sigma streams still cross the expert boundary exactly
    /// once and never return to the high expert.
    #[test]
    fn every_wan_curated_sampler_crosses_the_expert_boundary_once_per_eval() {
        use mlx_gen::gen_core::sampling::{sampler_by_name, CpuLatentOps, FlowModelSampling};
        use mlx_gen::TimestepConvention;

        let sigmas = compute_sigmas(6, 5.0, 1000);
        let boundary = 875.0_f32;
        let ops = CpuLatentOps;
        let model = FlowModelSampling::new(TimestepConvention::Sigma);
        for name in ["euler_ancestral", "heun", "dpmpp_sde", "ddim"] {
            let sampler = sampler_by_name::<CpuLatentOps>(name).expect("curated sampler");
            let mut routed_high = Vec::new();
            let mut denoise = |_x: &Vec<f32>, sigma: f32| {
                routed_high.push((sigma * 1000.0).trunc() >= boundary);
                Ok(vec![0.0])
            };
            sampler
                .sample(&ops, &model, &mut denoise, vec![1.0], &sigmas, 42)
                .unwrap();

            assert!(
                routed_high.contains(&true),
                "{name} must evaluate the high expert"
            );
            assert!(
                routed_high.contains(&false),
                "{name} must evaluate the low expert"
            );
            let transitions = routed_high
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count();
            assert_eq!(
                transitions, 1,
                "{name} must make exactly one high-to-low swap"
            );
            assert!(
                routed_high.windows(2).all(|pair| pair[0] || !pair[1]),
                "{name} must never route low-to-high"
            );
        }
    }

    // ── sc-12736: the A14B expert-swap residency contract (mlx Pillar-1). ────────────────────────

    /// [`crossing_index`] is the prefix/suffix boundary the resident loop and the sequential swap both
    /// split on — it must be **exactly** the per-step `t ≥ boundary` choice. Pinned against a
    /// hand-rolled per-step scan over a monotonically-decreasing integer-timestep schedule (T2V's 875
    /// boundary), including the two corners (never crossed / crossed at step 0).
    #[test]
    fn crossing_index_equals_the_per_step_boundary_choice() {
        let boundary = 875.0_f32; // 0.875 · 1000 (T2V)
        let timesteps = [1000.0, 950.0, 900.0, 875.0, 874.0, 500.0, 100.0, 0.0];
        let k = crossing_index(&timesteps, boundary);
        // Per-step witness: the first index whose `t < boundary` (t == boundary stays HIGH).
        let expected = timesteps.iter().position(|&t| t < boundary).unwrap();
        assert_eq!(k, expected);
        assert_eq!(k, 4, "875 (== boundary) is HIGH; 874 is the first LOW step");
        for (i, &t) in timesteps.iter().enumerate() {
            let is_high_by_split = i < k;
            let is_high_by_step = t >= boundary;
            assert_eq!(
                is_high_by_split, is_high_by_step,
                "step {i} (t={t}) disagrees between the split and the per-step rule"
            );
        }
        // Never crossed → all high (k == len); crossed at step 0 → all low (k == 0).
        assert_eq!(crossing_index(&[1000.0, 900.0, 880.0], boundary), 3);
        assert_eq!(crossing_index(&[800.0, 700.0], boundary), 0);
    }

    /// A liveness witness for the expert-swap residency tests (mirrors candle's `LiveTracker`): bumps a
    /// shared live-counter on construction and drops it on `Drop`, recording the peak concurrency and an
    /// ordered load/drop/note log.
    struct LiveTracker {
        live: std::cell::Cell<usize>,
        peak: std::cell::Cell<usize>,
        log: std::cell::RefCell<Vec<&'static str>>,
    }
    impl LiveTracker {
        fn new() -> Self {
            Self {
                live: std::cell::Cell::new(0),
                peak: std::cell::Cell::new(0),
                log: std::cell::RefCell::new(Vec::new()),
            }
        }
        fn born(&self, tag: &'static str) {
            self.live.set(self.live.get() + 1);
            if self.live.get() > self.peak.get() {
                self.peak.set(self.live.get());
            }
            self.log.borrow_mut().push(tag);
        }
        fn died(&self, tag: &'static str) {
            self.live.set(self.live.get() - 1);
            self.log.borrow_mut().push(tag);
        }
        fn note(&self, tag: &'static str) {
            self.log.borrow_mut().push(tag);
        }
    }

    /// Stands in for a loaded 14B expert: its lifetime on the live-counter is exactly the expert's
    /// residency window inside [`staged_expert_swap`].
    struct ExpertWitness<'a> {
        tracker: &'a LiveTracker,
        drop_tag: &'static str,
    }
    impl<'a> ExpertWitness<'a> {
        fn new(tracker: &'a LiveTracker, born_tag: &'static str, drop_tag: &'static str) -> Self {
            tracker.born(born_tag);
            Self { tracker, drop_tag }
        }
    }
    impl Drop for ExpertWitness<'_> {
        fn drop(&mut self) {
            self.tracker.died(self.drop_tag);
        }
    }

    /// The Pillar-1 invariant (sc-12736): the two experts are **never co-resident**. Driven through the
    /// production [`staged_expert_swap`] with `0 < k < steps` (a genuine swap), the peak live-expert
    /// count is 1 and the high expert drops (and is evicted) before the low expert loads — a drop-order
    /// witness, structural rather than a VRAM read (MLX's buffer cache makes a live memory probe blind
    /// to the drop until `clear_cache`).
    #[test]
    fn expert_swap_is_never_co_resident_and_high_drops_before_low_loads() {
        let tracker = LiveTracker::new();
        let mut state = ();
        let out = staged_expert_swap(
            3, // k: 0 < k < steps → both experts own steps → a genuine swap
            8, // steps
            &mut state,
            |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Ok(ExpertWitness::new(&tracker, "load-low", "drop-low")),
            |_w, _st| Ok(()),
            || {
                tracker.note("evict");
                Ok(())
            },
        );
        assert!(out.is_ok());
        assert_eq!(
            tracker.peak.get(),
            1,
            "the two 14B experts must NEVER be co-resident (the whole Pillar-1 win)"
        );
        assert_eq!(
            *tracker.log.borrow(),
            vec![
                "load-high",
                "drop-high",
                "evict", // high freed + cache flushed …
                "load-low",
                "drop-low",
                "evict", // … BEFORE low ever loads
            ],
            "high must be dropped AND evicted before the low expert loads (evict-then-load)"
        );
    }

    /// Mutation-check (sc-12736 acceptance): force both experts resident and confirm the
    /// never-co-resident assertion regresses — proving the passing test above is not a default-value
    /// false green. This is the exact both-resident behavior the sequential path removes (the resident
    /// `denoise_moe` holds both for the whole loop): binding `high` and `low` in one scope co-resides
    /// them, and the SAME liveness witness now reports peak concurrency 2, so `peak == 1` goes RED.
    #[test]
    fn forcing_both_experts_resident_regresses_the_never_co_resident_assertion() {
        let tracker = LiveTracker::new();
        {
            // MUTATION: the inactive expert is NOT dropped before the next loads.
            let _high = ExpertWitness::new(&tracker, "load-high", "drop-high");
            let _low = ExpertWitness::new(&tracker, "load-low", "drop-low");
            tracker.note("both-resident-denoise");
        }
        assert_eq!(
            tracker.peak.get(),
            2,
            "the forced-both-resident mutation co-resides the two experts"
        );
        assert!(
            tracker.peak.get() > 1,
            "the never-co-resident assertion (peak == 1) MUST fail under the both-resident mutation — \
             the passing test genuinely discriminates co-residence, it is not a false green"
        );
    }

    /// [`staged_expert_swap`] skips loading the expert whose step range is empty (memory-optimal single
    /// crossing): `k == 0` loads only the low expert, `k == steps` only the high expert.
    #[test]
    fn expert_swap_skips_the_expert_that_owns_no_steps() {
        // k == 0 → all-low: the high loader is never called.
        let low_only = LiveTracker::new();
        let mut st = ();
        staged_expert_swap(
            0,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&low_only, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Ok(ExpertWitness::new(&low_only, "load-low", "drop-low")),
            |_w, _st| Ok(()),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(
            *low_only.log.borrow(),
            vec!["load-low", "drop-low"],
            "k == 0 must load ONLY the low expert"
        );

        // k == steps → all-high: the low loader is never called.
        let high_only = LiveTracker::new();
        let mut st = ();
        staged_expert_swap(
            8,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&high_only, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Ok(ExpertWitness::new(&high_only, "load-low", "drop-low")),
            |_w, _st| Ok(()),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(
            *high_only.log.borrow(),
            vec!["load-high", "drop-high"],
            "k == steps must load ONLY the high expert"
        );
    }

    /// The evict runs after each expert is **used and dropped**, before the next loads (sc-12736): the
    /// MLX free-after-drop discipline — `clear_cache` returns the dropped expert's pages to the OS once
    /// its arrays are freed and before the incoming expert allocates.
    #[test]
    fn expert_swap_evicts_after_each_expert_drops_and_before_the_next_loads() {
        let tracker = LiveTracker::new();
        let mut st = ();
        staged_expert_swap(
            3,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
            |_w, _st| {
                tracker.note("use-high");
                Ok(())
            },
            |_st| Ok(ExpertWitness::new(&tracker, "load-low", "drop-low")),
            |_w, _st| {
                tracker.note("use-low");
                Ok(())
            },
            || {
                tracker.note("evict");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            *tracker.log.borrow(),
            vec![
                "load-high",
                "use-high",
                "drop-high",
                "evict", //
                "load-low",
                "use-low",
                "drop-low",
                "evict",
            ],
            "each expert must be used, dropped, then evicted before the next loads"
        );
    }

    /// A load failure on the low expert still drops the (already-used-and-evicted) high expert via the
    /// explicit drop before the `?` — no leak, peak stays 1, and the error propagates.
    #[test]
    fn expert_swap_propagates_a_low_load_failure_after_dropping_high() {
        let tracker = LiveTracker::new();
        let mut st = ();
        let out: Result<()> = staged_expert_swap(
            3,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Err(Error::Msg("low expert OOM".into())),
            |_w: &ExpertWitness, _st| Ok(()),
            || Ok(()),
        );
        assert!(matches!(out, Err(Error::Msg(_))));
        assert_eq!(
            *tracker.log.borrow(),
            vec!["load-high", "drop-high"],
            "high must have dropped before the low load was even attempted"
        );
        assert_eq!(tracker.peak.get(), 1);
    }

    /// **sc-12349: the z16 selector tiles past the write bound**, no matter how much memory the machine
    /// has. Wan z16 writes 96 channels at full output resolution, so at 720p only 24 output frames fit
    /// under `MAX_WRITABLE_ELEMS` — while the shipped `frame_num` is 81.
    ///
    /// On MLX 0.31.2 taking the single pass past the cap decoded silently-wrong pixels (the conv3d
    /// corruption). **sc-12748: on 0.32.0 that single pass is now correct** (#3524), so this cap is
    /// defense-in-depth rather than a correctness necessity — but the selector still tiles past it (a
    /// cheap, always-correct default), which this pins on an unlimited budget (the sharpest form).
    #[test]
    fn z16_tiles_past_the_write_bound_on_an_unlimited_budget() {
        let (h, w, f) = (720i32, 1280i32, 81i32);
        assert!(
            (f as i64) > VaeTiling::WAN.writable_frame_cap(h, w),
            "test precondition: 81 frames at 720p must exceed the z16 write cap ({})",
            VaeTiling::WAN.writable_frame_cap(h, w)
        );

        let cfg = plan_z16_tiling(h, w, f, f64::INFINITY)
            .expect("an unlimited budget must still yield a plan, not a budget error")
            .expect(
                "past the write bound the z16 decode MUST tile even with infinite memory — None here \
                 is the silently-wrong single pass sc-12349 exists to prevent",
            );

        // The chosen tile must itself be writable. z16's smallest temporal candidate is 32 frames,
        // above the 24-frame cap, so spatial tiling has to carry it — which is exactly why the bound
        // is checked against the whole tile volume rather than the frame count alone.
        let tf = cfg
            .temporal
            .map(|t| t.tile_frames as i64)
            .unwrap_or(f as i64);
        let (th, tw) = cfg
            .spatial
            .map(|s| {
                (
                    (s.tile_px as i64).min(h as i64),
                    (s.tile_px as i64).min(w as i64),
                )
            })
            .unwrap_or((h as i64, w as i64));
        let write = VaeTiling::WAN.full_res_channels as i64 * tf * th * tw;
        assert!(
            write <= mlx_gen::tiling::MAX_WRITABLE_ELEMS,
            "the selected z16 tile writes {write} elements, past the bound — plan {cfg:?}"
        );
    }

    #[test]
    fn denoise_peak_estimate_matches_5b_measurements() {
        // sc-4986 anchors (real 5B, dim 3072, batch 2 / CFG on). bf16 model.safetensors ≈ 11.5 GiB
        // resident; measured total denoise peak at L tokens. The estimate must land within ~2 GiB.
        let weights = (11.5 * 1024.0 * 1024.0 * 1024.0) as u64;
        for (tokens, measured_peak) in [(1760usize, 11.2_f64), (16720, 17.5), (32560, 24.9)] {
            let est = estimated_denoise_peak_gib(weights, tokens, 3072, true);
            assert!(
                (est - measured_peak).abs() < 2.0,
                "L={tokens}: estimate {est:.1} GiB vs measured {measured_peak:.1} GiB"
            );
        }
    }

    #[test]
    fn denoise_peak_scales_with_cfg_and_tokens() {
        let w = 10u64 << 30; // 10 GiB
                             // CFG doubles the activation term.
        let on = estimated_denoise_peak_gib(w, 32560, 3072, true);
        let off = estimated_denoise_peak_gib(w, 32560, 3072, false);
        assert!(
            on - 10.0 > 1.9 * (off - 10.0),
            "CFG should ~2× the activation term"
        );
        // Monotonic in tokens.
        assert!(estimated_denoise_peak_gib(w, 40000, 3072, true) > on);
    }

    #[test]
    fn guard_rejects_over_budget_and_passes_under() {
        use mlx_rs::memory::set_memory_limit;
        // Pin a deterministic budget (32 GiB) so the threshold is exercised on any machine, then
        // restore. set_memory_limit returns the previous value.
        let prev = set_memory_limit(32 << 30);
        // A 14B-class resident (two bf16 experts ≈ 56 GiB) blows the 32 GiB budget on weights alone.
        let res = preflight_denoise_memory_guard("wan_test", 56 << 30, 1024, 5120, true);
        // A tiny model + small request fits comfortably (10 GiB + ~1.5 GiB acts < 27 GiB safe).
        let ok = preflight_denoise_memory_guard("wan_test", 10 << 30, 5280, 3072, true);
        set_memory_limit(prev);
        assert!(
            res.is_err(),
            "56 GiB resident must be rejected under a 32 GiB budget"
        );
        assert!(ok.is_ok(), "11.5 GiB peak must pass under a 32 GiB budget");
    }

    #[test]
    fn resolve_sampler_knobs_falls_back_to_defaults_then_request() {
        // Unset request fields take the config defaults; an unset sampler → UniPC; the seed is some
        // value (drawn fresh). This is the byte-identical inline block the four generate paths used.
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let (steps, shift, kind, _seed) = resolve_sampler_knobs(&req, 40, 5.0);
        assert_eq!(steps, 40);
        assert_eq!(shift, 5.0);
        assert_eq!(kind, SolverKind::UniPC);

        // Explicit request fields win over the defaults, and the sampler name maps through.
        let req = GenerationRequest {
            prompt: "x".into(),
            steps: Some(12),
            scheduler_shift: Some(3.5),
            sampler: Some("euler".into()),
            seed: Some(99),
            ..Default::default()
        };
        let (steps, shift, kind, seed) = resolve_sampler_knobs(&req, 40, 5.0);
        assert_eq!((steps, shift, kind, seed), (12, 3.5, SolverKind::Euler, 99));
    }

    // --- sc-4998: memory-budgeted z48 vae22 decode tiling ---------------------------------------

    /// The estimated peak of the chosen tiling, recomputed from a returned config + output dims (the
    /// largest tile spans `min(tile_px, dim)` on each spatial axis and `min(tile_frames, f)` frames).
    fn chosen_peak_gib(cfg: &TilingConfig, h: i64, w: i64, f: i64, bf16: bool) -> f64 {
        let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
        let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(f))
            .unwrap_or(f);
        estimated_vae22_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w, bf16)
    }

    #[test]
    fn vae22_decode_peak_matches_wedge_anchors() {
        // sc-4998 f32 anchors (real 5B z48 vae22, M5 Max): the model must reproduce both within ~10 %.
        // A: 1024×576×97 video, 512 px / 64-frame tile → 60 GB.
        let a = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 512, 512, false);
        assert!((a - 60.0).abs() < 6.0, "anchor A estimate {a:.1} GiB vs 60");
        // B: 1280×704×145 video, 256 px / 32-frame tile → 12.6 GB.
        let b = estimated_vae22_decode_peak_gib(145, 704, 1280, 32, 256, 256, false);
        assert!(
            (b - 12.6).abs() < 2.0,
            "anchor B estimate {b:.1} GiB vs 12.6"
        );
        // sc-5039 bf16 anchors (real-weight, 1024×576×97): 768 px/64 f → 79.7 GB, 640 px/48 f →
        // 55.1 GB. bf16 must estimate below f32 and stay **conservative** (never below the measured
        // peak — the guard must not under-shoot). Tile dims use the selector's nominal frame count.
        let bf16_768 = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 576, 768, true);
        let f32_768 = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 576, 768, false);
        assert!(bf16_768 < f32_768, "bf16 peak must be below f32");
        assert!(
            bf16_768 >= 79.7,
            "bf16 768/64 estimate {bf16_768:.1} under-shoots 79.7"
        );
        let bf16_640 = estimated_vae22_decode_peak_gib(97, 576, 1024, 48, 576, 640, true);
        assert!(
            bf16_640 >= 55.1,
            "bf16 640/48 estimate {bf16_640:.1} under-shoots 55.1"
        );
    }

    #[test]
    fn vae22_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single-pass decode comfortably → no tiling.
        let plan = plan_vae22_tiling(256, 256, 33, 40.0, false).unwrap();
        assert!(
            plan.is_none(),
            "small clip should not need tiling: {plan:?}"
        );
    }

    #[test]
    fn vae22_tiling_bounds_moderate_res_peak() {
        // The regression: 1024×576×97 on a 64 GiB machine. The px-threshold `auto` chose 512 px tiles
        // → ~60 GB. The budgeted plan must tile, keep the peak under the safe budget, and crucially
        // below the 60 GB blow-up that OOMs a 64 GiB Mac.
        let safe = 64.0 * 0.85; // 54.4 GiB
        let cfg = plan_vae22_tiling(576, 1024, 97, safe, false)
            .unwrap()
            .expect("moderate res must tile");
        let peak = chosen_peak_gib(&cfg, 576, 1024, 97, false);
        assert!(
            peak <= safe,
            "chosen peak {peak:.1} GiB over safe {safe:.1}"
        );
        assert!(
            peak < 60.0,
            "chosen peak {peak:.1} GiB not below the 60 GB blow-up"
        );
    }

    #[test]
    fn vae22_bf16_tiling_stays_under_budget_and_below_f32_peak() {
        // sc-5039: the bf16 plan must keep its chosen tile under the safe budget (the 3100→3400
        // coefficient fix — 3100 let a tile measure 55.1 GB, over the 54.4 GB line), and the bf16
        // peak of a *given* tile is strictly below the f32 peak of the same tile (the headroom that
        // is bf16's only real win — no wall-clock benefit). No claim that it fits a *bigger* tile:
        // at this resolution the candidate grid lands bf16 on the same 384/full-97 tile as f32.
        let safe = 64.0 * 0.85; // 54.4 GiB
        let bf16 = plan_vae22_tiling(576, 1024, 97, safe, true)
            .unwrap()
            .expect("bf16 still needs tiling at 64 GiB");
        let bf16_peak = chosen_peak_gib(&bf16, 576, 1024, 97, true);
        assert!(
            bf16_peak <= safe,
            "bf16 chosen peak {bf16_peak:.1} GiB over safe {safe:.1}"
        );
        // Same tile, bf16 vs f32: bf16 must be the lighter estimate.
        let f32_same = chosen_peak_gib(&bf16, 576, 1024, 97, false);
        assert!(
            bf16_peak < f32_same,
            "bf16 peak {bf16_peak:.1} not below f32 {f32_same:.1} for the same tile"
        );
    }

    #[test]
    fn vae22_tiling_bounds_peak_across_output_sizes() {
        // The px-threshold `auto`'s real defect was *non-monotonic* peak: a moderate 1024×576×97
        // decode spiked to 60 GB while the *larger* 1280×704×145 sat at 12.6 GB — so the dangerous
        // peak hid at a routine resolution. The budgeted plan must hold every size under the safe
        // budget (and below that 60 GB spike), regardless of how output size grows.
        let safe = 64.0 * 0.85; // 54.4 GiB
        for (h, w, f) in [
            (576i64, 1024i64, 49i64),
            (576, 1024, 97),
            (576, 1024, 145),
            (704, 1280, 145),
            (1088, 1920, 97),
        ] {
            let peak = match plan_vae22_tiling(h as i32, w as i32, f as i32, safe, false).unwrap() {
                Some(cfg) => chosen_peak_gib(&cfg, h, w, f, false),
                None => estimated_vae22_decode_peak_gib(f, h, w, f, h, w, false), // single-pass fit
            };
            assert!(
                peak <= safe && peak < 60.0,
                "{w}×{h}×{f}: peak {peak:.1} GiB not bounded under safe {safe:.1} / 60 GB spike"
            );
        }
    }

    #[test]
    fn vae22_tiling_errors_when_unfittable() {
        // A huge video under a tiny budget: even the smallest tile (and the unavoidable output
        // accumulators) cannot fit → a catchable error, not an OOM/abort.
        let err = plan_vae22_tiling(1088, 1920, 241, 8.0, false);
        assert!(err.is_err(), "over-budget decode must error, got {err:?}");
    }

    // --- sc-6894 F-009: z16 Wan 2.1 VAE decode budgeting ----------------------------------------

    /// Re-derive a z16 plan's peak the way the selector sizes its largest tile.
    fn z16_chosen_peak(cfg: &TilingConfig, h: i64, w: i64, f: i64) -> f64 {
        let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
        let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(f))
            .unwrap_or(f);
        estimated_z16_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w)
    }

    #[test]
    fn z16_decode_peak_matches_sweep_anchors() {
        // Real-weight anchors from `vae16_decode_sweep.rs` (128 GB M-series, f32). The model must be
        // CONSERVATIVE (never below the measured peak — an under-shoot is an OOM) and within ~10 %.
        // (out_f, out_h, out_w, tile_f, tile_h, tile_w, measured_gib)
        let anchors = [
            (16, 512, 512, 16, 512, 512, 25.39),   // single-pass
            (16, 768, 768, 16, 768, 768, 56.35),   // single-pass
            (32, 512, 512, 32, 512, 512, 50.12),   // single-pass (temporal scaling == spatial)
            (16, 768, 768, 16, 384, 384, 14.46),   // tiled @384 px
            (16, 1024, 1024, 16, 512, 512, 25.66), // tiled @512 px
        ];
        for (of, oh, ow, tf, th, tw, measured) in anchors {
            let est = estimated_z16_decode_peak_gib(of, oh, ow, tf, th, tw);
            assert!(
                est >= measured,
                "z16 model {est:.2} GiB UNDER-shoots measured {measured} (OOM risk) for tile \
                 [{tf},{th},{tw}] of [{of},{oh},{ow}]"
            );
            assert!(
                est <= measured * 1.10,
                "z16 model {est:.2} GiB over-conservative vs measured {measured} (>10 %)"
            );
        }
    }

    #[test]
    fn z16_tiling_single_pass_when_small() {
        // A short, low-res z16 clip fits a single-pass decode → no tiling.
        let plan = plan_z16_tiling(256, 256, 16, 60.0).unwrap();
        assert!(plan.is_none(), "small z16 clip should not tile: {plan:?}");
    }

    #[test]
    fn z16_tiling_bounds_moderate_res_peak() {
        // 1280×720×80 on a 64 GiB machine: single-pass z16 would peak ~450 GB. The budgeted plan must
        // tile and keep the recomputed peak under the safe budget (the bounded/catchable guarantee).
        let safe = 64.0 * 0.85; // 54.4 GiB
        let cfg = plan_z16_tiling(720, 1280, 80, safe)
            .unwrap()
            .expect("moderate-res z16 must tile");
        let peak = z16_chosen_peak(&cfg, 720, 1280, 80);
        assert!(
            peak <= safe,
            "z16 chosen peak {peak:.1} GiB over safe {safe:.1}"
        );
    }

    #[test]
    fn z16_tiling_errors_when_unfittable() {
        // 4K × 240 frames under an 8 GiB budget: the output accumulators alone blow it → a catchable
        // error before the decode, not a SIGKILL.
        let err = plan_z16_tiling(2160, 3840, 240, 8.0);
        assert!(
            err.is_err(),
            "over-budget z16 decode must error, got {err:?}"
        );
    }

    #[test]
    fn vae22_tiling_budgeted_reads_free_memory() {
        use mlx_rs::memory::set_memory_limit;
        // Exercise the public wrapper end-to-end. Pin a 64 GiB limit; with a low baseline residency the
        // free-aware budget (≈ 64 × 0.85) still forces the 1024×576×97 decode to tile. Restore after.
        std::env::remove_var(WAN_VAE_BUDGET_ENV); // ensure the free-probe path, not an ambient override
        let prev = set_memory_limit(64 << 30);
        let plan = auto_tiling_budgeted(576, 1024, 97, false);
        set_memory_limit(prev);
        let cfg = plan.unwrap().expect("moderate res tiles at 64 GiB free");
        assert!(cfg.spatial.is_some() || cfg.temporal.is_some());
    }

    // --- sc-12737: free-aware VAE-decode budget (contract-aligned with candle sc-12734) ----------

    #[test]
    fn free_aware_budget_is_free_times_frac_clamped() {
        // The pure arithmetic: `(limit − resident) × frac`. With 24 GiB resident under a 96 GiB limit,
        // free = 72; at 0.85 → 61.2 GiB, strictly below the total-based 96×0.85 the old code used.
        let frac = WAN_VAE_BUDGET_SAFE_FRAC;
        let (limit, resident) = (96.0, 24.0);
        let free = limit - resident;
        assert!((free_aware_budget_gib(free, frac) - free * frac).abs() < 1e-9);
        assert!(
            free_aware_budget_gib(free, frac) < limit * frac,
            "resident memory must shrink the budget vs the total-based 0.85×limit"
        );
        // Never negative even if resident somehow exceeds the limit (defensive clamp).
        assert_eq!(free_aware_budget_gib(-5.0, frac), 0.0);
    }

    #[test]
    fn free_aware_resolver_uses_probe_and_env_override_wins() {
        // With N GiB resident the probe reports (limit − resident); the resolver returns free × frac,
        // strictly below the total-based budget for the same limit.
        let frac = WAN_VAE_BUDGET_SAFE_FRAC;
        let (limit, resident) = (96.0, 30.0);
        let stub_free = move || Some(limit - resident);
        let free_budget =
            resolve_free_aware_budget("WAN_VAE_TEST_BUDGET_UNSET", frac, 16.0, stub_free);
        assert!((free_budget - (limit - resident) * frac).abs() < 1e-9);
        assert!(
            free_budget < limit * frac,
            "free-aware budget must be below the total-based budget"
        );

        // The env override wins over the live probe (the deterministic worker/test injection point).
        // Capture then remove before asserting so a failure can't leak the var into other tests.
        std::env::set_var("WAN_VAE_TEST_BUDGET", "42.5");
        let overridden = resolve_free_aware_budget("WAN_VAE_TEST_BUDGET", frac, 16.0, || Some(1.0));
        std::env::remove_var("WAN_VAE_TEST_BUDGET");
        assert_eq!(overridden, 42.5);

        // No env + no probe (limit disabled → None) → the conservative default.
        assert_eq!(
            resolve_free_aware_budget("WAN_VAE_TEST_BUDGET_UNSET", frac, 16.0, || None),
            16.0
        );
    }

    #[test]
    fn wan_vae_budget_honors_env_override() {
        // The shared resolver both tilers use must let WAN_VAE_BUDGET_GIB pin the budget (the same knob
        // candle honors). Capture then remove before asserting so a failure can't leak the var.
        std::env::set_var(WAN_VAE_BUDGET_ENV, "42.5");
        let got = wan_vae_safe_budget_gib();
        std::env::remove_var(WAN_VAE_BUDGET_ENV);
        assert_eq!(got, 42.5);
    }

    #[test]
    fn free_aware_budget_tracks_resident_memory_end_to_end() {
        // The discriminating test: prove the LIVE budget shrinks as resident memory grows — i.e. it
        // budgets against FREE (limit − active), not TOTAL. A revert to `get_memory_limit() × 0.85`
        // would leave the budget unchanged when active rises AND equal to 48×0.85 exactly — both
        // assertions below would then fail.
        use mlx_rs::memory::{get_active_memory, set_memory_limit};
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        std::env::remove_var(WAN_VAE_BUDGET_ENV); // force the free-probe path, not an ambient override
        let prev = set_memory_limit(48 << 30); // pin a deterministic 48 GiB ceiling
        let total_based = 48.0 * WAN_VAE_BUDGET_SAFE_FRAC; // what the old get_memory_limit()×0.85 gave

        let a_before = get_active_memory();
        let b_before = wan_vae_safe_budget_gib();

        // Raise resident memory by a live, materialized 1 GiB f32 array (256Mi elems, < i32::MAX).
        let resident = Array::zeros::<f32>(&[268_435_456]).unwrap();
        mlx_rs::transforms::eval([&resident]).unwrap();

        let a_after = get_active_memory();
        let b_after = wan_vae_safe_budget_gib();

        set_memory_limit(prev);

        let delta_active_gib = a_after.saturating_sub(a_before) as f64 / GIB;
        assert!(
            delta_active_gib > 0.5,
            "test precondition: the live array must raise active memory (Δ {delta_active_gib:.2} GiB)"
        );
        // Free-aware ⇒ the budget drops by ≈ Δactive × frac. A total-based budget would not move.
        let expected_drop = delta_active_gib * WAN_VAE_BUDGET_SAFE_FRAC;
        let actual_drop = b_before - b_after;
        assert!(
            (actual_drop - expected_drop).abs() < 0.25,
            "budget must track FREE memory: dropped {actual_drop:.3} GiB, expected ~{expected_drop:.3} \
             (Δactive {delta_active_gib:.3} × {WAN_VAE_BUDGET_SAFE_FRAC}). Total-based would not move."
        );
        // And it must be strictly below the total-based 48×0.85 while memory is resident (active > 0).
        assert!(
            b_after < total_based,
            "free-aware budget {b_after:.2} must be below the total-based {total_based:.2} while \
             memory is resident"
        );
        drop(resident);
    }

    #[test]
    fn align_dim_rounds_down_to_tile() {
        // patch 2 × vae_stride 8 = 16-px grid.
        assert_eq!(align_dim(1280, 2, 8), 1280);
        assert_eq!(align_dim(1281, 2, 8), 1280);
        assert_eq!(align_dim(1295, 2, 8), 1280);
        assert_eq!(align_dim(1296, 2, 8), 1296);
    }

    #[test]
    fn latent_shape_and_seq_len_match_reference_formulas() {
        // 49 frames, 512×512, z16, stride (4,8,8), patch (1,2,2).
        let ls = latent_shape(49, 512, 512, 16, (4, 8, 8)).unwrap();
        assert_eq!(ls, [16, 13, 64, 64]); // (49-1)/4+1=13, 512/8=64
        let sl = seq_len(ls, (1, 2, 2));
        // ceil(64*64/(2*2) * 13) = 1024 * 13 = 13312
        assert_eq!(sl, 13312);
    }

    #[test]
    fn latent_shape_rejects_zero_frames() {
        // frames == 0 must be a clean error, not a usize underflow → huge t_lat (F-007).
        assert!(latent_shape(0, 512, 512, 16, (4, 8, 8)).is_err());
        assert!(latent_shape(1, 512, 512, 16, (4, 8, 8)).is_ok());
    }

    #[test]
    fn cfg_combine_is_uncond_plus_gs_delta() {
        let cond = Array::from_slice(&[2.0f32, 4.0], &[2]);
        let uncond = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let got = cfg_combine(&cond, &uncond, 3.0).unwrap();
        // 1 + 3*(2-1) = 4 ; 1 + 3*(4-1) = 10
        assert_eq!(got.as_slice::<f32>(), &[4.0, 10.0]);
    }

    #[test]
    fn py_round_is_half_to_even() {
        assert_eq!(py_round(19.2), 19);
        assert_eq!(py_round(16.0), 16);
        assert_eq!(py_round(0.5), 0); // half → even (down)
        assert_eq!(py_round(1.5), 2); // half → even (up)
        assert_eq!(py_round(2.5), 2); // half → even (down)
        assert_eq!(py_round(2.500001), 3); // just over half → up
    }

    /// sc-12308 — these two tests replace `best_output_size_caps_area_and_aligns`, which pinned the
    /// silent refit `best_output_size(1280, 720, 16, 16, 704*1280) == (1264, 704)`. Both the
    /// function and that behaviour are gone: an over-cap request is now rejected, and `1280×720` is
    /// not over the 14B family's real cap at all.
    #[test]
    fn over_area_is_rejected_not_refit() {
        let req = |w, h| GenerationRequest {
            prompt: "x".into(),
            width: w,
            height: h,
            ..Default::default()
        };

        // The canonical 720p is AT the 14B cap (`>` check) and must pass untouched — this is the
        // exact geometry the old refit turned into 1264×704, off every advertised bucket.
        assert!(
            reject_over_area("t2v", &req(1280, 720), 16, 16, crate::config::MAX_AREA_14B).is_ok()
        );
        assert!(
            reject_over_area("t2v", &req(720, 1280), 16, 16, crate::config::MAX_AREA_14B).is_ok()
        );

        // Genuinely over-envelope (sc-9028's 1280×1280) is rejected, with a message naming the cap.
        let err = reject_over_area("t2v", &req(1280, 1280), 16, 16, crate::config::MAX_AREA_14B)
            .expect_err("over-area must be rejected");
        assert!(err.to_string().contains("max area"), "actionable: {err}");

        // The 5B's own budget is smaller, and ITS 720p is 704-tall: 1280×704 is exactly at cap.
        assert!(
            reject_over_area("5b", &req(1280, 704), 32, 32, crate::config::MAX_AREA_5B).is_ok()
        );
        // A 1280×720 ask does NOT trip the 5B's area cap — its 32-px grid aligns 720 down to 704
        // first, landing at 901,120. That silent 720→704 is the STRIDE floor, a separate constraint
        // the manifest handles by advertising 1280x704 for this model; it is not an area rejection.
        assert!(
            reject_over_area("5b", &req(1280, 720), 32, 32, crate::config::MAX_AREA_5B).is_ok()
        );
        // What the 5B's cap does catch is genuinely over-envelope geometry.
        assert!(
            reject_over_area("5b", &req(1280, 1280), 32, 32, crate::config::MAX_AREA_5B).is_err()
        );
        // …and the 5B must not silently inherit the 14B family's larger budget.
        const { assert!(crate::config::MAX_AREA_5B < crate::config::MAX_AREA_14B) };

        // `max_area == 0` means uncapped.
        assert!(reject_over_area("x", &req(4096, 4096), 16, 16, 0).is_ok());
    }

    #[test]
    fn over_area_is_judged_on_the_aligned_geometry() {
        let req = |w, h| GenerationRequest {
            prompt: "x".into(),
            width: w,
            height: h,
            ..Default::default()
        };
        // An off-grid request is aligned DOWN before rendering, so it must be judged on what it
        // becomes, not what was typed: 1288×724 → 1280×720 = 921,600, exactly at the cap. Judging
        // the raw 932,512 would reject a request that renders perfectly legally.
        assert_eq!((align_dim(1288, 1, 16), align_dim(724, 1, 16)), (1280, 720));
        assert!(
            reject_over_area("t2v", &req(1288, 724), 16, 16, crate::config::MAX_AREA_14B).is_ok()
        );
    }

    #[test]
    fn build_i2v_mask_is_one_at_first_latent_frame() {
        // [4, T_lat=2, 1, 1]: channel-major, temporal index 0 → 1.0, index 1 → 0.0.
        let m = build_i2v_mask(2, 1, 1);
        assert_eq!(m.shape(), &[4, 2, 1, 1]);
        assert_eq!(m.as_slice::<f32>(), &[1., 0., 1., 0., 1., 0., 1., 0.]);
    }

    #[test]
    fn build_ti2v_mask_freezes_first_frame() {
        // z=2, T_lat=2, h=w=2, patch (1,2,2) → grid (2,1,1) → L=2 tokens.
        let (mask, tokens) = build_ti2v_mask(2, 2, 2, 2, (1, 2, 2));
        assert_eq!(mask.shape(), &[2, 2, 2, 2]);
        // Per channel (8 vals): temporal 0 → 0.0 (4 spatial), temporal 1 → 1.0 (4 spatial).
        assert_eq!(
            mask.as_slice::<f32>(),
            &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.]
        );
        // Token mask: first (t'=0) token frozen (0), second (t'=1) active (1).
        assert_eq!(tokens.shape(), &[1, 2]);
        assert_eq!(tokens.as_slice::<f32>(), &[0., 1.]);
    }

    #[test]
    fn build_ti2v_multi_mask_freezes_first_and_last() {
        // first_last_frame: z=1, T_lat=3, h=w=2, patch (1,2,2) → grid (3,1,1) → 3 tokens.
        // Pin frames [0, 2] (first + last). With indices=[0] it must equal build_ti2v_mask.
        let (mask, tokens) = build_ti2v_multi_mask(&[0, 2], 1, 3, 2, 2, (1, 2, 2));
        assert_eq!(mask.shape(), &[1, 3, 2, 2]);
        // temporal 0 → 0 (4), temporal 1 → 1 (4), temporal 2 → 0 (4).
        assert_eq!(
            mask.as_slice::<f32>(),
            &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.]
        );
        // token mask: frame0 + frame2 tokens 0, frame1 token 1.
        assert_eq!(tokens.as_slice::<f32>(), &[0., 1., 0.]);
        // Single-index [0] reproduces build_ti2v_mask exactly.
        let (m1, t1) = build_ti2v_multi_mask(&[0], 2, 2, 2, 2, (1, 2, 2));
        let (m0, t0) = build_ti2v_mask(2, 2, 2, 2, (1, 2, 2));
        assert_eq!(m1.as_slice::<f32>(), m0.as_slice::<f32>());
        assert_eq!(t1.as_slice::<f32>(), t0.as_slice::<f32>());
    }

    #[test]
    fn build_ti2v_keyframe_z_scatters_frames() {
        // z=1, h=w=1, T_lat=3; place A=[7] @0 and B=[9] @2; frame1 = 0.
        let a = Array::from_slice(&[7.0f32], &[1, 1, 1, 1]);
        let b = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
        let z = build_ti2v_keyframe_z(&[(a, 0), (b, 2)], 1, 3, 1, 1).unwrap();
        assert_eq!(z.shape(), &[1, 3, 1, 1]);
        assert_eq!(z.as_slice::<f32>(), &[7.0, 0.0, 9.0]);
    }

    #[test]
    fn ti2v_blend_init_freezes_first_frame() {
        // z=1,T=2,h=w=1: mask 0 at t=0, 1 at t=1. z_img=[9] (frame0), noise=[5,7].
        let z_img = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
        let mask = Array::from_slice(&[0.0f32, 1.0], &[1, 2, 1, 1]);
        let noise = Array::from_slice(&[5.0f32, 7.0], &[1, 2, 1, 1]);
        let out = ti2v_blend_init(&z_img, &mask, &noise).unwrap();
        // frame0 = z_img (9), frame1 = noise (7).
        assert_eq!(out.as_slice::<f32>(), &[9.0, 7.0]);
    }
}
