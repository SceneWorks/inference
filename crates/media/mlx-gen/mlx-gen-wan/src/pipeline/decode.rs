//! Latent-to-frame decode and the planned (tiled / single-pass) video decode.

use super::*;

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
/// [`Image`] per frame. Delegates to the shared [`mlx_gen::image::u8_frames_to_images`] (sc-27319):
/// shape-checked, forced to logical row-major order before the host read (the tensor is
/// transpose-strided), and split frame-major as `[F, H·W·3]` so a clip past `i32::MAX` total
/// elements still renders (sc-12748).
pub fn frames_to_images(frames_u8: &Array) -> Result<Vec<Image>> {
    mlx_gen::image::u8_frames_to_images(frames_u8)
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
