//! Request geometry: alignment, area / grid refusals, sampler knobs and latent shape.

use super::*;

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
