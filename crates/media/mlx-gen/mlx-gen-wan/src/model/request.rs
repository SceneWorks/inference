//! Request resolution shared by the providers: I2V reference, TI2V pins, strengths and capped
//! dimensions.

use super::*;

/// The single conditioning reference image for I2V (the first video frame) **and its resolved
/// conditioning strength**, if present.
///
/// sc-19571: this used to return the image alone, which is how the TI2V mask-blend came to build a
/// hard 0/1 pin no matter what the caller asked for. The per-`Reference` `strength` wins over the
/// request-level img2img `strength`, falling back to `1.0` (a full pin) — the identical resolution
/// candle's `candle_gen_wan` TI2V path performs, so the two lanes read one number the same way.
pub(super) fn i2v_reference(req: &GenerationRequest) -> Option<(&Image, f32)> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, strength } => {
            Some((image, strength.or(req.strength).unwrap_or(1.0)))
        }
        _ => None,
    })
}

/// **The TI2V mask-blend seam** (sc-19571): resolve the request's image conditioning into the
/// `(latent_frame, strength)` pins that drive [`build_ti2v_mask`], and nothing else.
///
/// * `Keyframe`s (first_last_frame / multi-keyframe) → one pin each, **in request order**, so the
///   caller can zip the pins against `req.keyframes()` to pair each resolved index with its image.
///   `frame_idx` is a latent index with negative-from-end resolution (`-1` = last latent frame).
/// * otherwise a single `Reference` image → one pin at latent frame 0 with the strength
///   [`i2v_reference`] resolves.
/// * no image conditioning → no pins, i.e. pure-noise T2V.
///
/// Pure and total (no GPU, no weights), because the whole defect this closes was a strength that
/// was carried as far as the provider and then dropped on the way into the mask — which is exactly
/// the hop a unit test can only pin if it is expressible without a render.
pub(super) fn resolve_ti2v_pins(
    req: &GenerationRequest,
    t_lat: usize,
) -> Result<Vec<(usize, f32)>> {
    let keyframes = req.keyframes();
    if keyframes.is_empty() {
        return Ok(i2v_reference(req)
            .map(|(_, strength)| vec![(0usize, strength)])
            .unwrap_or_default());
    }
    let mut pins = Vec::with_capacity(keyframes.len());
    for kf in &keyframes {
        let idx = if kf.frame_idx < 0 {
            t_lat as i32 + kf.frame_idx
        } else {
            kf.frame_idx
        };
        if idx < 0 || idx as usize >= t_lat {
            return Err(Error::Msg(format!(
                "wan2_2_ti2v_5b: keyframe latent frame index {} out of bounds for {t_lat} latent \
                 frames",
                kf.frame_idx
            )));
        }
        pins.push((idx as usize, kf.strength));
    }
    Ok(pins)
}

/// Range-gate every conditioning strength a Wan request can carry (sc-19571, mirroring candle's
/// `check_strength` and mlx-gen-ltx's F-054 gate).
///
/// `strength > 1` yields a **negative** denoise mask (`1 − strength`) → negative per-token timesteps
/// and extrapolating blends: silent garbage rather than a louder pin. `strength < 0` over-weights
/// the noise the same way. Now that the number actually reaches the mask, the range has to be a
/// request-boundary rejection instead of a value nobody read.
pub(super) fn reject_out_of_range_strengths(id: &str, req: &GenerationRequest) -> Result<()> {
    let check = |label: &str, strength: f32| -> Result<()> {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(Error::Msg(format!(
                "{id}: {label} strength must be finite and in [0, 1] (got {strength})"
            )));
        }
        Ok(())
    };
    if let Some(strength) = req.strength {
        check("img2img", strength)?;
    }
    for c in &req.conditioning {
        match c {
            Conditioning::Reference {
                strength: Some(strength),
                ..
            } => check("Reference", *strength)?,
            Conditioning::Keyframe { strength, .. } => check("Keyframe", *strength)?,
            _ => {}
        }
    }
    Ok(())
}

/// Resolve the output `(width, height)` for a **dense** Wan path (5B TI2V, A14B): round the requested
/// dims down to the `patch · vae_stride` grid. Byte-identical block shared by the two dense
/// `generate_impl` bodies (F-010); the VACE paths align the same way inline and don't use this.
///
/// **sc-12308:** this used to also refit an over-`max_area` request via `best_output_size`, silently
/// handing back a geometry the caller never asked for. The cap is now enforced in `validate_impl` by
/// rejecting instead, so by the time this runs the area is already within budget — and since the
/// alignment here only ever rounds *down*, it cannot push a validated request back over.
///
/// **sc-12607:** `validate_impl` now *also* rejects an off-grid `width`/`height` (via
/// [`reject_off_grid`], matching candle), so a validated request is already grid-aligned and this
/// rounding is a defensive identity — it never silently snaps a geometry the caller chose.
pub(super) fn resolve_capped_dims(
    req: &GenerationRequest,
    cfg: &WanModelConfig,
    vae: VaeTiling,
) -> (u32, u32) {
    let stride = provider_vae_stride(vae);
    let width = align_dim(req.width, cfg.patch_size.2, stride.2);
    let height = align_dim(req.height, cfg.patch_size.1, stride.1);
    (width, height)
}

/// The model's `(dw, dh)` pixel grid = `patch · vae_stride` — the lattice every Wan geometry sits on.
pub(super) fn grid(cfg: &WanModelConfig, vae: VaeTiling) -> (u32, u32) {
    let stride = provider_vae_stride(vae);
    (
        (cfg.patch_size.2 * stride.2) as u32,
        (cfg.patch_size.1 * stride.1) as u32,
    )
}
