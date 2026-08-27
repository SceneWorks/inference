//! LTX-2.3 video conditioning primitives.
//!
//! There are two native mechanisms. Image I2V / FLF / keyframes replace latent grid frames and carry
//! a clean latent plus a `1 - strength` denoise mask. Extend/bridge/replace-person clips are VAE
//! encoded and appended as in-context tokens with offset RoPE positions. Both mechanisms feed
//! per-token timesteps (`sigma * mask`) and blend the denoised prediction back toward the clean
//! conditioning every step.

use candle_gen::candle_core::{Device, Error, Result, Tensor};
use candle_gen::gen_core::{imageops, Image};

use crate::config::{SPATIAL_SCALE, TEMPORAL_SCALE};

const REPLACE_NEUTRAL: u32 = 118;

/// Materialize LTX replace-person's ordered 1–4 character-reference carrier as one
/// target-sized contact sheet. LTX's IC-LoRA accepts one image latent at frame zero;
/// treating a `MultiReference` as its first image would silently discard identities.
///
/// The grid is deliberately part of the provider contract: one reference occupies the
/// whole canvas, two occupy left-to-right halves, and three/four occupy row-major
/// quadrants. Every source is resized with the shared PIL-compatible bicubic helper,
/// so Candle and MLX hand the same RGB8 composite to their VAE encoders.
pub fn compose_ordered_character_references(
    images: &[Image],
    target_width: u32,
    target_height: u32,
) -> Result<Image> {
    if !(1..=4).contains(&images.len()) {
        return Err(Error::Msg(format!(
            "ltx: replace-person requires 1–4 ordered character references (got {})",
            images.len()
        )));
    }
    let (width, height) = (target_width as usize, target_height as usize);
    let expected = imageops::checked_image_buffer_len(width, height, 3)
        .ok_or_else(|| Error::Msg("ltx: replace-person composite dimensions overflow".into()))?;
    if width == 0 || height == 0 {
        return Err(Error::Msg(
            "ltx: replace-person composite dimensions must be non-zero".into(),
        ));
    }
    let (columns, rows) = match images.len() {
        1 => (1, 1),
        2 => (2, 1),
        3 | 4 => (2, 2),
        _ => unreachable!("the cardinality check above admits only 1–4"),
    };
    let mut pixels = vec![0_u8; expected];
    for (ordinal, image) in images.iter().enumerate() {
        let (input_width, input_height) = (image.width as usize, image.height as usize);
        let input_len = imageops::checked_image_buffer_len(input_width, input_height, 3)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "ltx: replace-person reference {ordinal} dimensions overflow"
                ))
            })?;
        if input_width == 0 || input_height == 0 || image.pixels.len() != input_len {
            return Err(Error::Msg(format!(
                "ltx: replace-person reference {ordinal} must be a non-empty RGB8 image"
            )));
        }
        let column = ordinal % columns;
        let row = ordinal / columns;
        let x0 = column * width / columns;
        let x1 = (column + 1) * width / columns;
        let y0 = row * height / rows;
        let y1 = (row + 1) * height / rows;
        let tile_width = x1 - x0;
        let tile_height = y1 - y0;
        let tile = imageops::resize_bicubic_u8(
            &image.pixels,
            input_height,
            input_width,
            tile_height,
            tile_width,
        )
        .map_err(|error| Error::Msg(error.to_string()))?;
        for y in 0..tile_height {
            let dst = ((y0 + y) * width + x0) * 3;
            let src = y * tile_width * 3;
            for x in 0..tile_width * 3 {
                pixels[dst + x] = tile[src + x].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(Image {
        width: target_width,
        height: target_height,
        pixels,
    })
}

/// Convert the request contract's latent-frame index into the output-frame coordinate consumed by
/// `VideoConditionByKeyframeIndex` RoPE positions.
pub fn latent_frame_to_output_offset(frame_idx: usize) -> Result<i32> {
    let offset = frame_idx
        .checked_mul(TEMPORAL_SCALE)
        .ok_or_else(|| Error::Msg("ltx: conditioning frame offset overflow".into()))?;
    i32::try_from(offset)
        .map_err(|_| Error::Msg("ltx: conditioning frame offset exceeds i32".into()))
}

#[derive(Clone)]
pub struct I2vConditioning {
    pub latent: Tensor,
    pub clean_latent: Tensor,
    pub denoise_mask: Tensor,
}

pub struct Keyframe<'a> {
    pub latent: &'a Tensor,
    pub frame_idx: usize,
    pub strength: f32,
}

/// Apply replace-latent keyframes to a base `[B,C,F,H,W]`. Later entries win on overlap, matching the
/// MLX/reference implementation.
pub fn apply_keyframes(base: &Tensor, keyframes: &[Keyframe<'_>]) -> Result<I2vConditioning> {
    let (b, c, f, h, w) = base.dims5()?;
    let mut owner: Vec<Option<(usize, usize)>> = vec![None; f];
    for (ki, kf) in keyframes.iter().enumerate() {
        if !(0.0..=1.0).contains(&kf.strength) || !kf.strength.is_finite() {
            return Err(Error::Msg(format!(
                "ltx: keyframe strength must be finite and in [0,1] (got {})",
                kf.strength
            )));
        }
        let (kb, kc, kf_frames, kh, kw) = kf.latent.dims5()?;
        if (kb, kc, kh, kw) != (b, c, h, w) {
            return Err(Error::Msg(format!(
                "ltx: keyframe {ki} latent shape {:?} is incompatible with base {:?}",
                kf.latent.dims(),
                base.dims()
            )));
        }
        if kf.frame_idx >= f {
            return Err(Error::Msg(format!(
                "ltx: keyframe {ki} index {} is out of bounds for {f} latent frames",
                kf.frame_idx
            )));
        }
        for sub in 0..kf_frames.min(f - kf.frame_idx) {
            owner[kf.frame_idx + sub] = Some((ki, sub));
        }
    }

    let zero = Tensor::zeros((b, c, 1, h, w), base.dtype(), base.device())?;
    let mut latent_frames = Vec::with_capacity(f);
    let mut clean_frames = Vec::with_capacity(f);
    let mut mask_frames = Vec::with_capacity(f);
    for (idx, owned) in owner.into_iter().enumerate() {
        match owned {
            Some((ki, sub)) => {
                let cond = keyframes[ki].latent.narrow(2, sub, 1)?;
                latent_frames.push(cond.clone());
                clean_frames.push(cond);
                mask_frames.push(Tensor::full(
                    1.0 - keyframes[ki].strength,
                    (b, 1, 1, 1, 1),
                    base.device(),
                )?);
            }
            None => {
                latent_frames.push(base.narrow(2, idx, 1)?);
                clean_frames.push(zero.clone());
                mask_frames.push(Tensor::ones((b, 1, 1, 1, 1), base.dtype(), base.device())?);
            }
        }
    }
    let latent_refs = latent_frames.iter().collect::<Vec<_>>();
    let clean_refs = clean_frames.iter().collect::<Vec<_>>();
    let mask_refs = mask_frames.iter().collect::<Vec<_>>();
    Ok(I2vConditioning {
        latent: Tensor::cat(&latent_refs, 2)?,
        clean_latent: Tensor::cat(&clean_refs, 2)?,
        denoise_mask: Tensor::cat(&mask_refs, 2)?.to_dtype(base.dtype())?,
    })
}

impl I2vConditioning {
    /// Stage-entry noiser: `noise*(mask*scale) + latent*(1-mask*scale)`.
    pub fn noised(&self, noise: &Tensor, noise_scale: f32) -> Result<Self> {
        let mask = self
            .denoise_mask
            .broadcast_as(self.latent.shape())?
            .to_dtype(self.latent.dtype())?;
        let scaled = (&mask * noise_scale as f64)?;
        let one_minus = (Tensor::ones_like(&scaled)? - &scaled)?;
        let latent = ((noise * &scaled)? + (&self.latent * &one_minus)?)?;
        Ok(Self {
            latent,
            clean_latent: self.clean_latent.clone(),
            denoise_mask: self.denoise_mask.clone(),
        })
    }

    /// `[B,S]` per-token timesteps in the same F/H/W token order as `flatten_latent`.
    pub fn token_timesteps(&self, sigma: f32, h: usize, w: usize) -> Result<Tensor> {
        let (b, _one, f, _mh, _mw) = self.denoise_mask.dims5()?;
        let mask = self
            .denoise_mask
            .broadcast_as((b, 1, f, h, w))?
            .reshape((b, f * h * w))?;
        mask * sigma as f64
    }
}

#[derive(Clone)]
pub struct VideoTokenState {
    pub latent: Tensor,
    pub clean_latent: Tensor,
    pub denoise_mask: Tensor,
    pub positions: Tensor,
    pub target_tokens: usize,
    /// `(B, S, 1)` generated-keyframe marker (`> 0` = a slot token that receives the learned
    /// keyframe absolute-position embedding, sc-18758/sc-18789). `None` until
    /// [`append_generated_keyframe_slots`] marks a run; every other append keeps tokens unmarked,
    /// mirroring the reference `extend_keyframes_mask(marked=False)`.
    pub keyframes_mask: Option<Tensor>,
    /// Where the single contiguous run of generated-keyframe slot tokens sits (`None` = no slots).
    pub generated_keyframe_layout: Option<candle_gen::gen_core::ltx_dfr::GeneratedKeyframeLayout>,
}

impl VideoTokenState {
    pub fn base(noise_grid: &Tensor, positions: &Tensor) -> Result<Self> {
        let latent = crate::pipeline::flatten_latent(noise_grid)?;
        let (b, s, c) = latent.dims3()?;
        Ok(Self {
            clean_latent: Tensor::zeros((b, s, c), latent.dtype(), latent.device())?,
            denoise_mask: Tensor::ones((b, s, 1), latent.dtype(), latent.device())?,
            positions: positions.clone(),
            target_tokens: s,
            latent,
            keyframes_mask: None,
            generated_keyframe_layout: None,
        })
    }

    pub fn from_i2v(state: &I2vConditioning, positions: &Tensor) -> Result<Self> {
        let latent = crate::pipeline::flatten_latent(&state.latent)?;
        let clean_latent = crate::pipeline::flatten_latent(&state.clean_latent)?;
        let (b, s, _c) = latent.dims3()?;
        let (_mb, _one, f, _mh, _mw) = state.denoise_mask.dims5()?;
        let spatial = s / f;
        let denoise_mask = state
            .denoise_mask
            .broadcast_as((b, 1, f, 1, spatial))?
            .reshape((b, s, 1))?;
        Ok(Self {
            latent,
            clean_latent,
            denoise_mask,
            positions: positions.clone(),
            target_tokens: s,
            keyframes_mask: None,
            generated_keyframe_layout: None,
        })
    }

    pub fn token_timesteps(&self, sigma: f32) -> Result<Tensor> {
        let (b, s, _one) = self.denoise_mask.dims3()?;
        self.denoise_mask.reshape((b, s))? * sigma as f64
    }
}

/// RoPE positions for an appended clip. `frame_offset` is in output-frame units, not latent units;
/// the causal first-frame fix applies only at offset zero, matching the upstream keyframe op.
pub fn keyframe_append_positions(
    frames: usize,
    height: usize,
    width: usize,
    frame_offset: i32,
    fps: f32,
    device: &Device,
) -> Result<Tensor> {
    let hw = height * width;
    let n = frames * hw;
    let ts = TEMPORAL_SCALE as i64;
    let ss = SPATIAL_SCALE as i64;
    let mut data = vec![0f32; 3 * n * 2];
    for p in 0..n {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let y = (rem / width) as i64;
        let x = (rem % width) as i64;
        for endpoint in 0..2i64 {
            let mut frame = (t + endpoint) * ts;
            if frame_offset == 0 {
                frame = (frame + 1 - ts).max(0);
            }
            frame += frame_offset as i64;
            let at = p * 2 + endpoint as usize;
            data[at] = frame as f32 / fps;
            data[n * 2 + at] = ((y + endpoint) * ss) as f32;
            data[2 * n * 2 + at] = ((x + endpoint) * ss) as f32;
        }
    }
    Tensor::from_vec(data, (1, 3, n, 2), device)
}

pub fn append_keyframe_clip(
    state: &VideoTokenState,
    clip_latent: &Tensor,
    frame_offset: i32,
    strength: f32,
    fps: f32,
) -> Result<VideoTokenState> {
    if !(0.0..=1.0).contains(&strength) || !strength.is_finite() {
        return Err(Error::Msg(format!(
            "ltx: clip strength must be finite and in [0,1] (got {strength})"
        )));
    }
    let (b, _c, cf, h, w) = clip_latent.dims5()?;
    let tokens = crate::pipeline::flatten_latent(&clip_latent.to_dtype(state.latent.dtype())?)?;
    let n = tokens.dim(1)?;
    let mask = Tensor::full(1.0 - strength, (b, n, 1), state.latent.device())?
        .to_dtype(state.latent.dtype())?;
    let mut pos = keyframe_append_positions(cf, h, w, frame_offset, fps, state.latent.device())?;
    if b > 1 {
        pos = pos.broadcast_as((b, 3, n, 2))?;
    }
    Ok(VideoTokenState {
        latent: Tensor::cat(&[&state.latent, &tokens], 1)?,
        clean_latent: Tensor::cat(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: Tensor::cat(&[&state.denoise_mask, &mask], 1)?,
        positions: Tensor::cat(&[&state.positions, &pos], 2)?,
        target_tokens: state.target_tokens,
        // In-context clip tokens are ordinary conditioning, never keyframe slots.
        keyframes_mask: extend_keyframes_mask(state, n, false)?,
        generated_keyframe_layout: state.generated_keyframe_layout.clone(),
    })
}

// ===================================================================================================
// DFR generated-keyframe slots + single-frame keyframe / reference-latent conditioning (sc-18789).
// Twin of mlx-gen-ltx/src/conditioning.rs — keep the two in step.
// ===================================================================================================

use candle_gen::gen_core::ltx_dfr::GeneratedKeyframeLayout;

/// Extend the state's keyframe marker for `num_new` appended tokens (reference
/// `mask_utils.extend_keyframes_mask`): `marked = false` keeps an absent mask absent; any present
/// mask — or a `marked = true` append — materializes zeros for existing tokens plus the new run.
pub fn extend_keyframes_mask(
    state: &VideoTokenState,
    num_new: usize,
    marked: bool,
) -> Result<Option<Tensor>> {
    let dt = state.latent.dtype();
    let dev = state.latent.device();
    let b = state.latent.dim(0)?;
    let existing = match &state.keyframes_mask {
        Some(mask) => mask.clone(),
        None if !marked => return Ok(None),
        None => Tensor::zeros((b, state.latent.dim(1)?, 1), dt, dev)?,
    };
    let fill = if marked { 1.0f64 } else { 0.0 };
    let new = Tensor::full(fill, (b, num_new, 1), dev)?.to_dtype(dt)?;
    Ok(Some(Tensor::cat(&[&existing, &new], 1)?))
}

/// RoPE positions for one **single-pixel-frame** appended token block (a generated-keyframe slot,
/// or given single-frame keyframe content): full spatial grid at the target latent resolution, the
/// frame axis spanning exactly `[pixel_frame, pixel_frame + 1) / fps` — reference
/// `VideoGeneratedKeyframeSlots._slot_positions` / `VideoConditionByKeyframeIndex`
/// (`num_pixel_frames == 1` narrowing). Output `(1, 3, h·w, 2)` f32.
pub fn single_frame_positions(
    h: usize,
    w: usize,
    pixel_frame: i64,
    spatial_scale: i64,
    fps: f32,
    device: &Device,
) -> Result<Tensor> {
    let n = h * w;
    let mut data = vec![0f32; 3 * n * 2];
    for p in 0..n {
        let y = (p / w) as i64;
        let x = (p % w) as i64;
        for endpoint in 0..2i64 {
            let at = p * 2 + endpoint as usize;
            data[at] = (pixel_frame + endpoint) as f32 / fps;
            data[n * 2 + at] = ((y + endpoint) * spatial_scale) as f32;
            data[2 * n * 2 + at] = ((x + endpoint) * spatial_scale) as f32;
        }
    }
    Tensor::from_vec(data, (1, 3, n, 2), device)
}

fn broadcast_positions(pos: Tensor, b: usize, n: usize) -> Result<Tensor> {
    if b > 1 {
        pos.broadcast_as((b, 3, n, 2))
    } else {
        Ok(pos)
    }
}

/// Temporal slice `x[:, :, i..i+1]` on a `(B, C, K, H, W)` latent.
fn frame_slice(x: &Tensor, i: usize) -> Result<Tensor> {
    x.narrow(2, i, 1)
}

/// Append **generated keyframe slots** — the DFR conditioning item
/// (`VideoGeneratedKeyframeSlots.apply_to`). Each slot occupies one latent frame's worth of tokens
/// at the target's spatial resolution, `denoise_mask = 1` (the stage noiser fills it from the slot
/// `latent` — zeros or `initial_keyframes` — and the loop generates its content), marked in
/// `keyframes_mask` so it receives the learned keyframe embedding, recorded as the state's single
/// [`GeneratedKeyframeLayout`].
#[allow(clippy::too_many_arguments)]
pub fn append_generated_keyframe_slots(
    state: &VideoTokenState,
    pixel_frame_indices: &[i64],
    initial_keyframes: Option<&Tensor>,
    num_pixel_frames: i64,
    h: usize,
    w: usize,
    spatial_scale: i64,
    fps: f32,
) -> Result<VideoTokenState> {
    if state.generated_keyframe_layout.is_some() {
        return Err(Error::Msg(
            "ltx dfr: generated keyframe slots were already applied to this state; append all \
             slots through a single call"
                .into(),
        ));
    }
    if pixel_frame_indices.is_empty() {
        return Err(Error::Msg(
            "ltx dfr: pixel_frame_indices must be non-empty".into(),
        ));
    }
    if pixel_frame_indices.windows(2).any(|p| p[1] <= p[0]) || pixel_frame_indices[0] < 0 {
        return Err(Error::Msg(format!(
            "ltx dfr: pixel_frame_indices must be non-negative and strictly increasing, got \
             {pixel_frame_indices:?}"
        )));
    }
    let last = *pixel_frame_indices.last().expect("non-empty");
    if last >= num_pixel_frames {
        return Err(Error::Msg(format!(
            "ltx dfr: generated keyframe at pixel frame {last} is outside the target's \
             {num_pixel_frames} frames"
        )));
    }

    let dt = state.latent.dtype();
    let dev = state.latent.device();
    let (b, _s, c) = state.latent.dims3()?;
    let k = pixel_frame_indices.len();
    let tokens_per_keyframe = h * w;
    let num_new = tokens_per_keyframe * k;

    let slot_tokens = match initial_keyframes {
        None => Tensor::zeros((b, num_new, c), dt, dev)?,
        Some(init) => {
            let (ib, _ic, ik, ih, iw) = init.dims5()?;
            if ik != k {
                return Err(Error::Msg(format!(
                    "ltx dfr: initial_keyframes must carry K={k} frames, got {ik}"
                )));
            }
            if ib != b {
                return Err(Error::Msg(format!(
                    "ltx dfr: initial_keyframes batch {ib} does not match latent batch {b}"
                )));
            }
            if (ih, iw) != (h, w) {
                return Err(Error::Msg(format!(
                    "ltx dfr: initial_keyframes spatial size ({ih}, {iw}) does not match target \
                     latent spatial size ({h}, {w})"
                )));
            }
            let mut blocks = Vec::with_capacity(k);
            for index in 0..k {
                blocks.push(crate::pipeline::flatten_latent(
                    &frame_slice(init, index)?.to_dtype(dt)?,
                )?);
            }
            Tensor::cat(&blocks.iter().collect::<Vec<_>>(), 1)?
        }
    };

    let mut position_blocks = Vec::with_capacity(k);
    for &pixel_frame in pixel_frame_indices {
        position_blocks.push(single_frame_positions(
            h,
            w,
            pixel_frame,
            spatial_scale,
            fps,
            dev,
        )?);
    }
    let positions = Tensor::cat(&position_blocks.iter().collect::<Vec<_>>(), 2)?;
    let positions = broadcast_positions(positions, b, num_new)?;

    let denoise_mask = Tensor::ones((b, num_new, 1), dt, dev)?;
    let keyframes_mask = extend_keyframes_mask(state, num_new, true)?;
    let first_token = state.latent.dim(1)?;

    Ok(VideoTokenState {
        latent: Tensor::cat(&[&state.latent, &slot_tokens], 1)?,
        clean_latent: Tensor::cat(
            &[
                &state.clean_latent,
                &Tensor::zeros((b, num_new, c), dt, dev)?,
            ],
            1,
        )?,
        denoise_mask: Tensor::cat(&[&state.denoise_mask, &denoise_mask], 1)?,
        positions: Tensor::cat(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
        keyframes_mask,
        generated_keyframe_layout: Some(GeneratedKeyframeLayout {
            pixel_frame_indices: pixel_frame_indices.to_vec(),
            tokens_per_keyframe,
            first_token,
        }),
    })
}

/// Append **given** single-pixel-frame keyframe content as clean-latent guidance at explicit pixel
/// frames (`VideoConditionByKeyframeIndex` with `num_pixel_frames = 1` — the DFR anchor-keyframe
/// carry). Deliberately **unmarked** in `keyframes_mask`: the reference marks only generated slots.
pub fn append_single_frame_keyframes(
    state: &VideoTokenState,
    keyframes: &Tensor,
    pixel_frame_indices: &[i64],
    strength: f32,
    spatial_scale: i64,
    fps: f32,
) -> Result<VideoTokenState> {
    let (b, _c, k, h, w) = keyframes.dims5()?;
    if k != pixel_frame_indices.len() {
        return Err(Error::Msg(format!(
            "ltx dfr: expected {} keyframe latents, got K={k}",
            pixel_frame_indices.len()
        )));
    }
    if pixel_frame_indices.iter().any(|&p| p < 0) {
        // Position 0 IS legal — the non-first-tile seam anchor at local frame 0; the reference's
        // causal-fixed span at frame_idx 0 equals `single_frame_positions`' `[0, 1)/fps` (see the
        // mlx twin's note).
        return Err(Error::Msg(format!(
            "ltx dfr: single-frame keyframe positions must be >= 0, got {pixel_frame_indices:?}"
        )));
    }
    let dt = state.latent.dtype();
    let dev = state.latent.device();
    let mut token_blocks = Vec::with_capacity(k);
    let mut position_blocks = Vec::with_capacity(k);
    for (index, &pixel_frame) in pixel_frame_indices.iter().enumerate() {
        token_blocks.push(crate::pipeline::flatten_latent(
            &frame_slice(keyframes, index)?.to_dtype(dt)?,
        )?);
        position_blocks.push(single_frame_positions(
            h,
            w,
            pixel_frame,
            spatial_scale,
            fps,
            dev,
        )?);
    }
    let tokens = Tensor::cat(&token_blocks.iter().collect::<Vec<_>>(), 1)?;
    let positions = Tensor::cat(&position_blocks.iter().collect::<Vec<_>>(), 2)?;
    let n = tokens.dim(1)?;
    let positions = broadcast_positions(positions, b, n)?;
    let mask = Tensor::full(f64::from(1.0 - strength), (b, n, 1), dev)?.to_dtype(dt)?;

    Ok(VideoTokenState {
        latent: Tensor::cat(&[&state.latent, &tokens.zeros_like()?], 1)?,
        clean_latent: Tensor::cat(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: Tensor::cat(&[&state.denoise_mask, &mask], 1)?,
        positions: Tensor::cat(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
        keyframes_mask: extend_keyframes_mask(state, n, false)?,
        generated_keyframe_layout: state.generated_keyframe_layout.clone(),
    })
}

/// Append a **reference video latent** for IC-LoRA detailing (`VideoConditionByReferenceLatent`,
/// the `temporal_scale_factor = 1` configuration the DFR stage-2 detailing pass uses): clean
/// in-context tokens whose spatial positions are scaled by `downscale_factor` into the target's
/// coordinate frame. Never marked as keyframes.
pub fn append_reference_latent(
    state: &VideoTokenState,
    reference: &Tensor,
    downscale_factor: i64,
    strength: f32,
    fps: f32,
) -> Result<VideoTokenState> {
    if downscale_factor < 1 {
        return Err(Error::Msg(format!(
            "ltx dfr: reference downscale_factor must be >= 1, got {downscale_factor}"
        )));
    }
    let (b, _c, cf, h, w) = reference.dims5()?;
    let dt = state.latent.dtype();
    let dev = state.latent.device();
    let tokens = crate::pipeline::flatten_latent(&reference.to_dtype(dt)?)?;
    let n = tokens.dim(1)?;
    // The standard causal appended-token grid at the reference's own dims, with the spatial axes
    // on the x(SPATIAL_SCALE·d) grid ((hh+e)·32·d ≡ upstream's positions[:,1:] · d).
    let ts = TEMPORAL_SCALE as i64;
    let ss = SPATIAL_SCALE as i64 * downscale_factor;
    let hw = h * w;
    let total = cf * hw;
    let mut data = vec![0f32; 3 * total * 2];
    for p in 0..total {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let y = (rem / w) as i64;
        let x = (rem % w) as i64;
        for endpoint in 0..2i64 {
            let frame = ((t + endpoint) * ts + 1 - ts).max(0);
            let at = p * 2 + endpoint as usize;
            data[at] = frame as f32 / fps;
            data[total * 2 + at] = ((y + endpoint) * ss) as f32;
            data[2 * total * 2 + at] = ((x + endpoint) * ss) as f32;
        }
    }
    let positions = Tensor::from_vec(data, (1, 3, total, 2), dev)?;
    let positions = broadcast_positions(positions, b, n)?;
    let mask = Tensor::full(f64::from(1.0 - strength), (b, n, 1), dev)?.to_dtype(dt)?;

    Ok(VideoTokenState {
        latent: Tensor::cat(&[&state.latent, &tokens.zeros_like()?], 1)?,
        clean_latent: Tensor::cat(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: Tensor::cat(&[&state.denoise_mask, &mask], 1)?,
        positions: Tensor::cat(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
        keyframes_mask: extend_keyframes_mask(state, n, false)?,
        generated_keyframe_layout: state.generated_keyframe_layout.clone(),
    })
}

/// Read the denoised generated-keyframe slots back out of a post-denoise state as a
/// `(B, C, K, H, W)` latent, via the recorded [`GeneratedKeyframeLayout`].
pub fn take_generated_keyframes(state: &VideoTokenState, h: usize, w: usize) -> Result<Tensor> {
    let layout = state.generated_keyframe_layout.as_ref().ok_or_else(|| {
        Error::Msg(
            "ltx dfr: this state carries no generated-keyframe layout; slots were never appended"
                .into(),
        )
    })?;
    if layout.tokens_per_keyframe != h * w {
        return Err(Error::Msg(format!(
            "ltx dfr: layout tokens_per_keyframe {} != h·w {}",
            layout.tokens_per_keyframe,
            h * w
        )));
    }
    let mut frames = Vec::with_capacity(layout.pixel_frame_indices.len());
    for k in 0..layout.pixel_frame_indices.len() {
        let start = layout.first_token + k * layout.tokens_per_keyframe;
        let tokens = state.latent.narrow(1, start, h * w)?;
        frames.push(crate::pipeline::unflatten_latent(&tokens, 1, h, w)?);
    }
    Tensor::cat(&frames.iter().collect::<Vec<_>>(), 2)
}

/// `denoised*mask + clean*(1-mask)` with broadcasting over channel/features.
pub fn apply_denoise_mask(denoised: &Tensor, clean: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask = mask
        .broadcast_as(denoised.shape())?
        .to_dtype(denoised.dtype())?;
    let inv = (Tensor::ones_like(&mask)? - &mask)?;
    (denoised * &mask)? + (clean * inv)?
}

pub fn preprocess_conditioning_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    let expected = imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX);
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "ltx: conditioning image buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized = if (iw, ih) == (tw, th) {
        image.pixels.iter().map(|&v| v as f32).collect()
    } else {
        imageops::resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
            .map_err(|e| Error::Msg(e.to_string()))?
    };
    let normalized: Vec<f32> = resized
        .into_iter()
        .map(|v| 2.0 * (v / 255.0) - 1.0)
        .collect();
    Tensor::from_vec(normalized, (1, th, tw, 3), device)?
        .permute((0, 3, 1, 2))?
        .unsqueeze(2)
}

pub fn preprocess_conditioning_clip(
    frames: &[Image],
    target_width: u32,
    target_height: u32,
    device: &Device,
) -> Result<Tensor> {
    if frames.is_empty() {
        return Err(Error::Msg(
            "ltx: conditioning clip must not be empty".into(),
        ));
    }
    let tensors = frames
        .iter()
        .map(|frame| preprocess_conditioning_image(frame, target_width, target_height, device))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&tensors.iter().collect::<Vec<_>>(), 2)
}

/// Neutralize the masked region of a replace-person control frame exactly like the MLX lane.
pub fn apply_replacement_mask(frame: &Image, mask: &Image, strength: f32) -> Result<Image> {
    if (frame.width, frame.height) != (mask.width, mask.height) {
        return Err(Error::Msg(format!(
            "ltx: replace-person mask {}x{} must match frame {}x{}",
            mask.width, mask.height, frame.width, frame.height
        )));
    }
    let count = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .ok_or_else(|| Error::Msg("ltx: replace-person dimensions overflow".into()))?;
    let expected = count
        .checked_mul(3)
        .ok_or_else(|| Error::Msg("ltx: replace-person RGB buffer size overflows".into()))?;
    if frame.pixels.len() != expected || mask.pixels.len() != expected {
        return Err(Error::Msg(
            "ltx: replace-person frame and mask must be RGB8".into(),
        ));
    }
    let strength = strength.clamp(0.0, 1.0);
    let mut pixels = vec![0u8; expected];
    for pixel in 0..count {
        let r = mask.pixels[pixel * 3] as u32;
        let g = mask.pixels[pixel * 3 + 1] as u32;
        let b = mask.pixels[pixel * 3 + 2] as u32;
        let luma = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
        let gate = ((luma as f32 * strength) as u32).min(255);
        for channel in 0..3 {
            let source = frame.pixels[pixel * 3 + channel] as u32;
            pixels[pixel * 3 + channel] =
                ((REPLACE_NEUTRAL * gate + source * (255 - gate) + 127) / 255) as u8;
        }
    }
    Ok(Image {
        width: frame.width,
        height: frame.height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframes_drive_per_token_timestep_and_blend() -> Result<()> {
        let dev = Device::Cpu;
        let base = Tensor::from_vec(vec![10f32, 20., 30.], (1, 1, 3, 1, 1), &dev)?;
        let pin = Tensor::from_vec(vec![99f32], (1, 1, 1, 1, 1), &dev)?;
        let state = apply_keyframes(
            &base,
            &[Keyframe {
                latent: &pin,
                frame_idx: 1,
                strength: 1.0,
            }],
        )?;
        assert_eq!(
            state.latent.flatten_all()?.to_vec1::<f32>()?,
            [10., 99., 30.]
        );
        assert_eq!(
            state.token_timesteps(0.9, 1, 1)?.to_vec2::<f32>()?,
            [vec![0.9, 0.0, 0.9]]
        );
        let denoised = Tensor::full(7f32, (1, 1, 3, 1, 1), &dev)?;
        let blended = apply_denoise_mask(&denoised, &state.clean_latent, &state.denoise_mask)?;
        assert_eq!(blended.flatten_all()?.to_vec1::<f32>()?, [7., 99., 7.]);
        Ok(())
    }

    #[test]
    fn appended_clip_preserves_target_count_and_offsets_positions() -> Result<()> {
        let dev = Device::Cpu;
        let noise = Tensor::from_vec(vec![3f32, 4.], (1, 2, 1, 1, 1), &dev)?;
        let pos = crate::rope::create_position_grid(1, 1, 1, 24.0, &dev)?;
        let base = VideoTokenState::base(&noise, &pos)?;
        let clip = Tensor::from_vec(vec![7f32, 9.], (1, 2, 1, 1, 1), &dev)?;
        let appended = append_keyframe_clip(&base, &clip, 3, 1.0, 24.0)?;
        assert_eq!(appended.target_tokens, 1);
        assert_eq!(appended.latent.dims(), &[1, 2, 2]);
        assert_eq!(
            appended.denoise_mask.flatten_all()?.to_vec1::<f32>()?,
            [1., 0.]
        );
        let p = appended
            .positions
            .narrow(2, 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!((p[0] - 3.0 / 24.0).abs() < 1e-7);
        assert!((p[1] - 11.0 / 24.0).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn negative_one_bridge_resolves_to_the_target_output_tail() -> Result<()> {
        let dev = Device::Cpu;
        let target = crate::rope::create_position_grid(7, 1, 1, 24.0, &dev)?;
        let target_tail_end = target.narrow(2, 6, 1)?.flatten_all()?.to_vec1::<f32>()?[1];
        let raw_frame_idx = -1i32;
        let latent_frames = 7i32;
        let resolved_latent_idx = (latent_frames + raw_frame_idx) as usize;
        assert_eq!(resolved_latent_idx, 6);
        let offset = latent_frame_to_output_offset(resolved_latent_idx)?;
        assert_eq!(offset, 48);
        let appended = keyframe_append_positions(1, 1, 1, offset, 24.0, &dev)?;
        let append_start = appended.flatten_all()?.to_vec1::<f32>()?[0];
        assert!((append_start - (target_tail_end - 1.0 / 24.0)).abs() < 1e-7);
        assert!((append_start - 48.0 / 24.0).abs() < 1e-7);
        assert!((append_start - 6.0 / 24.0).abs() > 1.0);
        Ok(())
    }

    #[test]
    fn replace_mask_matches_neutralization_contract() -> Result<()> {
        let frame = Image {
            width: 2,
            height: 1,
            pixels: vec![10, 20, 30, 40, 50, 60],
        };
        let mask = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };
        let got = apply_replacement_mask(&frame, &mask, 1.0)?;
        assert_eq!(got.pixels, [118, 118, 118, 40, 50, 60]);
        Ok(())
    }

    #[test]
    fn ordered_character_reference_grid_preserves_all_four_identities() {
        let image = |pixels| Image {
            width: 1,
            height: 1,
            pixels,
        };
        let references = vec![
            image(vec![255, 0, 0]),
            image(vec![0, 255, 0]),
            image(vec![0, 0, 255]),
            image(vec![255, 255, 0]),
        ];
        let composite = compose_ordered_character_references(&references, 4, 4).unwrap();
        let pixel = |x: usize, y: usize| &composite.pixels[(y * 4 + x) * 3..][..3];
        assert_eq!(pixel(0, 0), [255, 0, 0]);
        assert_eq!(pixel(3, 0), [0, 255, 0]);
        assert_eq!(pixel(0, 3), [0, 0, 255]);
        assert_eq!(pixel(3, 3), [255, 255, 0]);
    }

    #[test]
    fn ordered_character_reference_grid_refuses_cardinality_and_bad_rgb() {
        assert!(compose_ordered_character_references(&[], 64, 64).is_err());
        let image = Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        assert!(compose_ordered_character_references(&vec![image.clone(); 5], 64, 64).is_err());
        assert!(compose_ordered_character_references(
            &[Image {
                width: 1,
                height: 1,
                pixels: vec![0, 0],
            }],
            64,
            64,
        )
        .is_err());
    }

    #[test]
    fn ordered_character_reference_grid_uses_each_advertised_geometry() {
        let image = |pixels| Image {
            width: 1,
            height: 1,
            pixels,
        };
        let red = image(vec![255, 0, 0]);
        let green = image(vec![0, 255, 0]);
        let blue = image(vec![0, 0, 255]);
        let pixel = |image: &Image, x: usize, y: usize| {
            let start = (y * 4 + x) * 3;
            [
                image.pixels[start],
                image.pixels[start + 1],
                image.pixels[start + 2],
            ]
        };

        let one = compose_ordered_character_references(std::slice::from_ref(&red), 4, 4).unwrap();
        assert_eq!(pixel(&one, 3, 3), [255, 0, 0], "1 = full canvas");
        let two =
            compose_ordered_character_references(&[red.clone(), green.clone()], 4, 4).unwrap();
        assert_eq!(pixel(&two, 0, 3), [255, 0, 0], "2 = left tile");
        assert_eq!(pixel(&two, 3, 3), [0, 255, 0], "2 = right tile");
        let three = compose_ordered_character_references(&[red, green, blue], 4, 4).unwrap();
        assert_eq!(pixel(&three, 0, 3), [0, 0, 255], "3 = lower-left tile");
        assert_eq!(
            pixel(&three, 3, 3),
            [0, 0, 0],
            "3 leaves only lower-right empty"
        );
    }
}
