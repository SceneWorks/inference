//! I2V latent conditioning — port of the reference `mlx_video/conditioning/latent.py`. Injects a
//! VAE-encoded conditioning image as a **clean latent** at a chosen frame index and drives the denoise
//! loop with a per-frame **denoise mask** so the conditioned frame is preserved while the rest is
//! generated. Used by both stages of the I2V pipeline ([`crate::pipeline`]).
//!
//! The shape convention matches the rest of the VAE/pipeline: latents are **NCFHW**
//! `(B, 128, F, H, W)`; the mask is `(B, 1, F, 1, 1)` (one value per latent frame, broadcast over
//! channels + space). `1.0` = full denoise (generate), `0.0` = keep the clean conditioning. A
//! conditioning at `frame_idx` with `strength s` sets the mask there to `1 − s` (so `s = 1.0` →
//! mask 0 → the frame is fully pinned to the image latent; `s = 0.0` → mask 1 → no effect).
//!
//! Reference `generate.py` / `generate_av.py` wire exactly **one** image at **one** frame (default 0).
//! [`apply_conditioning`] keeps the general per-frame structure (a clean latent of `cond_f ≥ 1`
//! frames at any index) so the parity-plus multi-keyframe / first-last-frame extension is mechanically
//! reachable, but the [`crate::model`] Generator only wires the single-image case (strict parity).
//!
//! Everything is **dtype-preserving** (the `mx.array(1.0, dtype)` pattern from the reference): the
//! conditioning state, the noiser, and the mask all stay in the latent's dtype so the I2V path is
//! bit-exact to the reference at both `f32` and `bf16`.

use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

/// Materialize LTX replace-person's ordered 1–4 character-reference carrier as one
/// target-sized contact sheet. LTX's IC-LoRA accepts one image latent at frame zero;
/// treating a `MultiReference` as its first image would silently discard identities.
///
/// The grid is deliberately part of the provider contract: one reference occupies the
/// whole canvas, two occupy left-to-right halves, and three/four occupy row-major
/// quadrants. Every source is resized with the shared PIL-compatible bicubic helper,
/// so MLX and Candle hand the same RGB8 composite to their VAE encoders.
pub fn compose_ordered_character_references(
    images: &[mlx_gen::Image],
    target_width: u32,
    target_height: u32,
) -> Result<mlx_gen::Image> {
    if !(1..=4).contains(&images.len()) {
        return Err(Error::Msg(format!(
            "ltx_2_3: replace_person requires 1–4 ordered character references (got {})",
            images.len()
        )));
    }
    let (width, height) = (target_width as usize, target_height as usize);
    let expected = mlx_gen::gen_core::imageops::checked_image_buffer_len(width, height, 3)
        .ok_or_else(|| {
            Error::Msg("ltx_2_3: replace-person composite dimensions overflow".into())
        })?;
    if width == 0 || height == 0 {
        return Err(Error::Msg(
            "ltx_2_3: replace-person composite dimensions must be non-zero".into(),
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
        let input_len =
            mlx_gen::gen_core::imageops::checked_image_buffer_len(input_width, input_height, 3)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "ltx_2_3: replace-person reference {ordinal} dimensions overflow"
                    ))
                })?;
        if input_width == 0 || input_height == 0 || image.pixels.len() != input_len {
            return Err(Error::Msg(format!(
                "ltx_2_3: replace-person reference {ordinal} must be a non-empty RGB8 image"
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
        let tile = mlx_gen::gen_core::imageops::resize_bicubic_u8(
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
    Ok(mlx_gen::Image {
        width: target_width,
        height: target_height,
        pixels,
    })
}

/// Convert a request's resolved latent-frame index into the output-frame coordinate consumed by
/// `VideoConditionByKeyframeIndex` RoPE positions.
pub fn latent_frame_to_output_offset(frame_idx: i32, temporal_scale: i64) -> Result<i32> {
    let offset = i64::from(frame_idx)
        .checked_mul(temporal_scale)
        .ok_or_else(|| Error::Msg("ltx: conditioning frame offset overflow".into()))?;
    i32::try_from(offset)
        .map_err(|_| Error::Msg("ltx: conditioning frame offset exceeds i32".into()))
}

/// A scalar in `dt` (the dtype-preserving `mx.array(v, dtype=…)`).
fn scalar(v: f32, dt: Dtype) -> Result<Array> {
    Ok(Array::from_slice(&[v], &[1]).as_dtype(dt)?)
}

/// Temporal slice `x[:, :, i:i+1]` (a single latent frame, axis 2).
fn frame(x: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    Ok(x.take_axis(idx, 2)?)
}

/// Zeros with `tokens`' shape and dtype (`torch.zeros_like` for the appended-placeholder pattern).
fn zeros_like_tokens(tokens: &Array) -> Result<Array> {
    Ok(Array::zeros::<f32>(tokens.shape())?.as_dtype(tokens.dtype())?)
}

/// The I2V conditioning state (reference `LatentState`): the current (noised) latent, the clean
/// conditioning latent, and the per-frame denoise mask. `clean_latent` + `denoise_mask` are fixed
/// across the denoise loop; only `latent` evolves (it seeds the loop).
#[derive(Clone)]
pub struct I2vConditioning {
    /// Current latent `(B, C, F, H, W)` — seeds the denoise loop (already noised by [`Self::noised`]).
    pub latent: Array,
    /// Clean conditioning latent `(B, C, F, H, W)`: the image latent at the conditioned frame(s),
    /// zeros elsewhere. [`crate::pipeline::denoise`] blends toward this where the mask is `< 1`.
    pub clean_latent: Array,
    /// Per-frame denoise mask `(B, 1, F, 1, 1)`: `1 − strength` at the conditioned frame(s), `1`
    /// elsewhere.
    pub denoise_mask: Array,
}

/// One replace-latent keyframe: a clean `(B, C, cond_f, H, W)` latent pinned at output latent frame
/// `frame_idx` with `strength` (mask `1 − strength`). For single-image I2V `cond_f = 1`.
#[derive(Clone, Copy)]
pub struct Keyframe<'a> {
    pub latent: &'a Array,
    pub frame_idx: i32,
    pub strength: f32,
}

/// Build the conditioning state by injecting `cond_latent` (a clean `(B, C, cond_f, H, W)` latent —
/// for single-image I2V `cond_f = 1`) at `frame_idx` over `base_latent` `(B, C, F, H, W)`. The
/// single-keyframe form of [`apply_keyframes`] (strict-parity I2V; reference `apply_conditioning`).
pub fn apply_conditioning(
    base_latent: &Array,
    cond_latent: &Array,
    frame_idx: i32,
    strength: f32,
) -> Result<I2vConditioning> {
    apply_keyframes(
        base_latent,
        &[Keyframe {
            latent: cond_latent,
            frame_idx,
            strength,
        }],
    )
}

/// Build the conditioning state by injecting **multiple** clean keyframe latents at their frame
/// indices over `base_latent` `(B, C, F, H, W)` — the replace-latent mechanism (reference
/// `VideoConditionByLatentIndex` applied per item; **first_last_frame** = two keyframes at `0` and the
/// last latent frame). Mirrors the reference's per-item `apply_to`: each keyframe **overwrites** the
/// `latent` + `clean_latent` and sets the `denoise_mask` to `1 − strength` over its covered frames;
/// uncovered frames keep `base_latent` (latent), `0` (clean), `1` (mask). When two keyframes overlap,
/// the **later** one in the list wins (sequential application, matching torch).
///
/// Because this only rewrites existing grid frames in place (no appended tokens), the resulting state
/// drives the **existing grid** [`crate::pipeline::denoise`] / [`crate::pipeline::denoise_av`] loops
/// unchanged — FLF needs no token-native loop.
pub fn apply_keyframes(base_latent: &Array, keyframes: &[Keyframe]) -> Result<I2vConditioning> {
    let dt = base_latent.dtype();
    let sh = base_latent.shape(); // (B, C, F, H, W)
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);

    let mask_gen = broadcast_to(
        &scalar(1.0, dt)?.reshape(&[1, 1, 1, 1, 1])?,
        &[b, 1, 1, 1, 1],
    )?;
    let zero_frame = broadcast_to(
        &scalar(0.0, dt)?.reshape(&[1, 1, 1, 1, 1])?,
        &[b, c, 1, h, w],
    )?;

    // Per-output-frame assignment: which keyframe (if any) owns this frame, and its source sub-index.
    // Later keyframes override earlier ones (sequential `apply_to`).
    let mut owner: Vec<Option<(usize, i32)>> = vec![None; f as usize];
    for (ki, kf) in keyframes.iter().enumerate() {
        let cs = kf.latent.shape();
        let (cond_c, cond_f, cond_h, cond_w) = (cs[1], cs[2], cs[3], cs[4]);
        if (cond_c, cond_h, cond_w) != (c, h, w) {
            return Err(Error::Msg(format!(
                "keyframe {ki} latent spatial shape ({cond_c},{cond_h},{cond_w}) != target ({c},{h},{w})"
            )));
        }
        if kf.frame_idx < 0 || kf.frame_idx >= f {
            return Err(Error::Msg(format!(
                "keyframe {ki} frame index {} out of bounds for {f} latent frames",
                kf.frame_idx
            )));
        }
        let end_idx = (kf.frame_idx + cond_f).min(f);
        for i in kf.frame_idx..end_idx {
            owner[i as usize] = Some((ki, i - kf.frame_idx));
        }
    }

    let mut latent_frames = Vec::with_capacity(f as usize);
    let mut clean_frames = Vec::with_capacity(f as usize);
    let mut mask_frames = Vec::with_capacity(f as usize);
    for i in 0..f {
        match owner[i as usize] {
            Some((ki, sub)) => {
                let kf = &keyframes[ki];
                let cond = frame(kf.latent, sub)?;
                latent_frames.push(cond.clone());
                clean_frames.push(cond);
                let mask_keep = broadcast_to(
                    &scalar(1.0 - kf.strength, dt)?.reshape(&[1, 1, 1, 1, 1])?,
                    &[b, 1, 1, 1, 1],
                )?;
                mask_frames.push(mask_keep);
            }
            None => {
                latent_frames.push(frame(base_latent, i)?);
                clean_frames.push(zero_frame.clone());
                mask_frames.push(mask_gen.clone());
            }
        }
    }

    let latent = concatenate_axis(&latent_frames.iter().collect::<Vec<_>>(), 2)?;
    let clean_latent = concatenate_axis(&clean_frames.iter().collect::<Vec<_>>(), 2)?;
    let denoise_mask = concatenate_axis(&mask_frames.iter().collect::<Vec<_>>(), 2)?;
    Ok(I2vConditioning {
        latent,
        clean_latent,
        denoise_mask,
    })
}

impl I2vConditioning {
    /// Apply the stage-entry noiser (reference: `noise·(mask·scale) + latent·(1 − mask·scale)`), in
    /// the latent dtype. At conditioned frames (`mask = 1 − strength`, `0` when `strength = 1`) the
    /// clean image latent is preserved; elsewhere (`mask = 1`) this is the plain `noise·scale +
    /// latent·(1 − scale)` re-noise. Returns a new state with `latent` replaced (`clean_latent` +
    /// `denoise_mask` unchanged).
    pub fn noised(&self, noise: &Array, noise_scale: f32) -> Result<Self> {
        let dt = self.latent.dtype();
        let scale = scalar(noise_scale, dt)?;
        let scaled_mask = multiply(&self.denoise_mask, &scale)?; // (B,1,F,1,1)
        let one_minus = subtract(&scalar(1.0, dt)?, &scaled_mask)?;
        let latent = add(
            &multiply(noise, &scaled_mask)?,
            &multiply(&self.latent, &one_minus)?,
        )?;
        Ok(Self {
            latent,
            clean_latent: self.clean_latent.clone(),
            denoise_mask: self.denoise_mask.clone(),
        })
    }

    /// Per-token timesteps `σ·mask` shaped `(B, num_tokens)` for the DiT (reference: conditioned
    /// tokens get timestep `0`, the rest `σ`). The mask `(B,1,F,1,1)` is broadcast to `(B,1,F,H,W)`
    /// then flattened to token order `F·H·W`.
    pub fn token_timesteps(&self, sigma: f32, h: i32, w: i32) -> Result<Array> {
        let dt = self.latent.dtype();
        let ms = self.denoise_mask.shape(); // (B,1,F,1,1)
        let (b, f) = (ms[0], ms[2]);
        let mask_flat =
            broadcast_to(&self.denoise_mask, &[b, 1, f, h, w])?.reshape(&[b, f * h * w])?;
        Ok(multiply(&scalar(sigma, dt)?, &mask_flat)?)
    }
}

// ===================================================================================================
// Keyframe-append (IC-LoRA in-context) conditioning — extend_clip / video_bridge / replace_person.
// ===================================================================================================
//
// Port of `VideoConditionByKeyframeIndex.apply_to`: instead of overwriting grid frames in place (the
// replace-latent path above), the conditioning clip's VAE latents are **appended** as extra in-context
// tokens at the end of the token sequence, with their own RoPE positions (frame axis offset by
// `frame_idx`) and a `1 − strength` denoise mask. The target tokens attend to them; an IC-LoRA adapter
// is what teaches the DiT to use them. This is token-native: it operates on the flat `(B, S, C)` token
// sequence (the LTX DiT forward is fully token+positions driven), so the appended tokens never need to
// form a grid. Used by the stage-1 [`crate::pipeline::denoise_av_tokens`] loop.

/// A token-native video latent state (reference `LatentState` for the video stream): the latent as a
/// flat token sequence `(B, S, C)`, the matching per-token `clean_latent` `(B, S, C)`, `denoise_mask`
/// `(B, S, 1)` (`1` = generate, `1 − strength` at conditioning tokens), and `positions` `(B, 3, S, 2)`.
#[derive(Clone)]
pub struct VideoTokenState {
    pub latent: Array,
    pub clean_latent: Array,
    pub denoise_mask: Array,
    pub positions: Array,
    /// The target token count (the first `target_tokens` tokens are the generated grid; the rest are
    /// appended conditioning). `unpatchify` reads exactly these back into a grid.
    pub target_tokens: i32,
    /// `(B, S, 1)` generated-keyframe marker (`> 0` = a slot token that receives the model's learned
    /// keyframe absolute-position embedding, sc-18758/sc-18789). `None` until
    /// [`append_generated_keyframe_slots`] marks a run — every other append keeps existing tokens
    /// unmarked, mirroring the reference `extend_keyframes_mask(marked=False)`.
    pub keyframes_mask: Option<Array>,
    /// Where the single contiguous run of generated-keyframe slot tokens sits (`None` = no slots).
    /// At most one [`append_generated_keyframe_slots`] per state, like the reference item.
    pub generated_keyframe_layout: Option<mlx_gen::gen_core::ltx_dfr::GeneratedKeyframeLayout>,
}

/// Patchify a latent grid `(B, C, F, H, W)` → tokens `(B, F·H·W, C)` (patch size 1, the reference
/// `VideoLatentPatchifier.patchify`: `b c f h w -> b (f h w) c`).
pub fn patchify_grid(grid: &Array) -> Result<Array> {
    let sh = grid.shape(); // (B, C, F, H, W)
    let (b, c) = (sh[0], sh[1]);
    Ok(grid.reshape(&[b, c, -1])?.transpose_axes(&[0, 2, 1])?)
}

/// Inverse of [`patchify_grid`] for the generated grid: tokens `(B, F·H·W, C)` → `(B, C, F, H, W)`.
pub fn unpatchify_grid(tokens: &Array, c: i32, f: i32, h: i32, w: i32) -> Result<Array> {
    let b = tokens.shape()[0];
    Ok(tokens
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[b, c, f, h, w])?)
}

impl VideoTokenState {
    /// The base (T2V) token state over a noise grid `(B, C, F, H, W)` with its main `positions`
    /// `(B, 3, F·H·W, 2)`: latent = flattened noise, clean = 0, denoise_mask = 1 (all-generate).
    pub fn base(noise_grid: &Array, positions: &Array) -> Result<Self> {
        let dt = noise_grid.dtype();
        let latent = patchify_grid(noise_grid)?;
        let s = latent.shape()[1];
        let b = latent.shape()[0];
        let clean_latent = Array::zeros::<f32>(latent.shape())?.as_dtype(dt)?;
        let denoise_mask = broadcast_to(&scalar(1.0, dt)?.reshape(&[1, 1, 1])?, &[b, s, 1])?;
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

    /// Token-native view of a replace-latent [`I2vConditioning`] grid state (candle's `from_i2v`
    /// sibling): patchify latent + clean, broadcast the per-frame mask `(B,1,F,1,1)` over space and
    /// flatten it to `(B, F·H·W, 1)`. Lets the DFR stages compose grid image conditioning with
    /// appended slot/keyframe tokens on one sequence.
    pub fn from_i2v(state: &I2vConditioning, positions: &Array) -> Result<Self> {
        let latent = patchify_grid(&state.latent)?;
        let clean_latent = patchify_grid(&state.clean_latent)?;
        let sh = state.latent.shape(); // (B, C, F, H, W)
        let (b, f, h, w) = (sh[0], sh[2], sh[3], sh[4]);
        let denoise_mask =
            broadcast_to(&state.denoise_mask, &[b, 1, f, h, w])?.reshape(&[b, f * h * w, 1])?;
        let s = latent.shape()[1];
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
}

/// Per-token timesteps `σ · denoise_mask` shaped `(B, S)` for the DiT (conditioning tokens carry
/// `σ·(1−strength)`; a fully-pinned `strength=1` → `0`). Depends only on the fixed `denoise_mask` and
/// the compute `dtype`, so the denoise loop calls it directly rather than rebuilding a
/// `VideoTokenState` each step just to read it (F-060).
pub fn token_timesteps(denoise_mask: &Array, dtype: Dtype, sigma: f32) -> Result<Array> {
    let sh = denoise_mask.shape(); // (B, S, 1)
    let flat = denoise_mask.reshape(&[sh[0], sh[1]])?;
    Ok(multiply(&scalar(sigma, dtype)?, &flat)?)
}

/// Build the RoPE positions for an appended keyframe clip of latent shape `(cf, h, w)` placed at
/// `frame_offset` — port of `VideoConditionByKeyframeIndex`'s output-frame coordinate. The causal
/// first-frame fix is applied only when the output-frame offset is zero, matching the reference.
/// Output `(1, 3, cf·h·w, 2)`, f32, token order C-major
/// over `(frame, height, width)` with `[start, end]` last. Spatial axes are not divided by fps.
pub fn keyframe_append_positions(
    cf: usize,
    h: usize,
    w: usize,
    frame_offset: i32,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
) -> Array {
    let hw = h * w;
    let num = cf * hw;
    let causal = frame_offset == 0;
    let mut data = vec![0f32; 3 * num * 2];
    for p in 0..num {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let hh = (rem / w) as i64;
        let ww = (rem % w) as i64;
        for e in 0..2i64 {
            // frame axis: latent·scale → conditional causal fix → output-frame offset → /fps.
            let mut frame_pix = (t + e) * temporal_scale;
            if causal {
                frame_pix = (frame_pix + 1 - temporal_scale).max(0);
            }
            frame_pix += frame_offset as i64;
            let frame_f = frame_pix as f32 / fps;
            let height_f = ((hh + e) * spatial_scale) as f32;
            let width_f = ((ww + e) * spatial_scale) as f32;
            let base = p * 2 + e as usize;
            data[base] = frame_f;
            data[base + num * 2] = height_f;
            data[base + 2 * num * 2] = width_f;
        }
    }
    Array::from_slice(&data, &[1, 3, num as i32, 2])
}

/// Append a keyframe clip to a [`VideoTokenState`] — the IC-LoRA in-context conditioning op (reference
/// `VideoConditionByKeyframeIndex.apply_to`). `clip_latent` is the VAE-encoded clip `(B, C, cf, h, w)`
/// at the **target** spatial resolution; it is patchified and concatenated onto `latent`/`clean_latent`
/// (token axis), with `denoise_mask = 1 − strength` and positions from [`keyframe_append_positions`].
#[allow(clippy::too_many_arguments)]
pub fn append_keyframe_clip(
    state: &VideoTokenState,
    clip_latent: &Array,
    frame_offset: i32,
    strength: f32,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
) -> Result<VideoTokenState> {
    let dt = state.latent.dtype();
    let cs = clip_latent.shape(); // (B, C, cf, h, w)
    let (b, cf, h, w) = (cs[0], cs[2] as usize, cs[3] as usize, cs[4] as usize);
    let tokens = patchify_grid(&clip_latent.as_dtype(dt)?)?; // (B, cf·h·w, C)
    let n = tokens.shape()[1];
    let denoise_mask = broadcast_to(
        &scalar(1.0 - strength, dt)?.reshape(&[1, 1, 1])?,
        &[b, n, 1],
    )?;
    let positions =
        keyframe_append_positions(cf, h, w, frame_offset, temporal_scale, spatial_scale, fps);
    let positions = if b > 1 {
        broadcast_to(&positions, &[b, 3, n, 2])?
    } else {
        positions
    };
    Ok(VideoTokenState {
        latent: concatenate_axis(&[&state.latent, &tokens], 1)?,
        clean_latent: concatenate_axis(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: concatenate_axis(&[&state.denoise_mask, &denoise_mask], 1)?,
        positions: concatenate_axis(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
        // In-context clip tokens are ordinary conditioning, never keyframe slots.
        keyframes_mask: extend_keyframes_mask(state, n, false)?,
        generated_keyframe_layout: state.generated_keyframe_layout.clone(),
    })
}

// ===================================================================================================
// DFR generated-keyframe slots + single-frame keyframe / reference-latent conditioning (sc-18789).
// ===================================================================================================

use mlx_gen::gen_core::ltx_dfr::GeneratedKeyframeLayout;

/// Extend the state's keyframe marker for `num_new` appended tokens (reference
/// `mask_utils.extend_keyframes_mask`): `marked = false` keeps an absent mask absent (the common,
/// allocation-free path); any present mask — or a `marked = true` append — materializes zeros for
/// every existing token plus the new run's value.
pub fn extend_keyframes_mask(
    state: &VideoTokenState,
    num_new: i32,
    marked: bool,
) -> Result<Option<Array>> {
    let dt = state.latent.dtype();
    let b = state.latent.shape()[0];
    let existing = match &state.keyframes_mask {
        Some(mask) => mask.clone(),
        None if !marked => return Ok(None),
        None => {
            let s = state.latent.shape()[1];
            Array::zeros::<f32>(&[b, s, 1])?.as_dtype(dt)?
        }
    };
    let fill = scalar(if marked { 1.0 } else { 0.0 }, dt)?;
    let new = broadcast_to(&fill.reshape(&[1, 1, 1])?, &[b, num_new, 1])?;
    Ok(Some(concatenate_axis(&[&existing, &new], 1)?))
}

/// RoPE positions for one **single-pixel-frame** appended token block (a generated-keyframe slot,
/// or given single-frame keyframe content): full spatial grid at the target latent resolution, the
/// frame axis spanning exactly `[pixel_frame, pixel_frame + 1) / fps`. The single-frame temporal
/// extent is what distinguishes these tokens from a regular latent frame (which spans
/// `temporal_scale` pixel frames) in RoPE space — reference
/// `VideoGeneratedKeyframeSlots._slot_positions` / `VideoConditionByKeyframeIndex`
/// (`num_pixel_frames == 1` narrowing). Output `(1, 3, h·w, 2)` f32, same layout as
/// [`keyframe_append_positions`].
pub fn single_frame_positions(
    h: usize,
    w: usize,
    pixel_frame: i64,
    spatial_scale: i64,
    fps: f32,
) -> Array {
    let num = h * w;
    let mut data = vec![0f32; 3 * num * 2];
    for p in 0..num {
        let hh = (p / w) as i64;
        let ww = (p % w) as i64;
        for e in 0..2i64 {
            let base = p * 2 + e as usize;
            data[base] = (pixel_frame + e) as f32 / fps;
            data[base + num * 2] = ((hh + e) * spatial_scale) as f32;
            data[base + 2 * num * 2] = ((ww + e) * spatial_scale) as f32;
        }
    }
    Array::from_slice(&data, &[1, 3, num as i32, 2])
}

fn broadcast_positions(positions: Array, b: i32, n: i32) -> Result<Array> {
    if b > 1 {
        Ok(broadcast_to(&positions, &[b, 3, n, 2])?)
    } else {
        Ok(positions)
    }
}

/// Append **generated keyframe slots** — the DFR conditioning item
/// (`VideoGeneratedKeyframeSlots.apply_to`). Each slot occupies one latent frame's worth of tokens
/// (`h·w` at patch size 1) at the target's spatial resolution, with `denoise_mask = 1` so the
/// stage-entry noiser fills it from the slot `latent` (zeros, or `initial_keyframes` when given —
/// `clean` is ignored at mask 1) and the denoise loop generates its content. The run is marked in
/// `keyframes_mask` so it receives the learned keyframe absolute-position embedding, and recorded
/// as the state's single [`GeneratedKeyframeLayout`].
///
/// * `pixel_frame_indices` — strictly increasing, in `[0, num_pixel_frames)`.
/// * `initial_keyframes` — optional `(B, C, K, H, W)` latent content seeding the appended `latent`
///   tokens (the stage-2 / temporal-round warm start).
/// * `h`/`w` — the **target** latent spatial dims (slots always sit at target resolution).
#[allow(clippy::too_many_arguments)]
pub fn append_generated_keyframe_slots(
    state: &VideoTokenState,
    pixel_frame_indices: &[i64],
    initial_keyframes: Option<&Array>,
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
    let b = state.latent.shape()[0];
    let c = state.latent.shape()[2];
    let k = pixel_frame_indices.len();
    let tokens_per_keyframe = h * w;
    let num_new = (tokens_per_keyframe * k) as i32;

    let slot_tokens = match initial_keyframes {
        None => Array::zeros::<f32>(&[b, num_new, c])?.as_dtype(dt)?,
        Some(init) => {
            let ish = init.shape();
            if ish.len() != 5 || ish[2] as usize != k {
                return Err(Error::Msg(format!(
                    "ltx dfr: initial_keyframes must be (B, C, {k}, H, W), got {ish:?}"
                )));
            }
            if ish[0] != b {
                return Err(Error::Msg(format!(
                    "ltx dfr: initial_keyframes batch {} does not match latent batch {b}",
                    ish[0]
                )));
            }
            if (ish[3] as usize, ish[4] as usize) != (h, w) {
                return Err(Error::Msg(format!(
                    "ltx dfr: initial_keyframes spatial size ({}, {}) does not match target \
                     latent spatial size ({h}, {w})",
                    ish[3], ish[4]
                )));
            }
            // Patchify each single-frame block in index order so slot k's tokens are contiguous.
            let mut blocks = Vec::with_capacity(k);
            for index in 0..k {
                blocks.push(patchify_grid(&frame(init, index as i32)?.as_dtype(dt)?)?);
            }
            concatenate_axis(&blocks.iter().collect::<Vec<_>>(), 1)?
        }
    };

    let mut position_blocks = Vec::with_capacity(k);
    for &pixel_frame in pixel_frame_indices {
        position_blocks.push(single_frame_positions(h, w, pixel_frame, spatial_scale, fps));
    }
    let positions = concatenate_axis(&position_blocks.iter().collect::<Vec<_>>(), 2)?;
    let positions = broadcast_positions(positions, b, num_new)?;

    // denoise_mask 1 ⇒ the stage noiser lerps the slot latent toward noise (clean is ignored).
    let denoise_mask = broadcast_to(&scalar(1.0, dt)?.reshape(&[1, 1, 1])?, &[b, num_new, 1])?;
    let keyframes_mask = extend_keyframes_mask(state, num_new, true)?;
    let first_token = state.latent.shape()[1] as usize;

    Ok(VideoTokenState {
        latent: concatenate_axis(&[&state.latent, &slot_tokens], 1)?,
        clean_latent: concatenate_axis(
            &[&state.clean_latent, &Array::zeros::<f32>(&[b, num_new, c])?.as_dtype(dt)?],
            1,
        )?,
        denoise_mask: concatenate_axis(&[&state.denoise_mask, &denoise_mask], 1)?,
        positions: concatenate_axis(&[&state.positions, &positions], 2)?,
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
/// carry). `keyframes` is `(B, C, K, H, W)` with one latent frame per position; each block gets
/// placeholder zeros in the noisy latent, the content in `clean_latent`, `denoise_mask =
/// 1 − strength`, and single-frame RoPE spans. Deliberately **unmarked** in `keyframes_mask`: the
/// reference only marks generated slots, never given keyframe content.
#[allow(clippy::too_many_arguments)]
pub fn append_single_frame_keyframes(
    state: &VideoTokenState,
    keyframes: &Array,
    pixel_frame_indices: &[i64],
    strength: f32,
    spatial_scale: i64,
    fps: f32,
) -> Result<VideoTokenState> {
    let sh = keyframes.shape();
    if sh.len() != 5 {
        return Err(Error::Msg(format!(
            "ltx dfr: keyframes must be (B, C, K, H, W), got {sh:?}"
        )));
    }
    if sh[2] as usize != pixel_frame_indices.len() {
        return Err(Error::Msg(format!(
            "ltx dfr: expected {} keyframe latents, got K={}",
            pixel_frame_indices.len(),
            sh[2]
        )));
    }
    if pixel_frame_indices.iter().any(|&p| p <= 0) {
        // Position 0 never carries an anchor (frame 0 is not a keyframe on the DFR canvas), and a
        // zero offset would need the causal-fix path this single-frame builder deliberately omits.
        return Err(Error::Msg(format!(
            "ltx dfr: single-frame keyframe positions must be > 0, got {pixel_frame_indices:?}"
        )));
    }
    let dt = state.latent.dtype();
    let (b, h, w) = (sh[0], sh[3] as usize, sh[4] as usize);

    let mut token_blocks = Vec::with_capacity(pixel_frame_indices.len());
    let mut position_blocks = Vec::with_capacity(pixel_frame_indices.len());
    for (index, &pixel_frame) in pixel_frame_indices.iter().enumerate() {
        token_blocks.push(patchify_grid(&frame(keyframes, index as i32)?.as_dtype(dt)?)?);
        position_blocks.push(single_frame_positions(h, w, pixel_frame, spatial_scale, fps));
    }
    let tokens = concatenate_axis(&token_blocks.iter().collect::<Vec<_>>(), 1)?;
    let positions = concatenate_axis(&position_blocks.iter().collect::<Vec<_>>(), 2)?;
    let n = tokens.shape()[1];
    let positions = broadcast_positions(positions, b, n)?;
    let denoise_mask = broadcast_to(
        &scalar(1.0 - strength, dt)?.reshape(&[1, 1, 1])?,
        &[b, n, 1],
    )?;

    Ok(VideoTokenState {
        latent: concatenate_axis(&[&state.latent, &zeros_like_tokens(&tokens)?], 1)?,
        clean_latent: concatenate_axis(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: concatenate_axis(&[&state.denoise_mask, &denoise_mask], 1)?,
        positions: concatenate_axis(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
        keyframes_mask: extend_keyframes_mask(state, n, false)?,
        generated_keyframe_layout: state.generated_keyframe_layout.clone(),
    })
}

/// Append a **reference video latent** for IC-LoRA detailing (`VideoConditionByReferenceLatent`,
/// the `temporal_scale_factor = 1` configuration the DFR stage-2 detailing pass uses): the
/// half-res stage-1 video rides along as clean in-context tokens whose spatial positions are
/// scaled by `downscale_factor` into the target's coordinate frame (`(hh+e)·scale·d`), preserving
/// the positional relationship the detailing IC-LoRA was trained with. Frame positions keep the
/// standard causal grid (`frame_offset = 0`). Never marked as keyframes — the reference's own
/// first latent frame also spans a single pixel frame, so a position-derived marker would wrongly
/// claim it.
#[allow(clippy::too_many_arguments)]
pub fn append_reference_latent(
    state: &VideoTokenState,
    reference: &Array,
    downscale_factor: i64,
    strength: f32,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
) -> Result<VideoTokenState> {
    let sh = reference.shape(); // (B, C, F, h_ref, w_ref)
    if sh.len() != 5 {
        return Err(Error::Msg(format!(
            "ltx dfr: reference latent must be (B, C, F, H, W), got {sh:?}"
        )));
    }
    if downscale_factor < 1 {
        return Err(Error::Msg(format!(
            "ltx dfr: reference downscale_factor must be >= 1, got {downscale_factor}"
        )));
    }
    let dt = state.latent.dtype();
    let (b, cf, h, w) = (sh[0], sh[2] as usize, sh[3] as usize, sh[4] as usize);
    let tokens = patchify_grid(&reference.as_dtype(dt)?)?;
    let n = tokens.shape()[1];
    // The standard causal appended-token grid at the reference's own dims; the downscale factor
    // multiplies only the spatial axes ((hh+e)·32·d ≡ upstream's positions[:,1:] · d).
    let positions = keyframe_append_positions(
        cf,
        h,
        w,
        0,
        temporal_scale,
        spatial_scale * downscale_factor,
        fps,
    );
    let positions = broadcast_positions(positions, b, n)?;
    let denoise_mask = broadcast_to(
        &scalar(1.0 - strength, dt)?.reshape(&[1, 1, 1])?,
        &[b, n, 1],
    )?;

    Ok(VideoTokenState {
        latent: concatenate_axis(&[&state.latent, &zeros_like_tokens(&tokens)?], 1)?,
        clean_latent: concatenate_axis(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: concatenate_axis(&[&state.denoise_mask, &denoise_mask], 1)?,
        positions: concatenate_axis(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
        keyframes_mask: extend_keyframes_mask(state, n, false)?,
        generated_keyframe_layout: state.generated_keyframe_layout.clone(),
    })
}

/// Read the denoised generated-keyframe slots back out of a post-denoise state as a
/// `(B, C, K, H, W)` latent, using the recorded [`GeneratedKeyframeLayout`]. Errors when the state
/// carries no layout — the caller asked for slots it never appended.
pub fn take_generated_keyframes(state: &VideoTokenState, h: i32, w: i32) -> Result<Array> {
    let layout = state.generated_keyframe_layout.as_ref().ok_or_else(|| {
        Error::Msg(
            "ltx dfr: this state carries no generated-keyframe layout; slots were never appended"
                .into(),
        )
    })?;
    if layout.tokens_per_keyframe != (h * w) as usize {
        return Err(Error::Msg(format!(
            "ltx dfr: layout tokens_per_keyframe {} != h·w {}",
            layout.tokens_per_keyframe,
            h * w
        )));
    }
    let c = state.latent.shape()[2];
    let mut frames = Vec::with_capacity(layout.pixel_frame_indices.len());
    for k in 0..layout.pixel_frame_indices.len() {
        let start = (layout.first_token + k * layout.tokens_per_keyframe) as i32;
        let idx: Vec<i32> = (start..start + (h * w)).collect();
        let tokens = state
            .latent
            .take_axis(Array::from_slice(&idx, &[h * w]), 1)?;
        frames.push(unpatchify_grid(&tokens, c, 1, h, w)?);
    }
    concatenate_axis(&frames.iter().collect::<Vec<_>>(), 2).map_err(Into::into)
}

/// Blend a denoised latent toward the clean conditioning by the mask (reference `apply_denoise_mask`):
/// `denoised·mask + clean·(1 − mask)`. Where `mask = 0` (a fully conditioned frame) the output is the
/// clean image latent; where `mask = 1` it is the denoised generation.
pub fn apply_denoise_mask(denoised: &Array, clean: &Array, mask: &Array) -> Result<Array> {
    let dt = denoised.dtype();
    let one_minus = subtract(&scalar(1.0, dt)?, mask)?;
    Ok(add(
        &multiply(denoised, mask)?,
        &multiply(clean, &one_minus)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    #[test]
    fn apply_conditioning_pins_frame_and_builds_mask() {
        // base (1,1,3,1,1) = [10,20,30]; cond (1,1,1,1,1) = [99] at frame_idx=1, strength=0.75.
        let base = arr(&[10.0, 20.0, 30.0], &[1, 1, 3, 1, 1]);
        let cond = arr(&[99.0], &[1, 1, 1, 1, 1]);
        let st = apply_conditioning(&base, &cond, 1, 0.75).unwrap();
        // latent: frame 1 replaced by the cond, others keep base.
        assert_eq!(st.latent.as_slice::<f32>(), &[10.0, 99.0, 30.0]);
        // clean: cond at frame 1, zeros elsewhere.
        assert_eq!(st.clean_latent.as_slice::<f32>(), &[0.0, 99.0, 0.0]);
        // mask: 1 - strength at frame 1, 1 elsewhere.
        assert_eq!(st.denoise_mask.shape(), &[1, 1, 3, 1, 1]);
        let m = st.denoise_mask.as_slice::<f32>();
        assert!((m[0] - 1.0).abs() < 1e-6);
        assert!((m[1] - 0.25).abs() < 1e-6);
        assert!((m[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_keyframes_pins_first_and_last_frame() {
        // first_last_frame: base (1,1,4,1,1)=[10,20,30,40]; keyframe A=[99] @0 s=1.0,
        // keyframe B=[88] @3 s=0.5. Frames 1,2 stay base.
        let base = arr(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1, 1]);
        let a = arr(&[99.0], &[1, 1, 1, 1, 1]);
        let bb = arr(&[88.0], &[1, 1, 1, 1, 1]);
        let st = apply_keyframes(
            &base,
            &[
                Keyframe {
                    latent: &a,
                    frame_idx: 0,
                    strength: 1.0,
                },
                Keyframe {
                    latent: &bb,
                    frame_idx: 3,
                    strength: 0.5,
                },
            ],
        )
        .unwrap();
        assert_eq!(st.latent.as_slice::<f32>(), &[99.0, 20.0, 30.0, 88.0]);
        assert_eq!(st.clean_latent.as_slice::<f32>(), &[99.0, 0.0, 0.0, 88.0]);
        // mask: 1-1.0=0 @0; 1 @1,2; 1-0.5=0.5 @3.
        let m = st.denoise_mask.as_slice::<f32>();
        assert!((m[0] - 0.0).abs() < 1e-6);
        assert!((m[1] - 1.0).abs() < 1e-6);
        assert!((m[2] - 1.0).abs() < 1e-6);
        assert!((m[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn apply_keyframes_later_overrides_on_overlap() {
        // Two keyframes both at frame 0; the later (in list) wins.
        let base = arr(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let a = arr(&[5.0], &[1, 1, 1, 1, 1]);
        let bb = arr(&[7.0], &[1, 1, 1, 1, 1]);
        let st = apply_keyframes(
            &base,
            &[
                Keyframe {
                    latent: &a,
                    frame_idx: 0,
                    strength: 1.0,
                },
                Keyframe {
                    latent: &bb,
                    frame_idx: 0,
                    strength: 1.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(st.latent.as_slice::<f32>(), &[7.0, 2.0]);
    }

    #[test]
    fn noiser_pins_full_strength_frame() {
        // strength=1 → mask 0 at frame 0 → that frame keeps the clean latent regardless of noise.
        let base = arr(&[0.0, 0.0], &[1, 1, 2, 1, 1]);
        let cond = arr(&[7.0], &[1, 1, 1, 1, 1]);
        let st = apply_conditioning(&base, &cond, 0, 1.0).unwrap();
        let noise = arr(&[5.0, 5.0], &[1, 1, 2, 1, 1]);
        // scale 1.0 (stage-1 σ₀). frame 0: scaled_mask 0 → 5·0 + 7·1 = 7 (pinned); frame 1: mask 1 →
        // 5·1 + 0·0 = 5.
        let noised = st.noised(&noise, 1.0).unwrap();
        assert_eq!(noised.latent.as_slice::<f32>(), &[7.0, 5.0]);
    }

    #[test]
    fn token_timesteps_zero_at_conditioned_frame() {
        // 2 frames, 1x1 spatial → 2 tokens; strength=1 → frame0 timestep 0, frame1 = sigma.
        let base = arr(&[0.0, 0.0], &[1, 1, 2, 1, 1]);
        let cond = arr(&[1.0], &[1, 1, 1, 1, 1]);
        let st = apply_conditioning(&base, &cond, 0, 1.0).unwrap();
        let ts = st.token_timesteps(0.9, 1, 1).unwrap();
        assert_eq!(ts.shape(), &[1, 2]);
        assert_eq!(ts.as_slice::<f32>(), &[0.0, 0.9]);
    }

    #[test]
    fn free_token_timesteps_scales_mask_by_sigma() {
        // F-060: the free fn computes `σ · denoise_mask` (B,S,1)→(B,S) — conditioning tokens (mask 0)
        // get 0, generated tokens (mask 1) get σ — without rebuilding a VideoTokenState.
        let mask = arr(&[1.0, 0.0], &[1, 2, 1]);
        let ts = token_timesteps(&mask, Dtype::Float32, 0.9).unwrap();
        assert_eq!(ts.shape(), &[1, 2]);
        assert_eq!(ts.as_slice::<f32>(), &[0.9, 0.0]);
    }

    #[test]
    fn keyframe_append_positions_frame0_matches_main_grid() {
        // frame_idx=0 with causal fix == the main grid's frame-0 positions (causal-fixed). cf=1,h=1,w=2.
        let p = keyframe_append_positions(1, 1, 2, 0, 8, 32, 24.0);
        assert_eq!(p.shape(), &[1, 3, 2, 2]);
        let v = p.as_slice::<f32>();
        // frame axis (d=0): start clip(0+1-8,0)=0 → 0/24; end clip(8+1-8,0)=1 → 1/24. Same for both w.
        let at = |d: usize, tok: usize, e: usize| v[(d * 2 + tok) * 2 + e];
        assert!((at(0, 0, 0) - 0.0).abs() < 1e-7);
        assert!((at(0, 0, 1) - 1.0 / 24.0).abs() < 1e-7);
        // height axis (d=1): start 0, end 32.
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 1), 32.0);
        // width axis (d=2): token0 w=0 → [0,32]; token1 w=1 → [32,64].
        assert_eq!(at(2, 0, 0), 0.0);
        assert_eq!(at(2, 1, 0), 32.0);
        assert_eq!(at(2, 1, 1), 64.0);
    }

    #[test]
    fn keyframe_append_positions_offset_frame_no_causal() {
        // frame_idx=3 (>0): NO causal fix; frame = (t*8) + 3, /fps. cf=1,h=1,w=1.
        let p = keyframe_append_positions(1, 1, 1, 3, 8, 32, 24.0);
        let v = p.as_slice::<f32>();
        // frame start = (0+3)/24; end = (8+3)/24.
        assert!((v[0] - 3.0 / 24.0).abs() < 1e-7);
        assert!((v[1] - 11.0 / 24.0).abs() < 1e-7);
    }

    #[test]
    fn negative_one_bridge_resolves_to_the_target_output_tail() {
        let target = crate::positions::create_position_grid(1, 7, 1, 1);
        let target_values = target.as_slice::<f32>();
        let target_tail_end = target_values[(6 * 2) + 1];
        let raw_frame_idx = -1i32;
        let latent_frames = 7i32;
        let resolved_latent_idx = latent_frames + raw_frame_idx;
        assert_eq!(resolved_latent_idx, 6);
        let offset = latent_frame_to_output_offset(resolved_latent_idx, 8).unwrap();
        assert_eq!(offset, 48);
        let appended = keyframe_append_positions(1, 1, 1, offset, 8, 32, 24.0);
        let append_start = appended.as_slice::<f32>()[0];
        assert!((append_start - (target_tail_end - 1.0 / 24.0)).abs() < 1e-7);
        assert!((append_start - 48.0 / 24.0).abs() < 1e-7);
        assert!((append_start - 6.0 / 24.0).abs() > 1.0);
    }

    #[test]
    fn append_keyframe_clip_extends_tokens_and_mask() {
        // base grid (1,2,1,1,1) → 1 target token; append a 1-frame clip (1,2,1,1,1) at frame 0 s=1.0.
        let noise = arr(&[3.0, 4.0], &[1, 2, 1, 1, 1]);
        let pos = crate::positions::create_position_grid(1, 1, 1, 1);
        let st = VideoTokenState::base(&noise, &pos).unwrap();
        assert_eq!(st.latent.shape(), &[1, 1, 2]); // (B, S=1, C=2)
        assert_eq!(st.target_tokens, 1);

        let clip = arr(&[7.0, 9.0], &[1, 2, 1, 1, 1]);
        let st2 = append_keyframe_clip(&st, &clip, 0, 1.0, 8, 32, 24.0).unwrap();
        // S grows by the clip's token count (1).
        assert_eq!(st2.latent.shape(), &[1, 2, 2]);
        assert_eq!(st2.positions.shape(), &[1, 3, 2, 2]);
        assert_eq!(st2.denoise_mask.shape(), &[1, 2, 1]);
        assert_eq!(st2.target_tokens, 1); // unchanged
                                          // appended latent token == clip tokens; appended mask = 1-strength = 0.
        let lat = st2.latent.as_slice::<f32>(); // (1,2,2): [tok0=[3,4], tok1=[7,9]]
        assert_eq!(&lat[2..4], &[7.0, 9.0]);
        let m = st2.denoise_mask.as_slice::<f32>();
        assert!((m[0] - 1.0).abs() < 1e-6); // target token: generate
        assert!((m[1] - 0.0).abs() < 1e-6); // appended cond token: pinned
    }

    #[test]
    fn apply_denoise_mask_blends() {
        // mask 0 → clean; mask 1 → denoised; mask 0.5 → midpoint.
        let denoised = arr(&[10.0, 10.0, 10.0], &[3]);
        let clean = arr(&[2.0, 2.0, 2.0], &[3]);
        let mask = arr(&[0.0, 1.0, 0.5], &[3]);
        let got = apply_denoise_mask(&denoised, &clean, &mask).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[2.0, 10.0, 6.0]);
    }

    #[test]
    fn ordered_character_reference_grid_preserves_all_four_identities() {
        let image = |pixels| mlx_gen::Image {
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
        let image = mlx_gen::Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        assert!(compose_ordered_character_references(&vec![image.clone(); 5], 64, 64).is_err());
        assert!(compose_ordered_character_references(
            &[mlx_gen::Image {
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
        let image = |pixels| mlx_gen::Image {
            width: 1,
            height: 1,
            pixels,
        };
        let red = image(vec![255, 0, 0]);
        let green = image(vec![0, 255, 0]);
        let blue = image(vec![0, 0, 255]);
        let pixel = |image: &mlx_gen::Image, x: usize, y: usize| {
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
