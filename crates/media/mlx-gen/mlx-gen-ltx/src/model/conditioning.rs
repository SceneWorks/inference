//! Request conditioning: the replacement mask, the I2V reference, keyframes, clips and their
//! VAE encode.

use super::*;

/// Port of the worker's `_apply_replacement_mask` (native-LTX replace_person): blend each frame's
/// person region toward neutral gray 118 by `strength`, so the IC-LoRA keyframe-append regenerates it
/// while the background is preserved. Byte-exact to Pillow: `gate = int(L(mask) · strength)`, then
/// `out = composite(gray, frame, gate)` where `L` is PIL's RGB→L (`(R·19595 + G·38470 + B·7471 +
/// 0x8000) >> 16`) and `composite` blends with a single rounded division
/// `(gray·gate + frame·(255−gate) + 127) / 255` (verified vs Pillow). The mask must already match the
/// frame size (the host delivers per-frame masks at the output resolution).
pub fn apply_replacement_mask(frame: &Image, mask: &Image, strength: f32) -> Result<Image> {
    let strength = strength.clamp(0.0, 1.0);
    if (frame.width, frame.height) != (mask.width, mask.height) {
        return Err(Error::Msg(format!(
            "replace_person mask {}x{} must match frame {}x{}",
            mask.width, mask.height, frame.width, frame.height
        )));
    }
    let (width, height) = (frame.width as usize, frame.height as usize);
    let pixel_count = width.checked_mul(height).ok_or_else(|| {
        Error::Msg(format!(
            "replace_person frame dimensions {}x{} overflow the host pixel count",
            frame.width, frame.height
        ))
    })?;
    let expected = mlx_gen::gen_core::imageops::checked_image_buffer_len(width, height, 3)
        .ok_or_else(|| {
            Error::Msg(format!(
                "replace_person frame dimensions {}x{} overflow the RGB8 buffer size",
                frame.width, frame.height
            ))
        })?;
    if frame.pixels.len() != expected || mask.pixels.len() != expected {
        return Err(Error::Msg("replace_person frame/mask must be RGB8".into()));
    }
    let mut out = vec![0u8; expected];
    for i in 0..pixel_count {
        let (r, g, b) = (
            mask.pixels[i * 3] as u32,
            mask.pixels[i * 3 + 1] as u32,
            mask.pixels[i * 3 + 2] as u32,
        );
        let l = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16; // PIL RGB→L
        let gate = ((l as f32 * strength) as u32).min(255); // PIL .point(int(v·s))
        for c in 0..3 {
            let fpx = frame.pixels[i * 3 + c] as u32;
            // PIL `Image.composite` blend: single rounded division (not two-term MULDIV255).
            out[i * 3 + c] = ((REPLACE_NEUTRAL * gate + fpx * (255 - gate) + 127) / 255) as u8;
        }
    }
    Ok(Image {
        width: frame.width,
        height: frame.height,
        pixels: out,
    })
}

impl Ltx {
    /// Extract the single I2V conditioning image + its strength from the request. The per-reference
    /// strength wins over `req.strength`, falling back to [`DEFAULT_IMAGE_STRENGTH`]. LTX I2V
    /// conditions on exactly one image (multi-keyframe / first-last-frame is parity-plus), so more
    /// than one `Reference` is an error.
    pub(super) fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, f32)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "ltx_2_3: multiple reference images are not supported (single-image I2V \
                         only; multi-keyframe / first-last-frame is parity-plus, sc-2685)"
                            .into(),
                    ));
                }
                reference = Some((
                    image,
                    strength.or(req.strength).unwrap_or(DEFAULT_IMAGE_STRENGTH),
                ));
            }
        }
        Ok(reference)
    }

    /// Build the replace-latent keyframes (single-image I2V `Reference` at frame 0 + explicit
    /// `Keyframe`s) as owned `(stage1_latent, stage2_latent, latent_frame_idx, strength)` tuples, each
    /// VAE-encoded at both stage resolutions. [`Conditioning::Keyframe`]'s `frame_idx` is a **latent**
    /// frame index with Python-style negative indexing (`-1` = last latent frame), so first_last_frame
    /// is `[@0, @-1]` without the caller knowing the latent-frame count. Out-of-range indices error.
    pub(super) fn build_keyframes(
        &self,
        req: &GenerationRequest,
    ) -> Result<Vec<(Array, Array, i32, f32)>> {
        let lf = Self::latent_dims(req).0 as i32;
        let mut out = Vec::new();
        if let Some((image, strength)) = self.resolve_reference(req)? {
            // F-051: check + materialize before each VAE encode so a cancel during this (unbounded)
            // conditioning stage is honored between encodes rather than only at the first denoise step.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let s1 = self.encode_conditioning(image, req.height / 2, req.width / 2)?;
            let s2 = self.encode_conditioning(image, req.height, req.width)?;
            mlx_rs::transforms::eval([&s1, &s2])?;
            out.push((s1, s2, IMAGE_FRAME_IDX, strength));
        }
        if let Some(images) = req.conditioning.iter().find_map(|entry| match entry {
            Conditioning::MultiReference { images } => Some(images.as_slice()),
            _ => None,
        }) {
            // SC-20776: LTX's native IC-LoRA has exactly one frame-zero image-latent carrier.
            // Keep every ordered identity reference physically present by composing the closed
            // 1–4 surface before either VAE encode; do not accept it merely in the descriptor.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let composite = crate::conditioning::compose_ordered_character_references(
                images, req.width, req.height,
            )?;
            let s1 = self.encode_conditioning(&composite, req.height / 2, req.width / 2)?;
            let s2 = self.encode_conditioning(&composite, req.height, req.width)?;
            mlx_rs::transforms::eval([&s1, &s2])?;
            out.push((s1, s2, IMAGE_FRAME_IDX, 1.0));
        }
        for kf in req.keyframes() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let idx = if kf.frame_idx < 0 {
                lf + kf.frame_idx
            } else {
                kf.frame_idx
            };
            if idx < 0 || idx >= lf {
                return Err(Error::Msg(format!(
                    "ltx_2_3: keyframe latent frame index {} out of bounds for {lf} latent frames",
                    kf.frame_idx
                )));
            }
            let s1 = self.encode_conditioning(kf.image, req.height / 2, req.width / 2)?;
            let s2 = self.encode_conditioning(kf.image, req.height, req.width)?;
            mlx_rs::transforms::eval([&s1, &s2])?;
            out.push((s1, s2, idx, kf.strength));
        }
        Ok(out)
    }

    /// Build the in-context conditioning clips ([`Conditioning::VideoClip`] — extend_clip /
    /// video_bridge) as owned `(stage1_clip_latent, output_frame_offset, strength)` tuples, each
    /// VAE-encoded at **stage-1** (half-res) resolution into `(1, 128, cf, h1, w1)`. `frame_idx` is a
    /// latent frame index with negative-from-end indexing (`-1` = last latent frame), resolved against
    /// the target latent-frame count `lf` and converted to the output-frame coordinate required by
    /// the appended-token RoPE path. Video conditioning is stage-1 only (reference
    /// `ICLoraPipeline`), so no stage-2 encode.
    pub(super) fn build_clips(&self, req: &GenerationRequest) -> Result<Vec<(Array, i32, f32)>> {
        let lf = Self::latent_dims(req).0 as i32;
        let mut out = Vec::new();
        for clip in req.video_clips() {
            // F-051: honor a cancel between the (unbounded) per-clip VAE encodes.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            if clip.frames.is_empty() {
                return Err(Error::Msg(
                    "ltx_2_3: video conditioning clip is empty".into(),
                ));
            }
            let idx = if clip.frame_idx < 0 {
                lf + clip.frame_idx
            } else {
                clip.frame_idx
            };
            if idx < 0 || idx >= lf {
                return Err(Error::Msg(format!(
                    "ltx_2_3: clip latent frame index {} out of bounds for {lf} latent frames",
                    clip.frame_idx
                )));
            }
            let video = preprocess_conditioning_clip(clip.frames, req.width / 2, req.height / 2)?;
            let latent = self.vae.encode(&video)?;
            latent.eval()?;
            out.push((
                latent,
                crate::conditioning::latent_frame_to_output_offset(
                    idx,
                    crate::positions::TEMPORAL_SCALE,
                )?,
                clip.strength,
            ));
        }
        // replace_person: the masked control clip rides the same keyframe-append path. Build the
        // gray-neutralized control frames host-side (port of the worker's `_apply_replacement_mask`),
        // then append at `start_frame` with strength = masking_strength (the reference passes
        // `video_conditioning = [(masked_clip, masking_strength)]`). `mode` is carried on the contract
        // but does not change the math here — the per-frame mask already encodes the region (the native
        // LTX path is region-driven; `replacement_mode` only affects the diffusers WanVACE path).
        if let Some(cc) = req.control_clip() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            if cc.frames.is_empty() {
                return Err(Error::Msg(
                    "ltx_2_3: replace_person control clip is empty".into(),
                ));
            }
            if cc.frames.len() != cc.mask.len() {
                return Err(Error::Msg(format!(
                    "ltx_2_3: replace_person frame count {} != mask count {}",
                    cc.frames.len(),
                    cc.mask.len()
                )));
            }
            let idx = if cc.start_frame < 0 {
                lf + cc.start_frame
            } else {
                cc.start_frame
            };
            if idx < 0 || idx >= lf {
                return Err(Error::Msg(format!(
                    "ltx_2_3: replace_person start_frame {} out of bounds for {lf} latent frames",
                    cc.start_frame
                )));
            }
            let masked: Vec<Image> = cc
                .frames
                .iter()
                .zip(cc.mask.iter())
                .map(|(f, m)| apply_replacement_mask(f, m, cc.masking_strength))
                .collect::<Result<_>>()?;
            let video = preprocess_conditioning_clip(&masked, req.width / 2, req.height / 2)?;
            let latent = self.vae.encode(&video)?;
            latent.eval()?;
            out.push((
                latent,
                crate::conditioning::latent_frame_to_output_offset(
                    idx,
                    crate::positions::TEMPORAL_SCALE,
                )?,
                cc.masking_strength,
            ));
        }
        Ok(out)
    }

    /// VAE-encode the conditioning image at a stage's pixel resolution `(px_h, px_w)` → the f32 clean
    /// latent `(1, 128, 1, px_h/32, px_w/32)`. The encoder is an f32 quality island (like the VAE
    /// decode); the caller casts the latent to the path dtype.
    ///
    /// Re-compresses the image first at the checkpoint's resolved `default_image_crf` (sc-18759 —
    /// [`CHECKPOINT_MODEL_VERSION`] resolves to [`crate::params::LTX_2_3_PARAMS`]'s `crf: 33`)
    /// before the existing f32 normalize/layout in [`preprocess_conditioning_image`].
    pub(super) fn encode_conditioning(&self, image: &Image, px_h: u32, px_w: u32) -> Result<Array> {
        let video = condition_image_for_checkpoint(
            image,
            px_w,
            px_h,
            CHECKPOINT_MODEL_VERSION,
            None,
            &mut default_image_recompress,
        )?; // f32 (1,3,1,px_h,px_w)
        self.vae.encode(&video)
    }
}
