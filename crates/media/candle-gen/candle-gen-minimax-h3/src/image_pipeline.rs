//! MiniMax-H3-Image host pipeline on candle (SW-7091 text-to-image, SW-7092 edit): the
//! `[text | image]` and `[text | references | image]` packed layouts, the seeded initial latent, the fixed six-step velocity-form sampler, and the
//! decode-side pixel handling.
//!
//! Every piece is H3's own, at `T = 1`, rather than a family-local copy (E29):
//!
//! * the rotary grid is [`crate::dit::positions`] — text rows at `t = row`, the image's rows at
//!   `t = num_text_tokens` over [`frame_grid`] (`build_image_packed_sequence`,
//!   `modular_pipelines/minimax_h3_image/before_denoise.py`);
//! * the latent is drawn `[1, 24, 1, H, W]` and packed with
//!   [`crate::pipeline::patchify_video_latents`] (`patchify_video_latents`, same file);
//! * the sigma grid and the step coefficient are `gen_core_h3`'s
//!   [`SigmaSchedule::h3_image_distilled`] and [`SigmaSchedule::velocity_delta`] — the request
//!   contract is `gen_core::h3_image`;
//! * the DiT is [`crate::dit::ImageDit`], the H3 block stack with per-step output heads.
//!
//! # The sampler is the velocity form, not H3 video's two-source form
//!
//! `scheduler/scheduler_config.json` sets `euler_update_form = "velocity"`, so every step is
//! `x ← x + (σᵢ − σᵢ₊₁) · v` in float32, over **only** the generated rows (there are no reference
//! rows on `text2image`). The model is evaluated once per step — the checkpoint is
//! guidance-distilled — and step `i` projects through `pdd_proj_out.{i}`.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image::H3ImageSampler;
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::seed::seeded_normal_vec;
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_schedule::SigmaSchedule;
use rand::SeedableRng;

use crate::config::LATENT_CHANNELS;
use crate::denoise::{TEXT_TAG, VIDEO_TAG};
use crate::dit::positions::{
    frame_grid, reference_block_position_ids, text_position_ids, ReferenceLatentGeometry,
};
use crate::pipeline::{patchify_video_latents, unpatchify_video_rows, PIXEL_MEAN, PIXEL_STD};

/// The modality tag an image row carries — `gen_core::h3_image::IMAGE_TAG`, MiniMax-H3's video
/// slot of the AdaLN table.
pub const IMAGE_TAG: u32 = candle_gen::gen_core::h3_image::IMAGE_TAG as u32;
const _: () = assert!(IMAGE_TAG == VIDEO_TAG);
const _: () = assert!(candle_gen::gen_core::h3_image::TEXT_TAG as u32 == TEXT_TAG);

/// The packed `[text | generated image]` sequence of one `text2image` request.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageLayout {
    num_text_tokens: usize,
    /// `(latent_height, latent_width)` of every reference, in packed order; empty for
    /// `text2image`.
    reference_grids: Vec<(usize, usize)>,
    latent_height: usize,
    latent_width: usize,
    patch: [usize; 3],
    position_ids: Vec<[f64; 3]>,
    token_tags: Vec<u32>,
    text_indices: Vec<u32>,
    image_indices: Vec<u32>,
}

impl ImageLayout {
    /// `build_image_packed_sequence(text_token_tags, [], latent_h, latent_w, patch, image_tag)`:
    /// `num_text_tokens` text rows first (rotary `t = 0..L`, `h = w = 0`, tag text), then the
    /// generated image as ONE rotary frame at `t = L` over the aspect-normalized `(h, w)` grid
    /// (tag image).
    pub fn text_to_image(
        num_text_tokens: usize,
        latent_height: usize,
        latent_width: usize,
        patch: [usize; 3],
    ) -> Result<Self> {
        if num_text_tokens == 0 {
            return Err(CandleError::Msg(
                "minimax-h3-image layout: the prompt encoded to no tokens".into(),
            ));
        }
        if patch[0] != 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image layout: an image is one latent frame, so the temporal patch must \
                 be 1, got {patch:?}"
            )));
        }
        let (grid, _) = frame_grid(latent_height, latent_width, patch[1], patch[2])?;
        let mut position_ids = text_position_ids(num_text_tokens)?;
        let t = num_text_tokens as f64;
        position_ids.extend(grid.iter().map(|&[h, w]| [t, h, w]));
        let seq = position_ids.len();
        let mut token_tags = vec![TEXT_TAG; num_text_tokens];
        token_tags.resize(seq, IMAGE_TAG);
        Ok(Self {
            num_text_tokens,
            reference_grids: Vec::new(),
            latent_height,
            latent_width,
            patch,
            position_ids,
            token_tags,
            text_indices: (0..num_text_tokens as u32).collect(),
            image_indices: (num_text_tokens as u32..seq as u32).collect(),
        })
    }

    /// `build_image_packed_sequence(text_token_tags, reference_grids, ...)` for an **edit**:
    /// the text rows with the presentation's own tags (vision-block rows image-tagged), then every
    /// reference as ONE rotary frame on its own aspect-normalized grid at one integer slot each
    /// from `t = L` — H3's `ref2va` image reference block — then the generated image at the slot
    /// after the last reference. Every reference and generated row is image-tagged.
    pub fn edit(
        text_tags: &[u32],
        reference_grids: &[(usize, usize)],
        latent_height: usize,
        latent_width: usize,
        patch: [usize; 3],
    ) -> Result<Self> {
        if reference_grids.is_empty()
            || reference_grids.len() > candle_gen::gen_core::h3_image::MAX_REFERENCES
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3-image layout: an edit takes 1..={} references, got {}",
                candle_gen::gen_core::h3_image::MAX_REFERENCES,
                reference_grids.len()
            )));
        }
        let mut layout = Self::text_to_image(text_tags.len(), latent_height, latent_width, patch)?;
        let num_text = text_tags.len();
        let target = layout.position_ids.split_off(num_text);
        let (_, target_w) = frame_grid(latent_height, latent_width, patch[1], patch[2])?;
        let mut clock = num_text as f64;
        for &(h, w) in reference_grids {
            if !h.is_multiple_of(patch[1]) || !w.is_multiple_of(patch[2]) {
                return Err(CandleError::Msg(format!(
                    "minimax-h3-image layout: a {h}x{w} reference latent is not whole {patch:?}                      patches"
                )));
            }
            let geometry = ReferenceLatentGeometry {
                kind: crate::reference::ReferenceKind::Image,
                num_latent_frames: 1,
                latent_height: h,
                latent_width: w,
                num_audio_latents: 0,
            };
            let (_, video, advance) =
                reference_block_position_ids(&geometry, clock, patch[1], patch[2], 0, &target_w)?;
            layout.position_ids.extend(video);
            clock += advance;
        }
        layout
            .position_ids
            .extend(target.into_iter().map(|[_, h, w]| [clock, h, w]));
        let seq = layout.position_ids.len();
        layout.token_tags = text_tags.to_vec();
        layout.token_tags.resize(seq, IMAGE_TAG);
        layout.image_indices = (num_text as u32..seq as u32).collect();
        layout.reference_grids = reference_grids.to_vec();
        Ok(layout)
    }

    /// The references' latent grids, in packed order (empty for `text2image`).
    pub fn reference_grids(&self) -> &[(usize, usize)] {
        &self.reference_grids
    }

    /// Leading image rows that are reference rows rather than generated ones.
    pub fn num_reference_rows(&self) -> usize {
        self.reference_grids
            .iter()
            .map(|&(h, w)| (h / self.patch[1]) * (w / self.patch[2]))
            .sum()
    }

    /// Leading packed-sequence conditioning rows (`text + references`), unperturbed under PAG.
    pub fn num_conditioning_rows(&self) -> usize {
        self.num_text_tokens + self.num_reference_rows()
    }

    /// Per sequence row, its AdaLN row class: `1` for a reference row (held at
    /// `REFERENCE_NOISE_AUG`), `0` for every text and generated row (the step's timestep).
    pub fn row_classes(&self) -> Vec<u32> {
        let mut classes = vec![0u32; self.seq_len()];
        let start = self.num_text_tokens;
        classes[start..start + self.num_reference_rows()].fill(1);
        classes
    }

    /// Text rows.
    pub fn num_text_tokens(&self) -> usize {
        self.num_text_tokens
    }

    /// Generated image rows: `(latent_height / ph) · (latent_width / pw)`.
    pub fn num_image_rows(&self) -> usize {
        self.image_indices.len() - self.num_reference_rows()
    }

    /// Image rows the DiT reads: the reference rows then the generated rows.
    pub fn num_packed_image_rows(&self) -> usize {
        self.image_indices.len()
    }

    /// Text + image rows.
    pub fn seq_len(&self) -> usize {
        self.position_ids.len()
    }

    /// The latent grid.
    pub fn latent_hw(&self) -> (usize, usize) {
        (self.latent_height, self.latent_width)
    }

    /// The DiT patch.
    pub fn patch(&self) -> [usize; 3] {
        self.patch
    }

    /// The `(t, h, w)` rotary coordinate of every row.
    pub fn position_ids(&self) -> &[[f64; 3]] {
        &self.position_ids
    }

    /// The per-row modality tag (text 1, image 0).
    pub fn token_tags(&self) -> &[u32] {
        &self.token_tags
    }

    /// Sequence positions of the text rows.
    pub fn text_indices(&self) -> &[u32] {
        &self.text_indices
    }

    /// Sequence positions of the image rows, reference rows first.
    pub fn image_indices(&self) -> &[u32] {
        &self.image_indices
    }
}

/// The latent grid of a pixel canvas — `gen_core::h3_image::latent_grid` (the VAE's 16x spatial
/// compression of a 32-multiple canvas), as `usize` for the tensor shapes.
pub fn latent_grid(canvas_width: u32, canvas_height: u32) -> Result<(usize, usize)> {
    let (h, w) = candle_gen::gen_core::h3_image::latent_grid(canvas_width, canvas_height)
        .map_err(|e| CandleError::Msg(e.to_string()))?;
    Ok((h as usize, w as usize))
}

/// The seeded initial latent of the generated image, packed into rows: drawn as
/// `[1, 24, 1, H, W]` float32 on the host stream (the same `StdRng` + `seeded_normal_vec` draw
/// H3 video uses), then patchified — `[1, rows, 96]`.
pub fn initial_image_rows(
    latent_height: usize,
    latent_width: usize,
    patch: [usize; 3],
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    candle_gen::execution::draw("h3_image_initial_latents", seed, 0);
    candle_gen::execution::runtime(device, DType::F32);
    let shape = (1usize, LATENT_CHANNELS, 1usize, latent_height, latent_width);
    let n = LATENT_CHANNELS * latent_height * latent_width;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let latents = Tensor::from_vec(seeded_normal_vec(&mut rng, n), shape, device)?;
    patchify_video_latents(&latents, patch)
}

/// An edit's conditioning rows (`MiniMaxH3ImageEditPrepareLatentsStep`): every reference latent
/// `[1, 24, h, w]` noised to `REFERENCE_NOISE_AUG` with its `noise` (`t · x + (1 - t) · n`),
/// patchified as one latent frame, concatenated in packed order — `[1, num_reference_rows, 96]`.
pub fn reference_condition_rows(
    latents: &[Tensor],
    noise: &[Tensor],
    patch: [usize; 3],
) -> Result<Tensor> {
    if latents.len() != noise.len() || latents.is_empty() {
        return Err(CandleError::Msg(format!(
            "minimax-h3-image: {} reference latents for {} noise tensors",
            latents.len(),
            noise.len()
        )));
    }
    let mut rows = Vec::with_capacity(latents.len());
    for (latent, n) in latents.iter().zip(noise) {
        let latent = latent.to_dtype(DType::F32)?;
        let noised = crate::conditioning::scale_noise(
            &latent,
            candle_gen::gen_core::h3_image::REFERENCE_NOISE_AUG,
            n,
        )?;
        rows.push(patchify_video_latents(&noised.unsqueeze(2)?, patch)?);
    }
    Ok(Tensor::cat(&rows, 1)?)
}

/// Seeded standard-normal noise for an edit's per-reference draws, on the host `StdRng` stream
/// H3 uses: the posterior sample (`stream` 0) and the conditioning augmentation (`stream` 1) of
/// reference `index`, each its own seed so no two draws share a stream and none disturbs the
/// generated image's noise.
pub fn reference_noise(
    seed: u64,
    stream: u64,
    index: usize,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let seed = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(stream.wrapping_mul(1 << 32))
        .wrapping_add(index as u64 + 1);
    candle_gen::execution::stream("h3_image_reference_noise", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = shape.iter().product();
    Ok(Tensor::from_vec(
        seeded_normal_vec(&mut rng, n),
        shape,
        device,
    )?)
}

/// One edit reference resized for one of its consumers with the PIL-equivalent Lanczos
/// `VaeImageProcessor.resize` uses (`gen_core::imageops::resize_lanczos_u8`); a reference already
/// at `(width, height)` keeps its pixels.
pub fn resize_reference(image: &Image, (width, height): (u32, u32)) -> Result<Image> {
    if (image.width, image.height) == (width, height) {
        return Ok(image.clone());
    }
    let resized = candle_gen::gen_core::imageops::resize_lanczos_u8(
        &image.pixels,
        image.height as usize,
        image.width as usize,
        height as usize,
        width as usize,
    )
    .map_err(|e| CandleError::Msg(e.to_string()))?;
    Ok(Image {
        width,
        height,
        pixels: resized
            .into_iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    })
}

/// An edit reference's VAE pixels: cropped at the top left to whole 16 px cells, then
/// ImageNet-normalized to `[1, 3, H, W]` (`MiniMaxH3ImageEditVaeEncoderStep`).
pub fn reference_vae_pixels(image: &Image, device: &Device) -> Result<Tensor> {
    let r = candle_gen::gen_core::h3_image::VAE_SPATIAL_RATIO;
    let (w, h) = (image.width / r * r, image.height / r * r);
    if w == 0 || h == 0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3-image: a {}x{} reference holds no whole {r}px VAE cell",
            image.width, image.height
        )));
    }
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as usize {
        let row = y * image.width as usize * 3;
        pixels.extend_from_slice(&image.pixels[row..row + w as usize * 3]);
    }
    let cropped = Image {
        width: w,
        height: h,
        pixels,
    };
    Ok(crate::keyframe::keyframe_to_vae_pixels(&cropped, device)?.squeeze(2)?)
}

/// The packed image rows back to the VAE's `[1, 24, H, W]` latent.
pub fn unpack_image_rows(rows: &Tensor, layout: &ImageLayout) -> Result<Tensor> {
    let (h, w) = layout.latent_hw();
    let latents = unpatchify_video_rows(rows, LATENT_CHANNELS, 1, h, w, layout.patch())?;
    Ok(latents.squeeze(2)?)
}

/// The velocity model one denoise step evaluates: `(step, image_rows) → velocity`.
pub trait ImageVelocity {
    /// Evaluate the model at sampling step `step` (which selects the output head).
    fn forward(&mut self, step: usize, image_rows: &Tensor) -> Result<Tensor>;

    /// Evaluate the second conditional PAG pass at sampling step `step` (selecting the same
    /// `pdd_proj_out.{step}` output head while perturbing only target-image self-attention on
    /// `perturbed_layers`).
    fn forward_perturbed(
        &mut self,
        _step: usize,
        _image_rows: &Tensor,
        _perturbed_layers: &[u32],
    ) -> Result<Tensor> {
        Err(CandleError::Unsupported(
            "minimax-h3-image: ImageVelocity implementation does not support PAG perturbation"
                .into(),
        ))
    }

    /// Evaluate at sampling step `step` projecting through the PDD head mix `heads`
    /// ([`candle_gen::gen_core::h3_image::pdd_head_mix`], SW-7224). The default serves only the
    /// one-head-per-step mix `[(step, 1)]`, through [`Self::forward`].
    fn forward_heads(
        &mut self,
        step: usize,
        heads: &[(usize, f32)],
        image_rows: &Tensor,
    ) -> Result<Tensor> {
        if heads == [(step, 1.0)] {
            return self.forward(step, image_rows);
        }
        Err(CandleError::Unsupported(format!(
            "minimax-h3-image: ImageVelocity implementation cannot mix output heads {heads:?}"
        )))
    }

    /// [`Self::forward_perturbed`] through the PDD head mix `heads`; the default serves only
    /// `[(step, 1)]`.
    fn forward_perturbed_heads(
        &mut self,
        step: usize,
        heads: &[(usize, f32)],
        image_rows: &Tensor,
        perturbed_layers: &[u32],
    ) -> Result<Tensor> {
        if heads == [(step, 1.0)] {
            return self.forward_perturbed(step, image_rows, perturbed_layers);
        }
        Err(CandleError::Unsupported(format!(
            "minimax-h3-image: ImageVelocity implementation cannot mix output heads {heads:?}"
        )))
    }
}

impl<T: ImageVelocity + ?Sized> ImageVelocity for &mut T {
    fn forward(&mut self, step: usize, image_rows: &Tensor) -> Result<Tensor> {
        (**self).forward(step, image_rows)
    }

    fn forward_perturbed(
        &mut self,
        step: usize,
        image_rows: &Tensor,
        perturbed_layers: &[u32],
    ) -> Result<Tensor> {
        (**self).forward_perturbed(step, image_rows, perturbed_layers)
    }

    fn forward_heads(
        &mut self,
        step: usize,
        heads: &[(usize, f32)],
        image_rows: &Tensor,
    ) -> Result<Tensor> {
        (**self).forward_heads(step, heads, image_rows)
    }

    fn forward_perturbed_heads(
        &mut self,
        step: usize,
        heads: &[(usize, f32)],
        image_rows: &Tensor,
        perturbed_layers: &[u32],
    ) -> Result<Tensor> {
        (**self).forward_perturbed_heads(step, heads, image_rows, perturbed_layers)
    }
}

/// **The six-step loop**: one forward per step, then gen-core's shared
/// [`sampler_step`] in float32 — under
/// [`H3ImageSampler::Euler`] `x ← x + (σᵢ − σᵢ₊₁) · v`, under
/// [`H3ImageSampler::Lcm`] (the request default) ComfyUI's re-noised LCM step with noise seeded from `image_seed`
/// ([`crate::image_sampler`]). Both rungs (resident and streamed) run through this one loop.
///
/// Cancellation is checked before every step and the device synchronized after it, so a cancel
/// lands within one step and `on_step(completed)` reports work the GPU has actually done.
///
/// `initial_rows` is the packed image stream, an edit's `num_reference_rows` reference rows first;
/// only the generated rows after them are ever written
/// (`MiniMaxH3ImageLoopSchedulerStep`), so the conditioning rides through every step unchanged.
/// Returns the generated rows alone.
#[allow(clippy::too_many_arguments)]
pub fn denoise_image(
    model: &mut dyn ImageVelocity,
    schedule: &SigmaSchedule,
    sampler: &H3ImageSampler,
    image_seed: u64,
    initial_rows: &Tensor,
    num_reference_rows: usize,
    device: &Device,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Tensor> {
    denoise_image_with_pag(
        model,
        schedule,
        sampler,
        image_seed,
        initial_rows,
        num_reference_rows,
        device,
        cancel,
        None,
        on_step,
    )
}

/// [`denoise_image`] with optional Perturbed-Attention Guidance (`pag`, SW-7175).
///
/// When `pag` is active at step `i` (`pag.scale_at_step(i, num_steps) > 0.0`), executes two
/// conditional passes (`cond` then `perturbed`) through the identical step `i` `pdd_proj_out.{i}`
/// head, combines generated velocities via `pag_apply` (`v_cond + pag_scale * (v_cond - v_pert)`),
/// and invokes [`sampler_step`] once so stochastic LCM RNG consumption stays one draw per step.
#[allow(clippy::too_many_arguments)]
pub fn denoise_image_with_pag(
    model: &mut dyn ImageVelocity,
    schedule: &SigmaSchedule,
    sampler: &H3ImageSampler,
    image_seed: u64,
    initial_rows: &Tensor,
    num_reference_rows: usize,
    device: &Device,
    cancel: &CancelFlag,
    pag: Option<&candle_gen::gen_core::PagResolved>,
    on_step: &mut dyn FnMut(usize),
) -> Result<Tensor> {
    let total = initial_rows.dims()[1];
    if num_reference_rows >= total {
        return Err(CandleError::Msg(format!(
            "minimax-h3-image denoise: {num_reference_rows} reference rows of {total} leave no \
             generated row"
        )));
    }
    let condition = (num_reference_rows > 0)
        .then(|| {
            initial_rows
                .narrow(1, 0, num_reference_rows)?
                .to_dtype(DType::F32)
        })
        .transpose()?;
    let x = initial_rows
        .narrow(1, num_reference_rows, total - num_reference_rows)?
        .to_dtype(DType::F32)?;
    let num_steps = schedule.num_evals();
    candle_gen::execution::schedule(sampler.name(), schedule.sigmas(), "velocity");
    candle_gen::execution::policy(
        "h3_image_sampler",
        match *sampler {
            H3ImageSampler::Euler => serde_json::json!({ "solver": "euler" }),
            H3ImageSampler::EulerAncestral => serde_json::json!({
                "solver": "euler_ancestral",
                "eta_bits": candle_gen::gen_core::h3_image::ANCESTRAL_ETA.to_bits(),
                "s_noise_bits": candle_gen::gen_core::h3_image::ANCESTRAL_S_NOISE.to_bits(),
            }),
            H3ImageSampler::Lcm {
                s_noise,
                s_noise_end,
                noise_clip_std,
            } => serde_json::json!({
                "solver": "lcm",
                "s_noise_bits": s_noise.to_bits(),
                "s_noise_end_bits": s_noise_end.to_bits(),
                "noise_clip_std_bits": noise_clip_std.to_bits(),
            }),
        },
    );
    if let Some(pag) = pag {
        candle_gen::execution::policy(
            "h3_image_pag",
            serde_json::json!({
                "scale_bits": pag.scale.to_bits(),
                "adaptive_scale_bits": pag.adaptive_scale.to_bits(),
                "layers": &pag.layers,
                "start_bits": pag.start.to_bits(),
                "stop_bits": pag.stop.to_bits(),
                "active_steps": pag.active_steps(num_steps as u32),
            }),
        );
    }

    let driver = crate::drivers::CandleH3ImageDriver::new(
        "candle_h3_image",
        model,
        std::borrow::Cow::Borrowed(schedule),
        *sampler,
        pag.map(std::borrow::Cow::Borrowed),
        image_seed,
        condition,
        device.clone(),
    )?;
    let mut state = candle_gen::driver::DriverRunState::new();
    let mut progress = |event: candle_gen::gen_core::Progress| {
        if let candle_gen::gen_core::Progress::Step { current, .. } = event {
            on_step(current as usize);
        }
    };
    let mut hooks = candle_gen::driver::StepHooks::new(&mut progress);
    let out = candle_gen::run_candle_driver_slices(
        &driver,
        [candle_gen::driver::SliceSpec::full(num_steps)],
        candle_gen::gen_core::BoundaryState::XSigma(x),
        num_steps,
        image_seed,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, s, _| Ok(s),
    )?;
    match out {
        candle_gen::gen_core::BoundaryState::XSigma(x)
        | candle_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | candle_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => Ok(x),
    }
}

/// The decoded, ImageNet-normalized `[1, 3, H, W]` RGB to the delivered image, the released
/// pipeline's way (`decoders.py`): revert the normalization, clamp to `[0, 1]`, quantize to 8 bits
/// **by truncation** (`(image * 255).clamp(0, 255).to(uint8)`), then — when the canvas differs
/// from the requested size — resize the 8-bit image to `width x height` with a Lanczos filter.
pub fn decoded_to_image(decoded: &Tensor, width: u32, height: u32) -> Result<Image> {
    let s = decoded.dims();
    if s.len() != 4 || s[0] != 1 || s[1] != 3 {
        return Err(CandleError::Msg(format!(
            "minimax-h3-image decode: expected [1, 3, H, W], got {s:?}"
        )));
    }
    let (h, w) = (s[2], s[3]);
    let dev = decoded.device();
    let mean = Tensor::from_vec(PIXEL_MEAN.to_vec(), (1, 3, 1, 1), dev)?;
    let std = Tensor::from_vec(PIXEL_STD.to_vec(), (1, 3, 1, 1), dev)?;
    let pixels = decoded
        .to_dtype(DType::F32)?
        .broadcast_mul(&std)?
        .broadcast_add(&mean)?
        .clamp(0f32, 1f32)?;
    // Truncation, not rounding: `.to(torch.uint8)` of a non-negative float drops the fraction.
    let bytes = (pixels * 255.0)?
        .clamp(0f32, 255f32)?
        .floor()?
        .to_dtype(DType::U8)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .contiguous()?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<u8>()?;
    let image = Image {
        width: w as u32,
        height: h as u32,
        pixels: bytes,
    };
    if (image.width, image.height) == (width, height) {
        return Ok(image);
    }
    resize_lanczos(&image, width, height)
}

/// Resize an 8-bit RGB image with a Lanczos (a = 3) filter — the `VaeImageProcessor.resize` the
/// released pipeline applies when the canvas is larger than the request. `image`'s Lanczos3 is the
/// same windowed sinc as PIL's `LANCZOS`, not a byte-identical implementation of it.
fn resize_lanczos(image: &Image, width: u32, height: u32) -> Result<Image> {
    let rgb = image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .ok_or_else(|| {
            CandleError::Msg("minimax-h3-image decode: pixel buffer does not match its size".into())
        })?;
    let resized =
        image::imageops::resize(&rgb, width, height, image::imageops::FilterType::Lanczos3);
    Ok(Image {
        width,
        height,
        pixels: resized.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The t2i layout is `[text | image]`: text at `t = row` with `h = w = 0` and tag 1, the image
    /// as one frame at `t = L` over the aspect-normalized grid with tag 0, and the index tensors
    /// partition the sequence.
    #[test]
    fn the_text_to_image_layout_is_text_then_one_image_frame() {
        let l = ImageLayout::text_to_image(3, 4, 8, [1, 2, 2]).unwrap();
        assert_eq!(l.num_text_tokens(), 3);
        assert_eq!(l.num_conditioning_rows(), 3);
        assert_eq!(l.num_image_rows(), 2 * 4);
        assert_eq!(l.seq_len(), 11);
        assert_eq!(l.text_indices(), &[0, 1, 2]);
        assert_eq!(l.image_indices(), &(3..11).collect::<Vec<u32>>()[..]);
        assert_eq!(&l.token_tags()[..3], &[1, 1, 1]);
        assert!(l.token_tags()[3..].iter().all(|&t| t == 0));
        assert_eq!(l.position_ids()[1], [1.0, 0.0, 0.0]);
        assert!(l.position_ids()[3..].iter().all(|p| p[0] == 3.0));
        // 4x8 latent: sqrt_area = sqrt(32); the h axis spans less than the w axis.
        let (grid, _) = frame_grid(4, 8, 2, 2).unwrap();
        for (row, g) in l.position_ids()[3..].iter().zip(&grid) {
            assert_eq!([row[1], row[2]], *g);
        }
        assert!(ImageLayout::text_to_image(0, 4, 4, [1, 2, 2]).is_err());
        assert!(ImageLayout::text_to_image(2, 4, 4, [2, 2, 2]).is_err());
        assert!(ImageLayout::text_to_image(2, 3, 4, [1, 2, 2]).is_err());
    }

    /// The canvas → latent arithmetic, and that a 2048² request is a 128² latent of 64² rows.
    #[test]
    fn the_latent_grid_is_the_canvas_over_sixteen() {
        assert_eq!(latent_grid(2048, 2048).unwrap(), (128, 128));
        assert_eq!(latent_grid(512, 1024).unwrap(), (64, 32));
        assert!(latent_grid(2048, 2040).is_err());
        assert!(latent_grid(0, 2048).is_err());
        let l = ImageLayout::text_to_image(10, 128, 128, [1, 2, 2]).unwrap();
        assert_eq!(l.num_image_rows(), 64 * 64);
    }

    /// The initial rows are the patchified `[1, 24, 1, H, W]` draw, and the round trip through
    /// [`unpack_image_rows`] recovers the latent.
    #[test]
    fn the_initial_latent_round_trips_through_the_packed_rows() {
        let d = Device::Cpu;
        let rows = initial_image_rows(4, 6, [1, 2, 2], 7, &d).unwrap();
        assert_eq!(rows.dims(), &[1, 6, 96]);
        let again = initial_image_rows(4, 6, [1, 2, 2], 7, &d).unwrap();
        assert_eq!(
            rows.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            again.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the draw is a pure function of the seed"
        );
        let l = ImageLayout::text_to_image(1, 4, 6, [1, 2, 2]).unwrap();
        let latents = unpack_image_rows(&rows, &l).unwrap();
        assert_eq!(latents.dims(), &[1, 24, 4, 6]);
        let repacked = patchify_video_latents(&latents.unsqueeze(2).unwrap(), [1, 2, 2]).unwrap();
        assert_eq!(
            repacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            rows.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    struct Recorder {
        steps: Vec<usize>,
    }

    impl ImageVelocity for Recorder {
        fn forward(&mut self, step: usize, rows: &Tensor) -> Result<Tensor> {
            self.steps.push(step);
            Ok(Tensor::ones_like(rows)?)
        }
    }

    /// The loop evaluates each of the six steps once, in order (so step `i` reaches head `i`), and
    /// a unit velocity moves the rows by exactly the whole grid, Σ(σᵢ − σᵢ₊₁) = 1. A host-side
    /// recorder stands in for the DiT; no model runs.
    #[test]
    fn the_loop_runs_each_step_once_and_steps_by_the_velocity_form() {
        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image_distilled().unwrap();
        let x0 = Tensor::zeros((1, 2, 96), DType::F32, &d).unwrap();
        let mut model = Recorder { steps: Vec::new() };
        let mut seen = Vec::new();
        let out = denoise_image(
            &mut model,
            &schedule,
            &H3ImageSampler::Euler,
            0,
            &x0,
            0,
            &d,
            &CancelFlag::default(),
            &mut |n| seen.push(n),
        )
        .unwrap();
        assert_eq!(model.steps, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(seen, vec![1, 2, 3, 4, 5, 6]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-6), "{:?}", &v[..4]);

        let cancel = CancelFlag::default();
        cancel.cancel();
        let mut model = Recorder { steps: Vec::new() };
        assert!(matches!(
            denoise_image(
                &mut model,
                &schedule,
                &H3ImageSampler::Euler,
                0,
                &x0,
                0,
                &d,
                &cancel,
                &mut |_| {}
            ),
            Err(CandleError::Canceled)
        ));
        assert!(model.steps.is_empty());
    }

    /// Records the head mix each step asks for (SW-7224).
    struct HeadRecorder {
        heads: Vec<Vec<(usize, f32)>>,
    }

    impl ImageVelocity for HeadRecorder {
        fn forward(&mut self, _step: usize, _rows: &Tensor) -> Result<Tensor> {
            unreachable!("the loop evaluates through forward_heads")
        }
        fn forward_heads(
            &mut self,
            _step: usize,
            heads: &[(usize, f32)],
            rows: &Tensor,
        ) -> Result<Tensor> {
            self.heads.push(heads.to_vec());
            Ok(Tensor::ones_like(rows)?)
        }
    }

    /// Under a ComfyUI scheduler the loop steps on that scheduler's grid and hands each step its
    /// PDD head mix: `normal`'s step 2 mixes heads 2 and 3, and its last two steps share head 5.
    /// Euler ancestral lands on the clean estimate, so a unit velocity still moves the rows by the
    /// grid's whole span of 1.
    #[test]
    fn a_scheduler_grid_drives_the_steps_and_the_head_mix() {
        use candle_gen::gen_core::h3_image::{pdd_head_mix, H3ImageScheduler};
        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image(H3ImageScheduler::Normal).unwrap();
        assert_eq!(schedule.sigmas(), &H3ImageScheduler::Normal.sigmas()[..]);
        let x0 = Tensor::zeros((1, 2, 96), DType::F32, &d).unwrap();
        let mut model = HeadRecorder { heads: Vec::new() };
        denoise_image(
            &mut model,
            &schedule,
            &H3ImageSampler::Euler,
            0,
            &x0,
            0,
            &d,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap();
        let want: Vec<_> = (0..6)
            .map(|i| pdd_head_mix(schedule.sigmas(), i).unwrap())
            .collect();
        assert_eq!(model.heads, want);
        assert_eq!(
            model.heads[2].iter().map(|h| h.0).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(model.heads[4], vec![(5, 1.0)]);
        assert_eq!(model.heads[5], vec![(5, 1.0)]);

        let mut model = HeadRecorder { heads: Vec::new() };
        let out = denoise_image(
            &mut model,
            &schedule,
            &H3ImageSampler::EulerAncestral,
            7,
            &x0,
            0,
            &d,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(model.heads.len(), 6);
        assert_eq!(out.dims(), x0.dims());
    }

    /// A model that cannot mix heads refuses a mixed step rather than projecting through one head.
    #[test]
    fn a_model_without_head_mixing_refuses_a_mixed_step() {
        use candle_gen::gen_core::h3_image::H3ImageScheduler;
        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image(H3ImageScheduler::Normal).unwrap();
        let x0 = Tensor::zeros((1, 2, 96), DType::F32, &d).unwrap();
        let mut model = Recorder { steps: Vec::new() };
        let err = denoise_image(
            &mut model,
            &schedule,
            &H3ImageSampler::Euler,
            0,
            &x0,
            0,
            &d,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap_err();
        assert!(matches!(err, CandleError::Unsupported(_)), "{err:?}");
        assert_eq!(model.steps, vec![0, 1], "steps 0 and 1 are single-head");
    }

    /// Delivery: ImageNet revert, clamp, truncating 8-bit quantize, and a resize only when the
    /// canvas differs from the request.
    #[test]
    fn decoded_pixels_are_reverted_truncated_and_resized_back() {
        let d = Device::Cpu;
        // Normalized value that reverts to exactly 0.5 in every channel: (0.5 - mean) / std.
        let vals: Vec<f32> = (0..3)
            .flat_map(|c| std::iter::repeat_n((0.5 - PIXEL_MEAN[c]) / PIXEL_STD[c], 4 * 6))
            .collect();
        let decoded = Tensor::from_vec(vals, (1, 3, 4, 6), &d).unwrap();
        let img = decoded_to_image(&decoded, 6, 4).unwrap();
        assert_eq!((img.width, img.height), (6, 4));
        assert_eq!(img.pixels.len(), 6 * 4 * 3);
        // 0.5 * 255 = 127.5 -> truncated to 127 (rounding would give 128).
        assert!(
            img.pixels.iter().all(|&p| p == 127),
            "{:?}",
            &img.pixels[..6]
        );
        let small = decoded_to_image(&decoded, 3, 2).unwrap();
        assert_eq!((small.width, small.height), (3, 2));
        assert_eq!(small.pixels.len(), 3 * 2 * 3);
    }

    /// The edit layout: references at consecutive integer slots after the text, each on its own
    /// grid, the target one slot after the last; reference rows image-tagged and class 1.
    #[test]
    fn the_edit_layout_packs_references_between_text_and_image() {
        let l = ImageLayout::edit(&[1, 0, 0, 1], &[(4, 4), (2, 4)], 4, 4, [1, 2, 2]).unwrap();
        assert_eq!(l.num_reference_rows(), 6);
        assert_eq!(l.num_conditioning_rows(), 10);
        assert_eq!(l.num_image_rows(), 4);
        assert_eq!(l.num_packed_image_rows(), 10);
        assert_eq!(l.seq_len(), 14);
        assert_eq!(l.token_tags(), &[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(l.image_indices(), &(4..14).collect::<Vec<u32>>()[..]);
        let p = l.position_ids();
        assert_eq!(p[4], [4.0, 0.0, 0.0]);
        assert_eq!(p[7], [4.0, 16.0, 16.0]);
        assert_eq!(p[8][0], 5.0);
        assert!((p[8][1] - (1.0 - 2.0 / 8f64.sqrt()) / 2.0 * 32.0).abs() < 1e-12);
        assert_eq!(p[10], [6.0, 0.0, 0.0]);
        assert_eq!(p[13], [6.0, 16.0, 16.0]);
        assert_eq!(
            l.row_classes(),
            vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        );
        assert!(ImageLayout::edit(&[1], &[], 4, 4, [1, 2, 2]).is_err());
        assert!(ImageLayout::edit(&[1], &[(3, 4)], 4, 4, [1, 2, 2]).is_err());
        assert!(ImageLayout::edit(&[1], &[(2, 2); 17], 4, 4, [1, 2, 2]).is_err());
        assert_eq!(
            ImageLayout::edit(&[1], &[(2, 2); 16], 4, 4, [1, 2, 2])
                .unwrap()
                .num_reference_rows(),
            16
        );
    }

    /// Records the reference rows every step was evaluated with, and returns a unit velocity.
    struct RefRecorder {
        refs: usize,
        seen: Vec<Vec<f32>>,
    }

    impl ImageVelocity for RefRecorder {
        fn forward(&mut self, _step: usize, rows: &Tensor) -> Result<Tensor> {
            self.seen.push(
                rows.narrow(1, 0, self.refs)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            Ok(Tensor::ones_like(rows)?)
        }
    }

    /// SW-7106: under LCM the reference rows still ride through every step unchanged, the run is
    /// a pure function of the image seed, and it differs from Euler and from another seed.
    #[test]
    fn the_lcm_loop_holds_reference_rows_and_is_seeded() {
        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image_distilled().unwrap();
        let x0 = Tensor::arange(0f32, 5.0 * 96.0, &d)
            .unwrap()
            .reshape((1, 5, 96))
            .unwrap();
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let run = |sampler: &H3ImageSampler, seed: u64| {
            let mut model = RefRecorder {
                refs: 3,
                seen: Vec::new(),
            };
            let out = denoise_image(
                &mut model,
                &schedule,
                sampler,
                seed,
                &x0,
                3,
                &d,
                &CancelFlag::default(),
                &mut |_| {},
            )
            .unwrap();
            let want = x0.narrow(1, 0, 3).unwrap().flatten_all().unwrap();
            let want = want.to_vec1::<f32>().unwrap();
            assert_eq!(model.seen.len(), 6);
            assert!(
                model.seen.iter().all(|r| *r == want),
                "reference rows moved"
            );
            assert_eq!(out.dims(), &[1, 2, 96]);
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let a = run(&lcm, 11);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, run(&lcm, 11));
        assert_ne!(a, run(&lcm, 12));
        assert_ne!(a, run(&H3ImageSampler::Euler, 11));
    }

    /// The reference rows ride through the loop unchanged and only the generated rows are returned.
    #[test]
    fn the_loop_holds_reference_rows_and_returns_generated_rows() {
        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image_distilled().unwrap();
        let x0 = Tensor::zeros((1, 5, 96), DType::F32, &d).unwrap();
        let mut model = Recorder { steps: Vec::new() };
        let out = denoise_image(
            &mut model,
            &schedule,
            &H3ImageSampler::Euler,
            0,
            &x0,
            3,
            &d,
            &CancelFlag::default(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(out.dims(), &[1, 2, 96]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-6));
        assert!(denoise_image(
            &mut model,
            &schedule,
            &H3ImageSampler::Euler,
            0,
            &x0,
            5,
            &d,
            &CancelFlag::default(),
            &mut |_| {}
        )
        .is_err());
    }

    /// Reference rows: `0.999 · x + 0.001 · n`, one latent frame each, concatenated in order; the
    /// noise streams are distinct and deterministic.
    #[test]
    fn reference_condition_rows_noise_then_pack_in_order() {
        let d = Device::Cpu;
        let a = Tensor::ones((1, 24, 2, 4), DType::F32, &d).unwrap();
        let b = (Tensor::ones((1, 24, 4, 2), DType::F32, &d).unwrap() * 2.0).unwrap();
        let za = Tensor::zeros((1, 24, 2, 4), DType::F32, &d).unwrap();
        let zb = (Tensor::ones((1, 24, 4, 2), DType::F32, &d).unwrap() * 1000.0).unwrap();
        let rows = reference_condition_rows(&[a, b], &[za, zb], [1, 2, 2]).unwrap();
        assert_eq!(rows.dims(), &[1, 4, 96]);
        let v = rows.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 0.999).abs() < 1e-6);
        assert!((v[2 * 96] - (2.0 * 0.999 + 1000.0 * (1.0 - 0.999f32))).abs() < 1e-3);
        let draw = |seed, stream, i| {
            reference_noise(seed, stream, i, &[1, 4], &d)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(draw(7, 0, 0), draw(7, 0, 0));
        assert_ne!(draw(7, 0, 0), draw(7, 1, 0));
        assert_ne!(draw(7, 0, 0), draw(7, 0, 1));
    }

    /// The VAE copy is cropped at the top left to whole 16 px cells before normalization.
    #[test]
    fn reference_vae_pixels_crop_to_whole_cells() {
        let img = Image {
            width: 40,
            height: 35,
            pixels: (0..40 * 35 * 3).map(|v| (v % 251) as u8).collect(),
        };
        let px = reference_vae_pixels(&img, &Device::Cpu).unwrap();
        assert_eq!(px.dims(), &[1, 3, 32, 32]);
        let want = (f32::from(img.pixels[(40 + 2) * 3]) / 255.0 - PIXEL_MEAN[0]) / PIXEL_STD[0];
        let got = px.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((got[32 + 2] - want).abs() < 1e-6);
        let same = resize_reference(&img, (40, 35)).unwrap();
        assert_eq!(same.pixels, img.pixels);
        assert_eq!(
            resize_reference(&img, (20, 18)).unwrap().pixels.len(),
            20 * 18 * 3
        );
    }

    #[derive(Clone, Debug, PartialEq)]
    struct PagCall {
        step: usize,
        perturbed: Option<Vec<u32>>,
        reference_rows: Vec<f32>,
    }

    struct PagRecorder {
        refs: usize,
        cond_val: f32,
        pert_val: f32,
        calls: Vec<PagCall>,
    }

    impl ImageVelocity for PagRecorder {
        fn forward(&mut self, step: usize, rows: &Tensor) -> Result<Tensor> {
            let reference_rows = if self.refs == 0 {
                Vec::new()
            } else {
                rows.narrow(1, 0, self.refs)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            };
            self.calls.push(PagCall {
                step,
                perturbed: None,
                reference_rows,
            });
            Ok(Tensor::full(
                self.cond_val + step as f32 * 0.25,
                rows.dims(),
                rows.device(),
            )?)
        }

        fn forward_perturbed(
            &mut self,
            step: usize,
            rows: &Tensor,
            perturbed_layers: &[u32],
        ) -> Result<Tensor> {
            let reference_rows = if self.refs == 0 {
                Vec::new()
            } else {
                rows.narrow(1, 0, self.refs)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            };
            self.calls.push(PagCall {
                step,
                perturbed: Some(perturbed_layers.to_vec()),
                reference_rows,
            });
            Ok(Tensor::full(
                self.pert_val + step as f32 * 0.25,
                rows.dims(),
                rows.device(),
            )?)
        }
    }

    #[test]
    fn denoise_image_with_pag_runs_two_passes_on_active_steps_with_same_head_and_single_lcm_draw() {
        use candle_gen::gen_core::generator::{ExecutionIdentity, ExecutionRandomness};
        use candle_gen::gen_core::h3_image::minimax_h3_image_pag_capability;
        use candle_gen::gen_core::{GenerationOutput, PagConfig};

        let d = Device::Cpu;
        let schedule = SigmaSchedule::h3_image_distilled().unwrap();
        let cap = minimax_h3_image_pag_capability();

        // 1. scale == 0.0 runs 1 pass/step (6 total, 0 perturbed) identically to pag == None.
        let zero_pag = cap
            .resolve(Some(&PagConfig {
                scale: 0.0,
                layers: Some(vec![25]),
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(zero_pag.active_steps(6), 0);
        let x0_t2i = initial_image_rows(4, 4, [1, 2, 2], 42, &d).unwrap();
        let lcm = H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 0.0,
        };
        let mut base_rec = PagRecorder {
            refs: 0,
            cond_val: 1.5,
            pert_val: 0.5,
            calls: Vec::new(),
        };
        let out_none = denoise_image_with_pag(
            &mut base_rec,
            &schedule,
            &lcm,
            42,
            &x0_t2i,
            0,
            &d,
            &CancelFlag::default(),
            None,
            &mut |_| {},
        )
        .unwrap();
        let mut zero_rec = PagRecorder {
            refs: 0,
            cond_val: 1.5,
            pert_val: 0.5,
            calls: Vec::new(),
        };
        let out_zero = denoise_image_with_pag(
            &mut zero_rec,
            &schedule,
            &lcm,
            42,
            &x0_t2i,
            0,
            &d,
            &CancelFlag::default(),
            Some(&zero_pag),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(base_rec.calls.len(), 6);
        assert_eq!(zero_rec.calls, base_rec.calls);
        assert_eq!(
            out_none.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            out_zero.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // 2. Curated H3-Image PAG defaults (`scale = 1.2`, `layers = [25]`, `[0.0, 0.66)`) on a
        // 16-reference edit: steps 0..4 each run `(step, cond)` + `(step, pert, [25])` (same step
        // head `i`!), steps 4..6 run `(step, cond)` only (10 passes total), and all 16 reference
        // rows stay frozen throughout.
        let active_pag = cap
            .resolve(Some(&PagConfig {
                scale: 1.2,
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(active_pag.active_steps(6), 4);
        let edit_layout = ImageLayout::edit(&[1, 0, 1], &[(2, 2); 16], 4, 4, [1, 2, 2]).unwrap();
        assert_eq!(edit_layout.num_reference_rows(), 16);
        assert_eq!(edit_layout.num_conditioning_rows(), 19);
        let x0_edit = Tensor::arange(0f32, 20.0 * 96.0, &d)
            .unwrap()
            .reshape((1, 20, 96))
            .unwrap();
        let expected_ref = x0_edit
            .narrow(1, 0, 16)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut edit_rec = PagRecorder {
            refs: 16,
            cond_val: 2.0,
            pert_val: 1.0,
            calls: Vec::new(),
        };
        let out_edit = denoise_image_with_pag(
            &mut edit_rec,
            &schedule,
            &H3ImageSampler::Euler,
            7,
            &x0_edit,
            16,
            &d,
            &CancelFlag::default(),
            Some(&active_pag),
            &mut |_| {},
        )
        .unwrap();
        let observed_steps: Vec<(usize, Option<Vec<u32>>)> = edit_rec
            .calls
            .iter()
            .map(|c| (c.step, c.perturbed.clone()))
            .collect();
        assert_eq!(
            observed_steps,
            vec![
                (0, None),
                (0, Some(vec![25])),
                (1, None),
                (1, Some(vec![25])),
                (2, None),
                (2, Some(vec![25])),
                (3, None),
                (3, Some(vec![25])),
                (4, None),
                (5, None),
            ]
        );
        assert!(edit_rec
            .calls
            .iter()
            .all(|c| c.reference_rows == expected_ref));
        // Under Euler, step i adds Δσ_i * v_guided_i where (v_cond - v_pert) == 1.0 on every step:
        // guided steps 0..4 add Δσ_i * ((2.0 + 0.25*i) + 1.2 * 1.0), steps 4..6 add Δσ_i * (2.0 + 0.25*i).
        let total_increment: f32 = (0..6)
            .map(|i| {
                let d_sigma = schedule.velocity_delta(i).unwrap();
                let base_v = 2.0 + 0.25 * i as f32;
                let guided_v = if i < 4 { base_v + 1.2 } else { base_v };
                d_sigma * guided_v
            })
            .sum();
        let init_gen = x0_edit
            .narrow(1, 16, 4)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_gen = out_edit.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (got, init) in got_gen.iter().zip(&init_gen) {
            assert!(
                (got - (init + total_increment)).abs() < 1e-4,
                "{got} vs {}",
                init + total_increment
            );
        }

        // 3. LCM RNG draw invariance + ExecutionIdentity capture: when v_pert == v_cond across all
        // 6 steps (`stop = 1.0`, 12 passes), LCM output and recorded RNG draws are bit-identical to
        // `pag = None` (6 passes).
        let full_pag = cap
            .resolve(Some(&PagConfig {
                scale: 2.5,
                layers: Some(vec![0, 25, 49]),
                start: Some(0.0),
                stop: Some(1.0),
                adaptive_scale: None,
            }))
            .unwrap();
        assert_eq!(full_pag.active_steps(6), 6);
        let dummy_out = GenerationOutput::Images(vec![Image {
            width: 64,
            height: 64,
            pixels: vec![0; 64 * 64 * 3],
        }]);
        let run_lcm_attempt = |pag: Option<&candle_gen::gen_core::PagResolved>| {
            let attempt = candle_gen::execution::Attempt::begin();
            let init = initial_image_rows(4, 4, [1, 2, 2], 99, &d).unwrap();
            let mut rec = PagRecorder {
                refs: 0,
                cond_val: 1.25,
                pert_val: 1.25,
                calls: Vec::new(),
            };
            let out = denoise_image_with_pag(
                &mut rec,
                &schedule,
                &lcm,
                99,
                &init,
                0,
                &d,
                &CancelFlag::default(),
                pag,
                &mut |_| {},
            )
            .unwrap();
            let identity = attempt.finish_image(&crate::image_descriptor(), &dummy_out);
            (
                rec.calls.len(),
                out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                identity,
            )
        };
        let (calls_1pass, vals_1pass, id_1pass) = run_lcm_attempt(None);
        let (calls_2pass, vals_2pass, id_2pass) = run_lcm_attempt(Some(&full_pag));
        assert_eq!(calls_1pass, 6);
        assert_eq!(calls_2pass, 12);
        assert_eq!(vals_1pass, vals_2pass);
        let draws_1 = match id_1pass {
            ExecutionIdentity::Resolved { identity } => match identity.randomness {
                ExecutionRandomness::Draws { draws, .. } => draws,
                other => panic!("expected Draws, got {other:?}"),
            },
            other => panic!("expected Resolved, got {other:?}"),
        };
        let draws_2 = match id_2pass {
            ExecutionIdentity::Resolved { identity } => match identity.randomness {
                ExecutionRandomness::Draws { draws, .. } => draws,
                other => panic!("expected Draws, got {other:?}"),
            },
            other => panic!("expected Resolved, got {other:?}"),
        };
        // 1 initial latent draw + 5 non-terminal LCM re-noise draws (step 5 has sigma_next == 0).
        assert_eq!(draws_1.len(), 1 + 5);
        assert_eq!(draws_1, draws_2);
    }
}
