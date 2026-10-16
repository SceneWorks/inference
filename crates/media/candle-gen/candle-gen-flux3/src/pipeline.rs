//! `Pipeline.generate` for the distilled checkpoint (`src/flux3/pipeline.py` +
//! `sampling.denoise_flow_map`), the Candle twin of `mlx_gen_flux3::pipeline`: seeded
//! bf16-valued noise in the packed latent space → the flow-map Euler loop over
//! `timesteps(steps, alpha)` → the autoencoder decode → RGB8.
//!
//! The flow-map schedule, the target-time conditioning and the `dt · velocity` rounding are the
//! frozen source's, not FLUX.2's: the loop is written here against `gen_core::flux3` rather than
//! routed through the shared curated-sampler driver, whose sigma conventions do not express the
//! flow map's `(t, t_ref)` pair.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::flux3::{latent_ids, scaled_image_ids, GenerationParams};
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::{CandleError as Error, Result};
use rand::SeedableRng;

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::{Flux3Dit, SampleTokens};
use crate::nn::to;
use crate::text_encoder::TextConditioning;

/// `torch.randn((h·w, C), generator=seed).to(bfloat16).float()`: the repo's launch-portable
/// seeded CPU normal ([`candle_gen::seed`]), rounded through bf16 and held as f32 (bf16 values,
/// f32 state across the Euler steps), moved to `device`. Not bit-compatible with torch's (or the
/// MLX twin's) RNG; parity is measured with injected noise.
pub fn noise(seed: u64, h: usize, w: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let values = candle_gen::seed::seeded_normal_vec(&mut rng, h * w * channels);
    Ok(Tensor::from_vec(values, (h * w, channels), &Device::Cpu)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?
        .to_device(device)?)
}

/// `denoise_flow_map(dit, packed, schedule)` for one text-to-image sample: from `noise`
/// (`[h·w, C]` f32) over `schedule` (the `steps + 1` points), `x ← x + dt · dit(x, t, t_ref)`
/// with `dt · velocity` multiplied at f32 and its PRODUCT rounded to the DiT's compute dtype
/// before the f32 add, as torch forms upstream's `packed.data["image"] + dt * dit(...)` (a
/// Python float times a bf16 tensor: f32 opmath, bf16 result; the state stays f32 because the
/// noise is `.float()`). `on_step(i)` fires after step `i` (1-based). Cancellation is checked
/// before every network evaluation and surfaces as [`Error::Canceled`] — the partial state is
/// dropped.
pub fn denoise_flow_map(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Tensor> {
    if !dit.is_flow_map() {
        return Err(Error::Unsupported(
            "flux3: denoise_flow_map needs a flow-map DiT (the base checkpoint samples with CFG, \
             sc-26239)"
                .into(),
        ));
    }
    if schedule.len() < 2 {
        return Err(Error::Msg(format!(
            "flux3: a schedule needs at least two points, got {}",
            schedule.len()
        )));
    }
    let compute = dit.compute_dtype();
    let device = dit.device();
    let mut x = to(&noise.to_device(device)?, DType::F32)?;
    let text_tokens = text.tokens.to_device(device)?;
    for step in 0..schedule.len() - 1 {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let t = schedule[step];
        let dt = schedule[step + 1] - schedule[step];
        let target = schedule[step + 1];
        let velocity = dit.forward(&[SampleTokens {
            text: &text_tokens,
            text_ids: &text.ids,
            image: &x,
            image_ids,
            image_cond: None,
            timestep: t,
            target_timestep: Some(target),
        }])?;
        // torch: the Python-float `dt` is NOT rounded to bf16 first; the multiply runs at f32
        // opmath and only the product takes the velocity's dtype.
        let scaled = to(&to(&velocity, DType::F32)?.affine(dt as f64, 0.0)?, compute)?;
        x = to(&x.add(&to(&scaled, DType::F32)?)?, DType::F32)?;
        on_step(step + 1);
    }
    Ok(x)
}

/// Base checkpoint Euler CFG, preserving upstream's [unconditional, conditional] pack and
/// bf16 rounding after EACH velocity arithmetic operation, not just after the Euler product.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: &TextConditioning,
    noise: &Tensor,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Tensor> {
    if dit.is_flow_map() || !guidance.is_finite() || guidance < 0.0 {
        return Err(Error::Unsupported(
            "flux3: CFG needs a base DiT and finite non-negative guidance".into(),
        ));
    }
    if schedule.len() < 2 {
        return Err(Error::Msg("flux3: CFG schedule needs two points".into()));
    }
    let compute = dit.compute_dtype();
    let mut x = noise.to_device(dit.device())?.to_dtype(DType::F32)?;
    for step in 0..schedule.len() - 1 {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let samples = [negative, text].map(|conditioning| SampleTokens {
            text: &conditioning.tokens,
            text_ids: &conditioning.ids,
            image: &x,
            image_ids,
            image_cond: None,
            timestep: schedule[step],
            target_timestep: None,
        });
        let prediction = dit.forward(&samples)?;
        let rows = x.dim(0)?;
        let u = prediction.narrow(0, 0, rows)?;
        let c = prediction.narrow(0, rows, rows)?;
        let difference = to(
            &c.to_dtype(DType::F32)?.sub(&u.to_dtype(DType::F32)?)?,
            compute,
        )?;
        let scaled = to(
            &difference
                .to_dtype(DType::F32)?
                .affine(guidance as f64, 0.0)?,
            compute,
        )?;
        let velocity = to(
            &u.to_dtype(DType::F32)?.add(&scaled.to_dtype(DType::F32)?)?,
            compute,
        )?;
        let dt = schedule[step + 1] - schedule[step];
        let update = to(
            &velocity.to_dtype(DType::F32)?.affine(dt as f64, 0.0)?,
            compute,
        )?;
        x = x.add(&update.to_dtype(DType::F32)?)?;
        on_step(step + 1);
    }
    Ok(x)
}

/// The image ids of a `(h, w)` token grid, scaled for the DiT (`scaled_ids(latent_ids(h, w))`).
pub fn image_ids(h: usize, w: usize, image_rope_scale: f32) -> Vec<[f32; 4]> {
    scaled_image_ids(&latent_ids(h, w), image_rope_scale)
}

/// The packed tokens of one image: noise from `seed` → [`denoise_flow_map`] over the request's
/// schedule. `on_step(i)` fires per Euler step.
pub fn denoise_one(
    dit: &Flux3Dit,
    text: &TextConditioning,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    denoise_one_conditioned(dit, text, None, params, seed, cancel, on_step)
}

/// Render tokens with an independently encoded negative caption for CFG.
pub fn denoise_one_conditioned(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: Option<&TextConditioning>,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    let (h, w) = params.latent_shape()?;
    let channels = dit.config().image_in_channels;
    let ids = image_ids(h, w, dit.config().image_rope_scale as f32);
    let z = noise(seed, h, w, channels, dit.device())?;
    match params.guidance {
        Some(guidance) => denoise_cfg(
            dit,
            text,
            negative.ok_or_else(|| {
                Error::Unsupported(
                    "flux3: base CFG needs independently encoded negative conditioning".into(),
                )
            })?,
            &z,
            &ids,
            &params.schedule()?,
            guidance,
            cancel,
            on_step,
        ),
        None => denoise_flow_map(dit, text, &z, &ids, &params.schedule()?, cancel, on_step),
    }
}

/// Render one image: [`denoise_one`] → decode → RGB8 (the caller reports `Decoding`).
pub fn generate_one(
    dit: &Flux3Dit,
    ae: &Flux3AutoEncoder,
    text: &TextConditioning,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Image> {
    let (h, w) = params.latent_shape()?;
    if ae.latent_channels() != dit.config().image_in_channels {
        return Err(Error::Msg(format!(
            "flux3: the autoencoder packs {} channels per token, the DiT reads {}",
            ae.latent_channels(),
            dit.config().image_in_channels
        )));
    }
    let tokens = denoise_one(dit, text, params, seed, cancel, on_step)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    decode_tokens(ae, &tokens, h, w)
}

/// `to_pil(autoencoder.decode(tokens, ids))`: decode the packed tokens and convert the `[−1, 1]`
/// float image to RGB8 (`clamp(−1, 1)`, `(x + 1) · 127.5`, truncated to u8 — see
/// [`float_image_to_rgb8`]).
pub fn decode_tokens(ae: &Flux3AutoEncoder, tokens: &Tensor, h: usize, w: usize) -> Result<Image> {
    let image = ae.decode(tokens, h, w)?;
    float_image_to_rgb8(&image)
}

/// `to_pil`: `((image.clamp(−1, 1) + 1) · 127.5).to(uint8)` — torch's float→uint8 cast
/// **truncates**, so this is `floor`, not the nearest-even rounding of the shared
/// `candle_gen::round_rgb8`. `image` is `[1, 3, H, W]` f32 (NCHW).
pub fn float_image_to_rgb8(image: &Tensor) -> Result<Image> {
    let dims = image.dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 {
        return Err(Error::Msg(format!(
            "flux3: a decoded image must be [1, 3, H, W], got {dims:?}"
        )));
    }
    let (height, width) = (dims[2] as u32, dims[3] as u32);
    let scaled = image
        .to_dtype(DType::F32)?
        .clamp(-1f32, 1f32)?
        .affine(127.5, 127.5)?
        .squeeze(0)?
        .permute((1, 2, 0))?
        .contiguous()?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let pixels: Vec<u8> = scaled.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    Ok(Image {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The noise holds bf16 values in an f32 tensor and depends only on the seed.
    #[test]
    fn noise_is_bf16_valued_f32_and_seeded() {
        let dev = Device::Cpu;
        let a = noise(7, 2, 3, 4, &dev).unwrap();
        let b = noise(7, 2, 3, 4, &dev).unwrap();
        let c = noise(8, 2, 3, 4, &dev).unwrap();
        assert_eq!(a.dims(), &[6, 4]);
        assert_eq!(a.dtype(), DType::F32);
        let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(av, b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert_ne!(av, c.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        for v in &av {
            // A bf16 value has at most 8 significant bits: the low 16 bits of its f32 are zero.
            assert_eq!(v.to_bits() & 0xffff, 0, "{v} is not bf16-representable");
        }
        assert!(
            av.iter().any(|v| v.fract() != 0.0),
            "a normal draw, not integers"
        );
    }

    /// `to_pil` truncates (torch's float → uint8 cast), it does not round.
    #[test]
    fn rgb8_conversion_truncates_and_clamps() {
        let dev = Device::Cpu;
        let image = Tensor::new(
            &[
                -2.0f32, -1.0, -0.5, 0.0, 0.999, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &dev,
        )
        .unwrap()
        .reshape((1, 3, 2, 2))
        .unwrap();
        let rgb = float_image_to_rgb8(&image).unwrap();
        assert_eq!((rgb.width, rgb.height), (2, 2));
        // Pixel (y, x) channel c = image[c][y][x]; channel 0 of the four pixels:
        // -2 → clamp -1 → 0; -1 → 0; -0.5 → 63.75 → 63; 0 → 127.5 → 127.
        assert_eq!(rgb.pixels[0], 0);
        assert_eq!(rgb.pixels[3], 0);
        assert_eq!(rgb.pixels[6], 63);
        assert_eq!(rgb.pixels[9], 127);
        // channel 1: 0.999 → 254.87 → 254; 1 → 255; 1.5 → 255; 0 → 127.
        assert_eq!(rgb.pixels[1], 254);
        assert_eq!(rgb.pixels[4], 255);
        assert_eq!(rgb.pixels[7], 255);
        assert_eq!(rgb.pixels[10], 127);
        assert!(float_image_to_rgb8(&image.reshape((1, 2, 2, 3)).unwrap()).is_err());
    }
}
