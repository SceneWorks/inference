//! `iris3b.sampling.generate` — prompt conditioning (a batch of prompts) + CFG null → seeded pixel
//! noise → FlowDPM-Solver++ over the backbone → `clamp(−1, 1)` → RGB8, with an optional per-step
//! preview of the predicted clean image. The Candle twin of `mlx_gen_iris::pipeline`.

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::iris::{FlowConfig, GenerationParams};
use candle_gen::gen_core::{CancelFlag, Image, PreviewSink};
use candle_gen::preview::{emit_preview_at, project_latents, PreviewCounter};
use candle_gen::{CandleError as Error, Result};
use rand::SeedableRng;

use crate::dit::{IrisDiT, TextBatch};
use crate::solver::{cfg_combine, sample};
use crate::text_encoder::{IrisTextEncoder, TextConditioning};

/// The encoded prompt batch (and the CFG unconditional when guidance is on).
pub struct Conditioning {
    /// One conditioning per prompt of the batch, in order.
    pub cond: Vec<TextConditioning>,
    /// The negative prompt's conditioning, shared by every row of the batch.
    pub uncond: Option<TextConditioning>,
    /// Caption-overflow warnings raised while encoding (`caption_overflow = warn`), in order.
    pub warnings: Vec<String>,
}

/// Encode the request's text side under its caption-overflow policy. The unconditional is the
/// negative prompt run through the same template — `""` is the training-dropout null — and is only
/// encoded when CFG is on.
pub fn encode(te: &IrisTextEncoder, params: &GenerationParams) -> Result<Conditioning> {
    let mut warnings = Vec::new();
    let mut cond = Vec::with_capacity(params.prompts.len());
    for prompt in &params.prompts {
        let (c, warning) = te.encode_with_policy(prompt, params.caption_overflow)?;
        warnings.extend(warning);
        cond.push(c);
    }
    let uncond = if params.uses_cfg() {
        let (u, warning) =
            te.encode_with_policy(&params.negative_prompt, params.caption_overflow)?;
        warnings.extend(warning);
        Some(u)
    } else {
        None
    };
    Ok(Conditioning {
        cond,
        uncond,
        warnings,
    })
}

/// Move a conditioning produced on one device (the text encoder's) onto `device` (the backbone's).
pub fn to_device(conditioning: Conditioning, device: &Device) -> Result<Conditioning> {
    let move_one = |t: TextConditioning| -> Result<TextConditioning> {
        Ok(TextConditioning {
            states: t.states.to_device(device)?,
            ..t
        })
    };
    Ok(Conditioning {
        cond: conditioning
            .cond
            .into_iter()
            .map(move_one)
            .collect::<Result<_>>()?,
        uncond: conditioning.uncond.map(move_one).transpose()?,
        warnings: conditioning.warnings,
    })
}

/// `torch.randn(1, C, H, W)` stand-in: the repo's launch-portable seeded CPU normal (f32), moved to
/// `device`. Not bit-compatible with torch's (or MLX's) RNG; parity is measured with injected noise.
pub fn noise(
    seed: u64,
    channels: usize,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Ok(candle_gen::seed::seeded_noise_nchw(
        &mut rng,
        channels,
        height as usize,
        width as usize,
        device,
    )?)
}

/// One batch's noise `[B, C, H, W]`: row `i` is [`noise`] of `seeds[i]`, so an image's noise
/// depends only on its own seed, never on its batch neighbours.
pub fn noise_batch(
    seeds: &[u64],
    channels: usize,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let rows = seeds
        .iter()
        .map(|&seed| noise(seed, channels, width, height, device))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(&rows, 0)?)
}

/// Run the solver from `noise` (`[B, C, H, W]` f32, one row per prompt of `conditioning`) and return
/// the clamped samples `[B, C, H, W]`. `on_step(i)` fires after step `i`; when `preview` is active
/// every step also emits the predicted clean image of the batch's first row ([`preview_image`]).
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &IrisDiT,
    flow: &FlowConfig,
    conditioning: &Conditioning,
    noise: &Tensor,
    params: &GenerationParams,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
    preview: &PreviewSink,
) -> Result<Tensor> {
    let plan = params.plan()?;
    let device = dit.device();
    let rows = conditioning.cond.len();
    if rows == 0 || noise.dim(0)? != rows {
        return Err(Error::Msg(format!(
            "iris: noise batch {:?} does not match the {rows} encoded prompts",
            noise.dims()
        )));
    }
    let cond_parts: Vec<&Tensor> = conditioning.cond.iter().map(|c| &c.states).collect();
    let cond_states = Tensor::cat(&cond_parts, 0)?;
    let cond_mask: Vec<Vec<i32>> = conditioning.cond.iter().map(|c| c.mask.clone()).collect();
    let cond_text = TextBatch {
        states: &cond_states,
        mask: &cond_mask,
    };
    // CFG batch is cat([uncond, cond]) with the null expanded over the batch, the mask follows suit
    // (upstream's order).
    let cfg_states;
    let cfg_mask;
    let cfg_text = match &conditioning.uncond {
        Some(uncond) => {
            let mut parts: Vec<&Tensor> = vec![&uncond.states; rows];
            parts.extend(cond_parts.iter().copied());
            cfg_states = Tensor::cat(&parts, 0)?;
            cfg_mask = std::iter::repeat_n(uncond.mask.clone(), rows)
                .chain(cond_mask.iter().cloned())
                .collect::<Vec<_>>();
            Some(TextBatch {
                states: &cfg_states,
                mask: &cfg_mask,
            })
        }
        None => None,
    };
    let num_timesteps = flow.num_train_timesteps;
    let counter = PreviewCounter::with_steps(plan.len());
    let patch = dit.config().patch_size;
    let x = sample(
        noise,
        &plan,
        params.prediction,
        cancel,
        |x, step| {
            let t = step.model_time(num_timesteps);
            match &cfg_text {
                Some(batch) if params.cfg_at(step) => {
                    let xb = Tensor::cat(&[x, x], 0)?;
                    let tb = Tensor::new(vec![t; 2 * rows], device)?;
                    let out = dit.forward(&xb, &tb, batch)?;
                    cfg_combine(&out.i(0..rows)?, &out.i(rows..2 * rows)?, params.cfg_scale)
                }
                _ => dit.forward(x, &Tensor::new(vec![t; rows], device)?, &cond_text),
            }
        },
        |i, x0| {
            on_step(i);
            emit_preview_at(preview, &counter, i - 1, || preview_image(x0, patch));
        },
    )?;
    Ok(x.clamp(-1f32, 1f32)?)
}

/// The exact RGB decode of a pixel-space prediction (`(x + 1)/2`, clamped) as a preview projection.
const PREVIEW_FACTORS: [[f32; 3]; 3] = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]];
const PREVIEW_BIAS: [f32; 3] = [0.5; 3];

/// The per-step preview of a predicted clean image `x0` `[B, 3, H, W]` ([−1, 1] pixels): the first
/// row, average-pooled over each `patch × patch` cell — the backbone's own token grid, so a 1024²
/// render previews at 64×64 — and mapped to RGB8 by the **exact** pixel decode `(x + 1)/2`. Iris
/// denoises in pixel space, so no latent→RGB fit is involved: the frame is the model's current
/// clean-image estimate (the CFG-combined `x0` of `FlowDPMSolver._pred_x0`), downsampled. The same
/// projection as the MLX twin's `preview_image`.
pub fn preview_image(x0: &Tensor, patch: usize) -> Result<Image> {
    let dims = x0.dims();
    if dims.len() != 4
        || dims[1] != 3
        || patch == 0
        || !dims[2].is_multiple_of(patch)
        || !dims[3].is_multiple_of(patch)
    {
        return Err(Error::Msg(format!(
            "iris: preview state {dims:?} is not [B, 3, H, W] on the {patch}-pixel patch grid"
        )));
    }
    let pooled = x0.i(0..1)?.to_dtype(DType::F32)?.avg_pool2d(patch)?;
    project_latents(&pooled, &PREVIEW_FACTORS, PREVIEW_BIAS)
}

/// `[B, 3, H, W]` in `[−1, 1]` → one RGB8 image per row.
pub fn to_images(samples: &Tensor) -> Result<Vec<Image>> {
    (0..samples.dim(0)?)
        .map(|i| to_image(&samples.i(i..i + 1)?))
        .collect()
}

/// `[1, 3, H, W]` in `[−1, 1]` → RGB8 (`round(255 · (x + 1)/2)`, nearest-even ties — the same
/// denormalise/quantise chain as `mlx_gen::image::decoded_to_image`).
pub fn to_image(sample: &Tensor) -> Result<Image> {
    let x = sample.to_dtype(DType::F32)?;
    let unit = x.affine(0.5, 0.5)?.clamp(0f32, 1f32)?;
    let rgb = candle_gen::round_rgb8(&(unit * 255.0)?)?
        .i(0)?
        .to_device(&Device::Cpu)?;
    let (c, h, w) = rgb.dims3()?;
    if c != 3 {
        return Err(Error::Msg(format!(
            "iris: expected a 3-channel sample, got {c}"
        )));
    }
    let pixels = rgb.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preview_is_the_exact_decode_of_the_patch_mean_of_the_first_row() {
        // [2, 3, 8, 8] with values spanning past [−1, 1], patch 4 → a 2×2 frame of row 0 — the MLX
        // twin's case.
        let n = 2 * 3 * 8 * 8;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 97) as f32 / 32.0) - 1.5)
            .collect();
        let x0 = Tensor::from_vec(data.clone(), (2, 3, 8, 8), &Device::Cpu).unwrap();
        let frame = preview_image(&x0, 4).unwrap();
        assert_eq!((frame.width, frame.height), (2, 2));
        for py in 0..2 {
            for px in 0..2 {
                for c in 0..3 {
                    let mut sum = 0f32;
                    for y in 0..4 {
                        for x in 0..4 {
                            sum += data[(c * 8 + py * 4 + y) * 8 + px * 4 + x];
                        }
                    }
                    let unit = ((sum / 16.0 + 1.0) / 2.0).clamp(0.0, 1.0);
                    let want = (unit * 255.0).round_ties_even() as u8;
                    assert_eq!(frame.pixels[(py * 2 + px) * 3 + c], want, "({px},{py},{c})");
                }
            }
        }
        assert!(preview_image(&x0, 3).is_err(), "off the patch grid");
    }
}
