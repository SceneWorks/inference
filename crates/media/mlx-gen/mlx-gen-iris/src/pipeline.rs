//! `iris3b.sampling.generate` — prompt conditioning (a batch of prompts) + CFG null → seeded pixel
//! noise → FlowDPM-Solver++ over the backbone → `clamp(−1, 1)` → RGB8, with an optional per-step
//! preview of the predicted clean image.

use gen_core::iris::{FlowConfig, GenerationParams};
use mlx_gen::gen_core;
use mlx_gen::preview::{emit_preview, project_latents, PreviewCounter};
use mlx_gen::{CancelFlag, Error, Image, PreviewSink, Result};
use mlx_rs::ops::{clip, concatenate_axis};
use mlx_rs::{random, Array};

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

/// `torch.randn(1, C, H, W)` stand-in: the repo's seeded MLX normal (f32). Not bit-compatible with
/// torch's RNG; parity is measured with injected noise.
pub fn noise(seed: u64, channels: usize, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [1, channels as i32, height as i32, width as i32];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// One batch's noise `[B, C, H, W]`: row `i` is [`noise`] of `seeds[i]`, so an image's noise
/// depends only on its own seed, never on its batch neighbours.
pub fn noise_batch(seeds: &[u64], channels: usize, width: u32, height: u32) -> Result<Array> {
    let rows = seeds
        .iter()
        .map(|&seed| noise(seed, channels, width, height))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Array> = rows.iter().collect();
    Ok(concatenate_axis(&refs, 0)?)
}

/// Run the solver from `noise` (`[B, C, H, W]` f32, one row per prompt of `conditioning`) and return
/// the clamped samples `[B, C, H, W]`. `on_step(i)` fires after step `i`; when `preview` is active
/// every step also emits the predicted clean image of the batch's first row ([`preview_image`]).
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &IrisDiT,
    flow: &FlowConfig,
    conditioning: &Conditioning,
    noise: &Array,
    params: &GenerationParams,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
    preview: &PreviewSink,
) -> Result<Array> {
    let plan = params.plan()?;
    let rows = conditioning.cond.len();
    if rows == 0 || noise.shape()[0] as usize != rows {
        return Err(Error::Msg(format!(
            "iris: noise batch {:?} does not match the {rows} encoded prompts",
            noise.shape()
        )));
    }
    let cond_parts: Vec<&Array> = conditioning.cond.iter().map(|c| &c.states).collect();
    let cond_states = concatenate_axis(&cond_parts, 0)?;
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
            let mut parts: Vec<&Array> = vec![&uncond.states; rows];
            parts.extend(cond_parts.iter().copied());
            cfg_states = concatenate_axis(&parts, 0)?;
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
    let sigmas: Vec<f32> = plan
        .iter()
        .map(|step| step.s as f32)
        .chain(std::iter::once(0.0))
        .collect();
    let counter = PreviewCounter::new(&sigmas);
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
                    let xb = concatenate_axis(&[x, x], 0)?;
                    let tb = Array::from_slice(&vec![t; 2 * rows], &[2 * rows as i32]);
                    let out = dit.forward(&xb, &tb, batch)?;
                    let halves = out.split(2, 0)?;
                    cfg_combine(&halves[0], &halves[1], params.cfg_scale)
                }
                _ => dit.forward(
                    x,
                    &Array::from_slice(&vec![t; rows], &[rows as i32]),
                    &cond_text,
                ),
            }
        },
        |i, x0| {
            on_step(i);
            emit_preview(preview, &counter, &sigmas, sigmas[i - 1], || {
                preview_image(x0, patch)
            });
        },
    )?;
    Ok(clip(&x, (-1.0f32, 1.0f32))?)
}

/// The exact RGB decode of a pixel-space prediction (`(x + 1)/2`, clamped) as a preview projection.
const PREVIEW_FACTORS: [[f32; 3]; 3] = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]];
const PREVIEW_BIAS: [f32; 3] = [0.5; 3];

/// The per-step preview of a predicted clean image `x0` `[B, 3, H, W]` ([−1, 1] pixels): the first
/// row, average-pooled over each `patch × patch` cell — the backbone's own token grid, so a 1024²
/// render previews at 64×64 — and mapped to RGB8 by the **exact** pixel decode `(x + 1)/2`. Iris
/// denoises in pixel space, so no latent→RGB fit is involved: the frame is the model's current
/// clean-image estimate (the CFG-combined `x0` of `FlowDPMSolver._pred_x0`), downsampled.
pub fn preview_image(x0: &Array, patch: usize) -> Result<Image> {
    let sh = x0.shape();
    let p = patch as i32;
    if sh.len() != 4 || sh[1] != 3 || p == 0 || sh[2] % p != 0 || sh[3] % p != 0 {
        return Err(Error::Msg(format!(
            "iris: preview state {sh:?} is not [B, 3, H, W] on the {patch}-pixel patch grid"
        )));
    }
    let first = if sh[0] > 1 {
        x0.split(sh[0], 0)?.swap_remove(0)
    } else {
        x0.clone()
    };
    let (hp, wp) = (sh[2] / p, sh[3] / p);
    let pooled = first
        .as_dtype(mlx_rs::Dtype::Float32)?
        .reshape(&[1, 3, hp, p, wp, p])?
        .mean_axes(&[3, 5], None)?;
    project_latents(&pooled, &PREVIEW_FACTORS, PREVIEW_BIAS)
}

/// `[B, 3, H, W]` in `[−1, 1]` → one RGB8 image per row (`(x + 1)/2 · 255`, rounded — torchvision
/// `save_image` with `normalize=True, value_range=(−1, 1)`).
pub fn to_images(samples: &Array) -> Result<Vec<Image>> {
    let rows = samples.shape()[0];
    if rows == 1 {
        return Ok(vec![to_image(samples)?]);
    }
    samples.split(rows, 0)?.iter().map(to_image).collect()
}

/// `[1, 3, H, W]` in `[−1, 1]` → RGB8.
pub fn to_image(sample: &Array) -> Result<Image> {
    mlx_gen::image::decoded_to_image(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preview_is_the_exact_decode_of_the_patch_mean_of_the_first_row() {
        // [2, 3, 8, 8] with values spanning past [−1, 1], patch 4 → a 2×2 frame of row 0.
        let n = 2 * 3 * 8 * 8;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * 37 % 97) as f32 / 32.0) - 1.5)
            .collect();
        let x0 = Array::from_slice(&data, &[2, 3, 8, 8]);
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
