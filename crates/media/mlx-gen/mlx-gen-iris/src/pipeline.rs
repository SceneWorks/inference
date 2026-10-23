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
    encode_in(te, params, None, &CancelFlag::default())
}

/// [`encode`] with a streamed tower's layer window (rung 4) and the request's cancel flag.
pub fn encode_in(
    te: &IrisTextEncoder,
    params: &GenerationParams,
    layer_window: Option<usize>,
    cancel: &CancelFlag,
) -> Result<Conditioning> {
    let mut warnings = Vec::new();
    let mut cond = Vec::with_capacity(params.prompts.len());
    for prompt in &params.prompts {
        let (c, warning) =
            te.encode_with_policy_in(prompt, params.caption_overflow, layer_window, cancel)?;
        warnings.extend(warning);
        cond.push(c);
    }
    let uncond = if params.uses_cfg() {
        let (u, warning) = te.encode_with_policy_in(
            &params.negative_prompt,
            params.caption_overflow,
            layer_window,
            cancel,
        )?;
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
    on_step: impl FnMut(usize),
    preview: &PreviewSink,
) -> Result<Array> {
    denoise_budgeted(
        dit,
        flow,
        conditioning,
        noise,
        params,
        cancel,
        on_step,
        preview,
        mlx_gen::AttentionBudget::UNBOUNDED,
    )
}

/// Solver with a bounded-attention forward, leaving the legacy/training API unchanged.
#[allow(clippy::too_many_arguments)]
pub fn denoise_budgeted(
    dit: &IrisDiT,
    flow: &FlowConfig,
    conditioning: &Conditioning,
    noise: &Array,
    params: &GenerationParams,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
    preview: &PreviewSink,
    budget: mlx_gen::AttentionBudget,
) -> Result<Array> {
    denoise_in(
        dit,
        flow,
        conditioning,
        noise,
        params,
        cancel,
        on_step,
        preview,
        budget,
        None,
        false,
    )
}

/// [`denoise_budgeted`] with rung 4's backbone window (`None`: a streamed backbone's whole-stack
/// window; a resident backbone ignores it) and rung 3's per-row pixel stage
/// ([`crate::dit::Execution`]).
#[allow(clippy::too_many_arguments)]
pub fn denoise_in(
    dit: &IrisDiT,
    flow: &FlowConfig,
    conditioning: &Conditioning,
    noise: &Array,
    params: &GenerationParams,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
    preview: &PreviewSink,
    budget: mlx_gen::AttentionBudget,
    window: Option<usize>,
    pixel_rows: bool,
) -> Result<Array> {
    let plan_metadata = params.plan_metadata()?;
    mlx_gen::ios_quantum::before(crate::solver::PLAN_QUANTUM, plan_metadata.host_bytes()?)?;
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
    mlx_gen::ios_quantum::finish(crate::solver::PLAN_QUANTUM, &[])?;
    let patch = dit.config().patch_size;
    let mut evaluations = 0u64;
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
                    evaluations += 1;
                    mlx_gen::ios_quantum::diagnostic_metadata(
                        crate::model::MODEL_ID,
                        "forward",
                        &[
                            ("evaluation", evaluations),
                            ("rows", xb.shape()[0] as u64),
                            ("conditioning_rows", batch.states.shape()[0] as u64),
                            ("guided", 1),
                            ("input_f32", u64::from(xb.dtype() == mlx_rs::Dtype::Float32)),
                        ],
                    );
                    let out = dit.forward_exec(
                        &xb,
                        &tb,
                        batch,
                        crate::dit::Execution {
                            plan: mlx_gen::AttentionPlan::budgeted(budget).with_cancel(cancel),
                            window,
                            pixel_rows,
                        },
                    )?;
                    let halves = out.split(2, 0)?;
                    cfg_combine(&halves[0], &halves[1], params.cfg_scale)
                }
                _ => {
                    evaluations += 1;
                    mlx_gen::ios_quantum::diagnostic_metadata(
                        crate::model::MODEL_ID,
                        "forward",
                        &[
                            ("evaluation", evaluations),
                            ("rows", x.shape()[0] as u64),
                            ("conditioning_rows", cond_text.states.shape()[0] as u64),
                            ("guided", 0),
                            ("input_f32", u64::from(x.dtype() == mlx_rs::Dtype::Float32)),
                        ],
                    );
                    dit.forward_exec(
                        x,
                        &Array::from_slice(&vec![t; rows], &[rows as i32]),
                        &cond_text,
                        crate::dit::Execution {
                            plan: mlx_gen::AttentionPlan::budgeted(budget).with_cancel(cancel),
                            window,
                            pixel_rows,
                        },
                    )
                }
            }
        },
        |i, x0| {
            on_step(i);
            emit_preview(preview, &counter, &sigmas, sigmas[i - 1], || {
                preview_image(x0, patch)
            });
        },
    )?;
    // Do not leave the final clamp lazy across the denoise/decode boundary: it allocates while
    // this solve still holds its text batches, and belongs to the denoise's live set.
    let solved = mlx_gen::ios_quantum::run(
        crate::solver::FINALIZE_QUANTUM,
        crate::solver::final_clamp_scratch(x.size() as u64),
        || Ok(clip(&x, (-1.0f32, 1.0f32))?),
    )?;
    mlx_gen::ios_quantum::diagnostic_metadata(
        crate::model::MODEL_ID,
        "solve_complete",
        &[("evaluations", evaluations), ("steps", plan.len() as u64)],
    );
    Ok(solved)
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
