//! `iris3b.sampling.generate` — prompt conditioning + CFG null → seeded pixel noise → FlowDPM-Solver++
//! over the backbone → `clamp(−1, 1)` → RGB8.

use gen_core::iris::{
    cfg_active, dpm_solver_plan, FlowConfig, GenerationParams, DEFAULT_CFG_INTERVAL,
    DEFAULT_SOLVER_ORDER,
};
use mlx_gen::gen_core;
use mlx_gen::{CancelFlag, Image, Result};
use mlx_rs::ops::{clip, concatenate_axis};
use mlx_rs::{random, Array};

use crate::dit::{IrisDiT, TextBatch};
use crate::solver::{cfg_combine, sample};
use crate::text_encoder::{IrisTextEncoder, TextConditioning};

/// The encoded prompt (and the CFG unconditional when guidance is on).
pub struct Conditioning {
    pub cond: TextConditioning,
    pub uncond: Option<TextConditioning>,
}

/// Encode the request's text side. The unconditional is the negative prompt run through the same
/// template — `""` is the training-dropout null — and is only encoded when CFG is on.
pub fn encode(
    te: &IrisTextEncoder,
    prompt: &str,
    params: &GenerationParams,
) -> Result<Conditioning> {
    let cond = te.encode(prompt)?;
    let uncond = if params.uses_cfg() {
        Some(te.encode(&params.negative_prompt)?)
    } else {
        None
    };
    Ok(Conditioning { cond, uncond })
}

/// `torch.randn(1, C, H, W)` stand-in: the repo's seeded MLX normal (f32). Not bit-compatible with
/// torch's RNG; parity is measured with injected noise.
pub fn noise(seed: u64, channels: usize, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [1, channels as i32, height as i32, width as i32];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Run the solver from `noise` (`[1, C, H, W]` f32) and return the clamped sample `[1, C, H, W]`.
pub fn denoise(
    dit: &IrisDiT,
    flow: &FlowConfig,
    conditioning: &Conditioning,
    noise: &Array,
    params: &GenerationParams,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    let plan = dpm_solver_plan(params.steps, DEFAULT_SOLVER_ORDER, flow.shift)?;
    let cond_mask = vec![conditioning.cond.mask.clone()];
    let cond_text = TextBatch {
        states: &conditioning.cond.states,
        mask: &cond_mask,
    };
    // CFG batch is cat([uncond, cond]), the mask follows suit (upstream's order).
    let cfg_states;
    let cfg_mask;
    let cfg_text = match &conditioning.uncond {
        Some(uncond) => {
            cfg_states = concatenate_axis(&[&uncond.states, &conditioning.cond.states], 0)?;
            cfg_mask = vec![uncond.mask.clone(), conditioning.cond.mask.clone()];
            Some(TextBatch {
                states: &cfg_states,
                mask: &cfg_mask,
            })
        }
        None => None,
    };
    let num_timesteps = flow.num_train_timesteps;
    let x = sample(
        noise,
        &plan,
        cancel,
        |x, step| {
            let t = step.model_time(num_timesteps);
            match &cfg_text {
                Some(batch) if cfg_active(params.cfg_scale, step.s, DEFAULT_CFG_INTERVAL) => {
                    let xb = concatenate_axis(&[x, x], 0)?;
                    let tb = Array::from_slice(&[t, t], &[2]);
                    let out = dit.forward(&xb, &tb, batch)?;
                    let halves = out.split(2, 0)?;
                    cfg_combine(&halves[0], &halves[1], params.cfg_scale)
                }
                _ => dit.forward(x, &Array::from_slice(&[t], &[1]), &cond_text),
            }
        },
        on_step,
    )?;
    Ok(clip(&x, (-1.0f32, 1.0f32))?)
}

/// `[1, 3, H, W]` in `[−1, 1]` → RGB8 (`(x + 1)/2 · 255`, rounded — torchvision `save_image` with
/// `normalize=True, value_range=(−1, 1)`).
pub fn to_image(sample: &Array) -> Result<Image> {
    mlx_gen::image::decoded_to_image(sample)
}
