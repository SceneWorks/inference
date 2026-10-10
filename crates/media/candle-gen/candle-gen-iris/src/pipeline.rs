//! `iris3b.sampling.generate` — prompt conditioning + CFG null → seeded pixel noise → FlowDPM-Solver++
//! over the backbone → `clamp(−1, 1)` → RGB8. The Candle twin of `mlx_gen_iris::pipeline`.

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::iris::{
    cfg_active, dpm_solver_plan, FlowConfig, GenerationParams, DEFAULT_CFG_INTERVAL,
    DEFAULT_SOLVER_ORDER,
};
use candle_gen::gen_core::{CancelFlag, Image};
use candle_gen::{CandleError as Error, Result};
use rand::SeedableRng;

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

/// Move a conditioning produced on one device (the text encoder's) onto `device` (the backbone's).
pub fn to_device(conditioning: Conditioning, device: &Device) -> Result<Conditioning> {
    let move_one = |t: TextConditioning| -> Result<TextConditioning> {
        Ok(TextConditioning {
            states: t.states.to_device(device)?,
            ..t
        })
    };
    Ok(Conditioning {
        cond: move_one(conditioning.cond)?,
        uncond: conditioning.uncond.map(move_one).transpose()?,
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

/// Run the solver from `noise` (`[1, C, H, W]` f32) and return the clamped sample `[1, C, H, W]`.
pub fn denoise(
    dit: &IrisDiT,
    flow: &FlowConfig,
    conditioning: &Conditioning,
    noise: &Tensor,
    params: &GenerationParams,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Tensor> {
    let plan = dpm_solver_plan(params.steps, DEFAULT_SOLVER_ORDER, flow.shift)?;
    let device = dit.device();
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
            cfg_states = Tensor::cat(&[&uncond.states, &conditioning.cond.states], 0)?;
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
                    let xb = Tensor::cat(&[x, x], 0)?;
                    let tb = Tensor::new(&[t, t], device)?;
                    let out = dit.forward(&xb, &tb, batch)?;
                    cfg_combine(&out.i(0..1)?, &out.i(1..2)?, params.cfg_scale)
                }
                _ => dit.forward(x, &Tensor::new(&[t], device)?, &cond_text),
            }
        },
        on_step,
    )?;
    Ok(x.clamp(-1f32, 1f32)?)
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
