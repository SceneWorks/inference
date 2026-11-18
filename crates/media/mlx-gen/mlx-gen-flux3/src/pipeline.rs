//! `Pipeline.generate` for the distilled checkpoint (`src/flux3/pipeline.py` +
//! `sampling.denoise_flow_map`): seeded bf16-valued noise in the packed latent space → the
//! flow-map Euler loop over `timesteps(steps, alpha)` → the autoencoder decode → RGB8.

use gen_core_flux3::{latent_ids, scaled_image_ids, GenerationParams};
use mlx_gen::{CancelFlag, Error, Image, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::{Flux3Dit, SampleTokens};
use crate::references::ReferenceConditioning;
use crate::text_encoder::TextConditioning;

/// `torch.randn((h·w, C), generator=seed).to(bfloat16).float()`: the repo's seeded MLX normal,
/// rounded through bf16 and held as f32 (bf16 values, f32 state across the Euler steps). Not
/// bit-compatible with torch's RNG; parity is measured with injected noise.
pub fn noise(seed: u64, h: usize, w: usize, channels: usize) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [(h * w) as i32, channels as i32];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?
        .as_dtype(Dtype::Bfloat16)?
        .as_dtype(Dtype::Float32)?)
}

/// `denoise_flow_map(dit, packed, schedule)` for one text-to-image sample: from `noise`
/// (`[h·w, C]` f32) over `schedule` (the `steps + 1` points), `x ← x + dt · dit(x, t, t_ref)`
/// with `dt · velocity` multiplied at f32 and its PRODUCT rounded to the DiT's compute dtype
/// before the f32 add, as torch forms upstream's `packed.data["image"] + dt * dit(...)` (a
/// Python float times a bf16 tensor: f32 opmath, bf16 result; the state stays f32 because the
/// noise is `.float()`). `on_step(i)` fires after step `i` (1-based). Cancellation is checked
/// before every network evaluation and surfaces as [`Error::Canceled`] — the partial state is
/// dropped.
#[allow(clippy::too_many_arguments)]
pub fn denoise_flow_map(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    denoise_flow_map_with_references(dit, text, noise, image_ids, schedule, None, cancel, on_step)
}

fn apply_pag_mlx(
    base: &Array,
    cond: &Array,
    pert: &Array,
    step_pag: f32,
    compute: Dtype,
) -> Result<Array> {
    let diff = cond
        .as_dtype(Dtype::Float32)?
        .subtract(&pert.as_dtype(Dtype::Float32)?)?
        .as_dtype(compute)?;
    let scaled = diff
        .as_dtype(Dtype::Float32)?
        .multiply(Array::from_f32(step_pag))?
        .as_dtype(compute)?;
    Ok(base
        .as_dtype(Dtype::Float32)?
        .add(&scaled.as_dtype(Dtype::Float32)?)?
        .as_dtype(compute)?)
}

#[allow(clippy::too_many_arguments)]
fn denoise_flow_map_with_references_pag(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    references: Option<&ReferenceConditioning>,
    pag: Option<&mlx_gen::gen_core::PagResolved>,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Array> {
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
    let total_steps = schedule.len() - 1;
    let mut x = noise.as_dtype(Dtype::Float32)?;
    for step in 0..total_steps {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let t = schedule[step];
        let dt = schedule[step + 1] - schedule[step];
        let target = schedule[step + 1];
        let sample = SampleTokens {
            text: &text.tokens,
            text_ids: &text.ids,
            image: &x,
            image_ids,
            image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
            timestep: t,
            target_timestep: Some(target),
        };
        let cond = dit.forward_pag(std::slice::from_ref(&sample), None)?;
        let step_pag = pag.map_or(0.0, |p| p.step_scale(step, total_steps));
        let velocity = if step_pag > 0.0 {
            eval([&cond])?;
            let layers = pag.map_or(&[][..], |p| p.step_layers(step, total_steps));
            let pert = dit.forward_pag(std::slice::from_ref(&sample), Some(layers))?;
            apply_pag_mlx(&cond, &cond, &pert, step_pag, compute)?
        } else {
            cond
        };
        // torch: the Python-float `dt` is NOT rounded to bf16 first; the multiply runs at f32
        // opmath and only the product takes the velocity's dtype.
        let scaled = velocity
            .as_dtype(Dtype::Float32)?
            .multiply(Array::from_f32(dt))?
            .as_dtype(compute)?;
        x = x.add(&scaled)?.as_dtype(Dtype::Float32)?;
        eval([&x])?;
        on_step(step + 1);
    }
    Ok(x)
}

/// Same frozen integrator with immutable AE conditioning shared by both CFG branches.
#[allow(clippy::too_many_arguments)]
pub fn denoise_flow_map_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    references: Option<&ReferenceConditioning>,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    denoise_flow_map_with_references_pag(
        dit, text, noise, image_ids, schedule, references, None, cancel, on_step,
    )
}

/// Base Euler CFG with independent unconditional/conditional samples in that order.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    denoise_cfg_with_references(
        dit, text, negative, noise, image_ids, schedule, guidance, None, cancel, on_step,
    )
}

/// Same frozen integrator with immutable AE conditioning shared by both CFG branches.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: &TextConditioning,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    references: Option<&ReferenceConditioning>,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    denoise_base_with_references_pag(
        dit,
        text,
        Some(negative),
        noise,
        image_ids,
        schedule,
        guidance,
        references,
        None,
        cancel,
        on_step,
    )
}

#[allow(clippy::too_many_arguments)]
fn denoise_base_with_references_pag(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: Option<&TextConditioning>,
    noise: &Array,
    image_ids: &[[f32; 4]],
    schedule: &[f32],
    guidance: f32,
    references: Option<&ReferenceConditioning>,
    pag: Option<&mlx_gen::gen_core::PagResolved>,
    cancel: &CancelFlag,
    mut on_step: impl FnMut(usize),
) -> Result<Array> {
    if dit.is_flow_map() || !guidance.is_finite() || guidance < 0.0 {
        return Err(Error::Unsupported(
            "flux3: CFG needs a base DiT and finite non-negative guidance".into(),
        ));
    }
    if schedule.len() < 2 {
        return Err(Error::Msg("flux3: CFG schedule needs two points".into()));
    }
    let compute = dit.compute_dtype();
    let total_steps = schedule.len() - 1;
    let cfg_off_with_pag = pag.is_some() && guidance <= 1.0;
    let mut x = noise.as_dtype(Dtype::Float32)?;
    for step in 0..total_steps {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let step_pag = pag.map_or(0.0, |p| p.step_scale(step, total_steps));
        let layers = pag.map_or(&[][..], |p| p.step_layers(step, total_steps));
        let velocity = if cfg_off_with_pag {
            let pos_sample = SampleTokens {
                text: &text.tokens,
                text_ids: &text.ids,
                image: &x,
                image_ids,
                image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                timestep: schedule[step],
                target_timestep: None,
            };
            let cond = dit.forward_pag(std::slice::from_ref(&pos_sample), None)?;
            if step_pag > 0.0 {
                eval([&cond])?;
                let pert = dit.forward_pag(std::slice::from_ref(&pos_sample), Some(layers))?;
                apply_pag_mlx(&cond, &cond, &pert, step_pag, compute)?
            } else {
                cond
            }
        } else {
            let negative = negative.ok_or_else(|| {
                Error::Unsupported(
                    "flux3: base CFG needs independently encoded negative conditioning".into(),
                )
            })?;
            let samples = [negative, text].map(|conditioning| SampleTokens {
                text: &conditioning.tokens,
                text_ids: &conditioning.ids,
                image: &x,
                image_ids,
                image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                timestep: schedule[step],
                target_timestep: None,
            });
            let prediction = dit.forward(&samples)?;
            let rows = prediction.split(2, 0)?;
            let difference = rows[1].subtract(&rows[0])?.as_dtype(compute)?;
            let scaled = difference
                .as_dtype(Dtype::Float32)?
                .multiply(Array::from_f32(guidance))?
                .as_dtype(compute)?;
            let pred_cfg = rows[0].add(&scaled)?.as_dtype(compute)?;
            if step_pag > 0.0 {
                let cond = rows[1].clone();
                eval([&pred_cfg, &cond])?;
                let pos_sample = SampleTokens {
                    text: &text.tokens,
                    text_ids: &text.ids,
                    image: &x,
                    image_ids,
                    image_cond: references.map(|r| (&r.tokens, r.ids.as_slice())),
                    timestep: schedule[step],
                    target_timestep: None,
                };
                let pert = dit.forward_pag(std::slice::from_ref(&pos_sample), Some(layers))?;
                apply_pag_mlx(&pred_cfg, &cond, &pert, step_pag, compute)?
            } else {
                pred_cfg
            }
        };
        let dt = schedule[step + 1] - schedule[step];
        let update = velocity
            .as_dtype(Dtype::Float32)?
            .multiply(Array::from_f32(dt))?
            .as_dtype(compute)?;
        x = x.add(&update)?.as_dtype(Dtype::Float32)?;
        eval([&x])?;
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
) -> Result<Array> {
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
) -> Result<Array> {
    denoise_one_with_references(dit, text, negative, None, params, seed, cancel, on_step)
}

/// Generate fresh noise, while reference tokens remain pinned image_cond (not img2img).
#[allow(clippy::too_many_arguments)]
pub fn denoise_one_with_references(
    dit: &Flux3Dit,
    text: &TextConditioning,
    negative: Option<&TextConditioning>,
    references: Option<&ReferenceConditioning>,
    params: &GenerationParams,
    seed: u64,
    cancel: &CancelFlag,
    on_step: impl FnMut(usize),
) -> Result<Array> {
    let (h, w) = params.latent_shape()?;
    let channels = dit.config().image_in_channels;
    let ids = image_ids(h, w, dit.config().image_rope_scale as f32);
    let z = noise(seed, h, w, channels)?;
    let schedule = params.schedule()?;
    let output = match params.guidance {
        Some(guidance) => denoise_base_with_references_pag(
            dit,
            text,
            negative,
            &z,
            &ids,
            &schedule,
            guidance,
            references,
            params.pag.as_ref(),
            cancel,
            on_step,
        ),
        None => denoise_flow_map_with_references_pag(
            dit,
            text,
            &z,
            &ids,
            &schedule,
            references,
            params.pag.as_ref(),
            cancel,
            on_step,
        ),
    }?;
    mlx_gen::image_execution::sampled_native_with_rng(
        if params.guidance.is_some() {
            "flux3-euler-cfg-v1"
        } else {
            "flux3-flow-map-v1"
        },
        "native-timestep-and-target",
        output.dtype(),
        &schedule,
        seed,
        "mlx-keyed-normal-f32-v1;initial=output-seed;no-step-draws",
    );
    mlx_gen::image_execution::policy_debug("flux3.compute_dtype", &dit.compute_dtype());
    mlx_gen::image_execution::policy_debug("flux3.reference_tokens", &references.is_some());
    if let Some(guidance) = params.guidance {
        mlx_gen::image_execution::scalar("flux3.guidance", guidance);
    }
    Ok(output)
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
    if ae.latent_channels() as usize != dit.config().image_in_channels {
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
pub fn decode_tokens(ae: &Flux3AutoEncoder, tokens: &Array, h: usize, w: usize) -> Result<Image> {
    let image = ae.decode(tokens, h as i32, w as i32)?;
    float_image_to_rgb8(&image)
}

/// `to_pil`: `((image.clamp(−1, 1) + 1) · 127.5).to(uint8)` — torch's float→uint8 cast
/// **truncates**, so this is `floor`, not the round-half-up of the shared
/// `decoded_to_image`. `image` is `[1, H, W, 3]` f32 (NHWC).
pub fn float_image_to_rgb8(image: &Array) -> Result<Image> {
    let sh = image.shape();
    if sh.len() != 4 || sh[0] != 1 || sh[3] != 3 {
        return Err(Error::Msg(format!(
            "flux3: a decoded image must be [1, H, W, 3], got {sh:?}"
        )));
    }
    let (height, width) = (sh[1] as u32, sh[2] as u32);
    let scaled = mlx_rs::ops::clip(&image.as_dtype(Dtype::Float32)?, (-1.0f32, 1.0f32))?
        .add(Array::from_f32(1.0))?
        .multiply(Array::from_f32(127.5))?;
    // A decoded/transposed NHWC view may still have planar (NCHW) physical storage. Elementwise
    // arithmetic need not change its strides; materialize logical C-order before host readback,
    // just like mlx-gen's shared decoded_to_interleaved_u8 boundary.
    let flat = scaled.reshape(&[-1])?;
    eval([&flat])?;
    let values = flat.as_slice::<f32>();
    let pixels: Vec<u8> = values.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    Ok(Image {
        width,
        height,
        pixels,
    })
}
