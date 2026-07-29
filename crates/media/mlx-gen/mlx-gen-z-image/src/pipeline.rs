//! Z-Image sampling pipeline: seeded latent creation, latent unpacking, the decoded-tensor →
//! [`Image`] conversion (ports of the fork's `ZImageLatentCreator` + `ImageUtil`), img2img
//! init-latent encoding, and the denoise loops ([`denoise_with_progress`] /
//! [`denoise_control_with_progress`]) that compose these with the transformer
//! ([`crate::transformer`]), scheduler ([`mlx_gen::FlowMatchEuler`]) and VAE ([`crate::vae`]). The
//! four generators drive them through the staged `denoise_batch` → `decode_batch` split (sc-13571):
//! [`mlx_gen::Residency::run_staged`] frees the DiT between the two phases so the tiled decode fits a
//! small Mac.

use mlx_gen::array::host_i32;
use mlx_gen::attention::{AttentionBudget, AttentionPlan};
// The img2img leaves (start-step / init-image preprocess / noise-interp blend) are shared in core;
// re-export so the crate's public surface (`mlx_gen_z_image::…`) and internal callers are unchanged.
use mlx_gen::gen_core::TransformerComponent;
pub use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::tokenizer::{TextTokenizer, TokenizerOutput};
use mlx_gen::{
    run_flow_sampler, CancelFlag, Conditioning, Error, FlowMatchEuler, GenerationRequest, Image,
    LatentDecoder, Progress, Result, TimestepConvention,
};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Array, Dtype};

use crate::control_transformer::ZImageControlTransformer;
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;
use crate::ZImageTransformer;

/// Z-Image latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;

// The decoded-tensor → Image step is identical across families and now lives in core (F-006);
// re-exported so `crate::pipeline::decoded_to_image` and the crate's public surface are unchanged.
pub use mlx_gen::image::decoded_to_image;

/// Seeded txt2img latent noise — shape `[16, 1, height/8, width/8]`, f32. Port of
/// `ZImageLatentCreator.create_noise` (`mx.random.normal` with `key(seed)`). The fork casts to
/// the model precision (bf16) when the latents enter the loop; this returns the raw f32 sample
/// so seeded-RNG parity can be checked directly.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        LATENT_CHANNELS,
        1,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Port of `ZImageLatentCreator.unpack_latents`: `[C, 1, H, W]` → `[1, C, H, W]` (add a batch
/// axis, drop the singleton temporal axis) before VAE decode.
pub fn unpack_latents(latents: &Array) -> Result<Array> {
    Ok(latents.expand_dims(0)?.squeeze_axes(&[2])?)
}

/// `cap_feats = encoder_out[0, :num_valid, :]` — drop the batch axis and the padded tail. The
/// text encoder returns `[1, seq, hidden]` (padded to max length); the DiT consumes only the
/// valid caption tokens. (mlx-rs has no slice op, so this is a range-gather.)
pub fn slice_valid(encoder_out: &Array, num_valid: i32) -> Result<Array> {
    let sh = encoder_out.shape();
    let (s, h) = (sh[1], sh[2]);
    let flat = encoder_out.reshape(&[s, h])?;
    let idx = Array::from_slice(&(0..num_valid).collect::<Vec<i32>>(), &[num_valid]);
    Ok(flat.take_axis(&idx, 0)?)
}

/// Flow-match Euler denoise loop with progress + cooperative cancellation: each step predicts the
/// velocity with the DiT and takes an Euler step, emitting a [`Progress::Step`] and checking
/// `cancel` between steps. `latents` is the seeded init (see [`create_noise`]); `cap_feats` is the
/// text-encoder conditioning. Returns the final latents (pre-VAE).
///
/// `start_step` is the first schedule index to run — `0` for txt2img, `init_time_step` for img2img
/// (the fork's `range(init_time_step, num_steps)`). Progress is reported over the steps actually
/// run (`total = num_steps - start_step`).
///
/// Mirrors the fork's loop: `timestep = 1 - sigma[t]` (the transformer applies its own
/// `t_scale`), `latents += (sigma[t+1] - sigma[t]) * velocity`.
/// `block_window` is ladder rung 4 (SC-15754): `Some(n)` streams the unified stack `n` blocks at a
/// time instead of running the resident one; `None` is the historical forward, untouched.
#[allow(clippy::too_many_arguments)]
pub fn denoise_with_progress(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cap_feats: &Array,
    start_step: usize,
    budget: AttentionBudget,
    block_window: Option<usize>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // The patchify metadata + RoPE freqs depend only on the (loop-constant) latent dims and caption,
    // so build them once and reuse across steps instead of rederiving them every forward (F-042).
    let sh = latents.shape();
    let prep = transformer.prepare((sh[0], sh[1], sh[2], sh[3]), cap_feats)?;
    // Z-Image is rectified-flow (FLOW prediction) and feeds `1 - sigma` as the transformer timestep
    // (the model applies its own `t_scale`) — [`TimestepConvention::OneMinusSigma`]. Route through the
    // unified curated-sampler framework (epic 7114 P3): `euler` reproduces the legacy flow-match step
    // within the N1 tolerance, and the curated menu becomes selectable. Cancellation, the per-step
    // `eval` (sc-5522 / sc-5399), and progress live in `run_flow_sampler`. img2img slices the schedule
    // from `start_step` so the blended init latents are denoised from the matching sigma.
    // The cancel flag joins the budget here so a bounded call can stop between chunks (SC-15615);
    // the unbounded fast path has no boundary, so this is inert for every unselected request.
    let plan = AttentionPlan::budgeted(budget).with_cancel(cancel);
    // Rung 4 (SC-15754): the window is built once for the whole loop (it is loop-constant), and the
    // per-window cancellation check inside `run_windowed` joins the per-step one below.
    let window = transformer.block_window(block_window, cancel)?;
    let predict =
        |x: &Array, timestep: f32| transformer.forward_with_rungs(&prep, x, timestep, plan, window);
    let start = start_step.min(scheduler.sigmas.len().saturating_sub(1));
    run_flow_sampler(
        sampler_name,
        TimestepConvention::OneMinusSigma,
        &scheduler.sigmas[start..],
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// [`denoise_with_progress`] from step 0, with no progress callback and no cancellation — the bare
/// loop used by the stage-wise parity tests. Composes the parity-proven transformer + scheduler;
/// full-weights numeric parity is the real-hardware E2E (sc-2352).
pub fn denoise(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    cap_feats: &Array,
) -> Result<Array> {
    denoise_with_progress(
        transformer,
        scheduler,
        None,
        0,
        latents,
        cap_feats,
        0,
        AttentionBudget::UNBOUNDED,
        None,
        &CancelFlag::default(),
        &mut |_| {},
    )
}

/// Flow-match Euler denoise loop with **classifier-free guidance** for the **base** (non-distilled)
/// Z-Image model (sc-8320). Unlike [`denoise_with_progress`] (which runs a single cond forward — the
/// distilled Turbo path is guidance-free), the base is an undistilled foundation model that supports
/// real CFG: each step runs the DiT **twice** — once on the prompt conditioning (`cap_feats`) and once
/// on the negative/uncond conditioning (`neg_cap_feats`) — and combines the two velocities as
/// `v = v_uncond + guidance·(v_cond − v_uncond)` before the Euler step (the diffusers `ZImagePipeline`
/// CFG combine; `cfg_normalization=False` default, so no per-step rescale). The cond + uncond
/// conditionings are loop-constant, so each gets its own cached `Prepared` built once.
///
/// `guidance == 1.0` (or `neg_cap_feats == None`) collapses to a single cond forward — identical to the
/// Turbo loop — so a base request with `guidance=1` costs the same as Turbo. `start_step` mirrors the
/// base loop (`0` for txt2img, [`init_time_step`] for img2img).
///
/// **CFG gate (F-093, reference-verified):** the `guidance != 1.0` skip is CORRECT and bit-identical,
/// NOT the sana/diffusers `guidance_scale > 1.0`. mlx-gen (like the mflux fork) uses the *shifted*
/// guidance convention `v = v_uncond + guidance·(v_cond − v_uncond)`, so `guidance == 1.0` yields
/// exactly `v_cond` — running the uncond forward would be wasted. The diffusers `ZImagePipeline` gates
/// on `_guidance_scale > 0` with the un-shifted formula `pred = cond + scale·(cond − uncond)`; its
/// `scale == 0` (CFG off) is precisely this convention's `guidance == 1.0` (`scale = guidance − 1`), so
/// the two gates agree. Do NOT "fix" this to `> 1.0`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg_with_progress(
    transformer: &ZImageTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cap_feats: &Array,
    neg_cap_feats: Option<&Array>,
    guidance: f32,
    start_step: usize,
    budget: AttentionBudget,
    block_window: Option<usize>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // Cond (and, when CFG is active, uncond) patchify metadata + RoPE freqs depend only on the
    // loop-constant latent dims + caption — build each once and reuse across steps (F-042).
    let sh = latents.shape();
    let prep = transformer.prepare((sh[0], sh[1], sh[2], sh[3]), cap_feats)?;
    let cfg_on = guidance != 1.0;
    let neg_prep = match (cfg_on, neg_cap_feats) {
        (true, Some(neg)) => Some(transformer.prepare((sh[0], sh[1], sh[2], sh[3]), neg)?),
        _ => None,
    };
    let g = Array::from_slice(&[guidance], &[1]);
    let plan = AttentionPlan::budgeted(budget).with_cancel(cancel);
    // Rung 4 covers BOTH CFG forwards — the uncond pass runs the same 30 blocks and would otherwise
    // re-materialize the whole stack resident, leaving the bound half-applied.
    let window = transformer.block_window(block_window, cancel)?;
    // Same FLOW / `1 - sigma` convention + unified curated-sampler routing as the base loop; the only
    // delta is the per-step CFG combine of two velocities.
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        let v_cond = transformer.forward_with_rungs(&prep, x, timestep, plan, window)?;
        match &neg_prep {
            Some(np) => {
                let v_uncond = transformer.forward_with_rungs(np, x, timestep, plan, window)?;
                // v = v_uncond + guidance·(v_cond − v_uncond).
                let delta = v_cond.subtract(&v_uncond)?;
                Ok(v_uncond.add(&delta.multiply(&g)?)?)
            }
            None => Ok(v_cond),
        }
    };
    let start = start_step.min(scheduler.sigmas.len().saturating_sub(1));
    run_flow_sampler(
        sampler_name,
        TimestepConvention::OneMinusSigma,
        &scheduler.sigmas[start..],
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// [`denoise_with_progress`] for the ControlNet variant: each step predicts the velocity with the
/// [`ZImageControlTransformer`], passing the (constant) `control_context` + `control_context_scale`
/// to every forward (the fork's `ZImageControl._control_predict`). Same Euler step, progress, and
/// cooperative cancellation as the base loop. `start_step` is `0` for txt2img+control and
/// `init_time_step` for img2img+control.
#[allow(clippy::too_many_arguments)]
pub fn denoise_control_with_progress(
    transformer: &ZImageControlTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cap_feats: &Array,
    control_context: &Array,
    control_context_scale: f32,
    start_step: usize,
    budget: AttentionBudget,
    block_window: Option<usize>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // Patchify metadata, RoPE freqs, and the embedded (constant) control context depend only on the
    // loop-constant latent dims + caption + control context — build once and reuse every step (F-042).
    let sh = latents.shape();
    let prep =
        transformer.prepare_control((sh[0], sh[1], sh[2], sh[3]), cap_feats, control_context)?;
    // Same unified-framework routing as the base loop (epic 7114 P3), with the control branch in the
    // `predict` closure (the fork's `ZImageControl._control_predict`).
    let plan = AttentionPlan::budgeted(budget).with_cancel(cancel);
    let window = transformer.control_block_window(block_window, cancel)?;
    let predict = |x: &Array, timestep: f32| {
        transformer.forward_with_control_rungs(
            &prep,
            x,
            timestep,
            control_context_scale,
            None,
            plan,
            window,
        )
    };
    let start = start_step.min(scheduler.sigmas.len().saturating_sub(1));
    run_flow_sampler(
        sampler_name,
        TimestepConvention::OneMinusSigma,
        &scheduler.sigmas[start..],
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// [`denoise_control_with_progress`] with **classifier-free guidance** for the **base** (non-distilled)
/// Z-Image ControlNet variant (sc-8251). The Turbo control loop runs a single cond forward (Turbo is
/// guidance-distilled → CFG-free); the base control variant is built on the undistilled base DiT, so it
/// composes the same real CFG as [`denoise_cfg_with_progress`] **on top of** the control branch: each
/// step runs the control DiT **twice** — once on the prompt conditioning (`cap_feats`) and once on the
/// negative/uncond conditioning (`neg_cap_feats`) — both threading the **same** (loop-constant) f32
/// `control_context` + `control_context_scale`, and combines the two velocities as
/// `v = v_uncond + guidance·(v_cond − v_uncond)` before the Euler step (the diffusers `ZImagePipeline`
/// CFG combine; `cfg_normalization=False` default, so no per-step rescale).
///
/// The control embedding (`c_emb`) is derived from `control_context` independently of the caption, so
/// each branch gets its own cached `ControlPrepared` (cond + uncond captions, identical control
/// context) built once and reused every step (F-042). `guidance == 1.0` (or `neg_cap_feats == None`)
/// collapses to a single cond forward — byte-identical to [`denoise_control_with_progress`] — so a base
/// control request with `guidance=1` costs exactly what the Turbo control loop does. `start_step` is `0`
/// for txt2img+control and [`init_time_step`] for img2img+control, mirroring the other control loop.
#[allow(clippy::too_many_arguments)]
pub fn denoise_control_cfg_with_progress(
    transformer: &ZImageControlTransformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cap_feats: &Array,
    neg_cap_feats: Option<&Array>,
    guidance: f32,
    control_context: &Array,
    control_context_scale: f32,
    start_step: usize,
    budget: AttentionBudget,
    block_window: Option<usize>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // Cond (and, when CFG is active, uncond) control prep — patchify metadata + RoPE freqs + the
    // embedded constant control context — depend only on the loop-constant latent dims + caption +
    // control context, so build each once and reuse across steps (F-042).
    let sh = latents.shape();
    let prep =
        transformer.prepare_control((sh[0], sh[1], sh[2], sh[3]), cap_feats, control_context)?;
    let cfg_on = guidance != 1.0;
    let neg_prep = match (cfg_on, neg_cap_feats) {
        (true, Some(neg)) => {
            Some(transformer.prepare_control((sh[0], sh[1], sh[2], sh[3]), neg, control_context)?)
        }
        _ => None,
    };
    let g = Array::from_slice(&[guidance], &[1]);
    let plan = AttentionPlan::budgeted(budget).with_cancel(cancel);
    let window = transformer.control_block_window(block_window, cancel)?;
    // Same FLOW / `1 - sigma` convention + unified curated-sampler routing as the base CFG loop; the
    // delta vs `denoise_cfg_with_progress` is that each forward runs the control branch (constant
    // control context + scale threaded through every step, the fork's `ZImageControl._control_predict`).
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        let v_cond = transformer.forward_with_control_rungs(
            &prep,
            x,
            timestep,
            control_context_scale,
            None,
            plan,
            window,
        )?;
        match &neg_prep {
            Some(np) => {
                let v_uncond = transformer.forward_with_control_rungs(
                    np,
                    x,
                    timestep,
                    control_context_scale,
                    None,
                    plan,
                    window,
                )?;
                // v = v_uncond + guidance·(v_cond − v_uncond).
                let delta = v_cond.subtract(&v_uncond)?;
                Ok(v_uncond.add(&delta.multiply(&g)?)?)
            }
            None => Ok(v_cond),
        }
    };
    let start = start_step.min(scheduler.sigmas.len().saturating_sub(1));
    run_flow_sampler(
        sampler_name,
        TimestepConvention::OneMinusSigma,
        &scheduler.sigmas[start..],
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// img2img init image → packed clean latents `[16, 1, H/8, W/8]` (f32). Port of the fork's
/// `LatentCreator.encode_image` ∘ `ZImageLatentCreator.pack_latents`: PIL-LANCZOS scale to the
/// target dims, normalize `[0,255] → [-1,1]` as NCHW, VAE-encode (mean → latent space), pack.
pub fn encode_init_latents(
    vae: &Vae,
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, target_width, target_height)?;
    let encoded = vae.encode(&image_nchw)?; // [1, 16, H/8, W/8]
    pack_latents(&encoded)
}

/// Build the 33-channel VAE-encoded **control context** from a control image (e.g. a rendered
/// pose skeleton) — the fork's `ZImageControl._encode_control_context`. VAE-encode to 16ch latents
/// (the exact img2img [`encode_init_latents`] path: LANCZOS → `[-1,1]` NCHW → encode → pack
/// `[16,1,H/8,W/8]`), then concat a zero mask (1ch) and a zero inpaint latent (16ch) → `[33, 1,
/// H/8, W/8]`. Pure-pose control has no init image and no mask, so those two channel groups are
/// zeros; the channel layout (control latent | mask | inpaint) matches the Fun-Controlnet-Union
/// `control_all_x_embedder`'s 33ch input.
pub fn encode_control_context(
    vae: &Vae,
    control_image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let control_latents = encode_init_latents(vae, control_image, target_width, target_height)?;
    let sh = control_latents.shape(); // [16, 1, H/8, W/8]
    let (c, fdim, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    // i64 intermediates before the usize cast — bounded by the validated max resolution today, but a
    // raw i32 product is a latent overflow footgun if the caps change (F-056).
    let mask = Array::from_slice(
        &vec![0f32; (fdim as i64 * h as i64 * w as i64) as usize],
        &[1, fdim, h, w],
    );
    let inpaint = Array::from_slice(
        &vec![0f32; (c as i64 * fdim as i64 * h as i64 * w as i64) as usize],
        &[c, fdim, h, w],
    );
    Ok(concatenate_axis(&[&control_latents, &mask, &inpaint], 0)?)
}

/// Port of `ZImageLatentCreator.pack_latents`: VAE-encoder latent `[1, C, H/8, W/8]` (or a 5-D
/// `[1, C, 1, H/8, W/8]`) → `[C, 1, H/8, W/8]`, matching the seeded-noise layout so the two can be
/// blended.
pub fn pack_latents(encoded: &Array) -> Result<Array> {
    let sh = encoded.shape();
    let e = if sh.len() == 5 {
        encoded.reshape(&[sh[0], sh[1], sh[3], sh[4]])? // drop temporal axis
    } else {
        encoded.clone()
    };
    Ok(e.expand_dims(2)?.squeeze_axes(&[0])?)
}

/// Prompt → `cap_feats` (f32): tokenize with the Qwen chat template, run the text encoder, slice off
/// the padded tail to the valid caption tokens. Shared by the base + control generators and the
/// trainer (F-035); `id` only labels the empty-prompt error. An empty prompt tokenizes to `[1, 0]`,
/// so guard on shape before any host readback (`host_i32` on a size-0 array would panic) —
/// `validate_request` already rejects it, this is defense-in-depth at the encode boundary.
pub(crate) fn encode_prompt(
    tokenizer: &TextTokenizer,
    text_encoder: &TextEncoder,
    prompt: &str,
    id: &str,
    encoder_window: Option<EncoderWindow<'_>>,
) -> Result<Array> {
    let t = tokenizer.tokenize(prompt)?;
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&t);
    if input_ids.shape()[1] == 0 {
        return Err(Error::Msg(format!("{id}: empty prompt")));
    }
    let num_valid: i32 = host_i32(&attention_mask)?.iter().sum();
    if num_valid == 0 {
        return Err(Error::Msg(format!("{id}: empty prompt")));
    }
    let enc = EncoderWindow::encode(encoder_window, text_encoder, &input_ids, &attention_mask)?;
    slice_valid(&enc, num_valid)
}

/// Negative/unconditional caption → `cap_feats` (f32) for the base model's CFG path (sc-8320). Unlike
/// [`encode_prompt`], an **empty** caption is legitimate here — it is the unconditional embedding the
/// CFG uncond branch consumes.
///
/// For an empty negative prompt we render the QwenInstruct chat scaffolding around `""` directly via
/// [`TextTokenizer::encode_chat_ids`] (`<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n`) rather
/// than going through [`TextTokenizer::tokenize`]. The templated tokens are all valid, so the
/// attention mask is all-ones. This mirrors the non-padded `tokenize` output for a real caption and
/// matches the diffusers/Qwen reference uncond (the base-CFG-with-unset-negative path that sc-8958
/// fixed). A non-empty negative prompt takes the ordinary `tokenize` path.
///
/// Rationale note (sc-8958 / review F-031): an earlier comment justified this on the grounds that
/// gen-core short-circuits an empty prompt to a `[1, 0]` sequence "before the chat template is applied
/// (`pad_to_max_length = false`)", which would trip the size-0 guard below. That mechanism **cannot
/// fire here**: z-image builds its tokenizer with `pad_to_max_length: true` (loader.rs), and gen-core's
/// empty→`[1,0]` short-circuit is gated on `!pad_to_max_length` (tokenizer.rs). The real reason to
/// render the scaffolding directly is fidelity + cost: it produces the diffusers-correct uncond
/// embedding without depending on max-length padding. (No MLX-side failure was ever reproduced for the
/// original sc-8958 report; the candle-side sc-8646 trap it cited was real, but does not apply to the
/// MLX backend under `pad_to_max_length: true`.)
pub(crate) fn encode_uncond(
    tokenizer: &TextTokenizer,
    text_encoder: &TextEncoder,
    negative: &str,
    encoder_window: Option<EncoderWindow<'_>>,
) -> Result<Array> {
    let t = if negative.is_empty() {
        // `add_special_tokens = true` mirrors `tokenize`'s `encode(text, true)` (Qwen adds no BOS/EOS,
        // so the ids equal the templated tokens). Mask is all-ones: every scaffolding token is valid.
        let ids = tokenizer.encode_chat_ids("", true)?;
        let mask = vec![1; ids.len()];
        TokenizerOutput { ids, mask }
    } else {
        tokenizer.tokenize(negative)?
    };
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&t);
    if input_ids.shape()[1] == 0 {
        return Err(Error::Msg(
            "z_image: negative conditioning tokenized to an empty sequence".into(),
        ));
    }
    let num_valid: i32 = host_i32(&attention_mask)?.iter().sum();
    if num_valid == 0 {
        return Err(Error::Msg(
            "z_image: negative conditioning has no valid tokens".into(),
        ));
    }
    let enc = EncoderWindow::encode(encoder_window, text_encoder, &input_ids, &attention_mask)?;
    slice_valid(&enc, num_valid)
}

/// Resolve the single img2img init image + its strength from the request's conditioning (F-035). The
/// per-reference strength wins over `req.strength`. Z-Image conditions on exactly one init image, so
/// more than one `Reference` is an error (multi-image is `MultiReference`, unadvertised here).
pub(crate) fn resolve_reference<'a>(
    req: &'a GenerationRequest,
    id: &str,
) -> Result<Option<(&'a Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(Error::Msg(format!(
                    "{id}: multiple reference images are not supported (single img2img init only)"
                )));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
}

/// The decode-time tiling policy for a request (sc-13571, GitHub #1658). `is_sequential` is the
/// fit-gate's memory-constrained-Mac signal (`OffloadPolicy::Sequential`): under it the VAE decode is
/// tiled to bound its ~14 GiB 1024² transient; a large-memory `Resident` machine decodes EXACTLY
/// (untiled, `None`), except for a very large output that would spike past even a big Mac. 512 px is the
/// parity sweet spot for this GroupNorm VAE — visually seam-free at 1024²/1280², where smaller tiles
/// would drift the per-tile norm statistics.
pub(crate) fn decode_tiling(req: &GenerationRequest, is_sequential: bool) -> Option<TilingConfig> {
    // SC-15615: the shared selector's explicit bounded-decode signal (`GenerationMemory::tile_vae_decode`)
    // joins the two historical triggers. The ladder is cumulative — a selector that picks rung 3
    // (bounded attention) also sets rung 2 — so without this a contract-driven rung-3 selection on a
    // `Resident` load would silently decode untiled.
    //
    // SC-15510: the tile geometry is no longer hardcoded. A selection carries the exact edge/overlap
    // it was calibrated at, from the ladder the contract publishes; the two historical triggers, and
    // a selection that names no geometry, keep the 512/64 default this VAE was tuned at (sc-13571).
    // Publishing a one-element candidate list was what blocked SC-15508's "a single-point pass cannot
    // mark untested production parameters Verified".
    let requested = req.memory.is_some_and(|m| m.tile_vae_decode);
    (is_sequential || requested || req.width.max(req.height) > 2048).then(|| {
        let (edge, overlap) = decode_tile_geometry(req);
        TilingConfig::spatial_only(edge as i32, overlap as i32)
    })
}

/// The (edge, overlap) a request decodes at: the selection's values when it names them, else the
/// provider default. Pure, so the ladder's plumbing is unit-testable without a VAE.
pub(crate) fn decode_tile_geometry(req: &GenerationRequest) -> (u32, u32) {
    let memory = req.memory;
    (
        memory
            .and_then(|m| m.decode_tile_edge)
            .unwrap_or(crate::image_memory::DECODE_TILE_EDGE),
        memory
            .and_then(|m| m.decode_overlap)
            .unwrap_or(crate::image_memory::DECODE_OVERLAP),
    )
}

/// The transformer-residency window for one request (SC-15754, ladder rung 4).
///
/// [`GenerationMemory::stream_transformer_blocks`] is the shared selector's rung-4 signal and
/// `transformer_window_size` its parameter; a selection that sets the flag without a size falls back
/// to the provider's default window. `None` — every request that did not select rung 4 — is the
/// historical resident stack, untouched.
pub(crate) fn block_window_size(req: &GenerationRequest) -> Option<usize> {
    let memory = req.memory?;
    memory.stream_transformer_blocks.then(|| {
        memory
            .transformer_window_size
            .unwrap_or(crate::image_memory::TRANSFORMER_WINDOW_SIZE) as usize
    })
}

/// [`block_window_size`] plus the residency-policy precondition, shared by all four generators.
///
/// **Why this rejects instead of degrading.** Streaming bounds transformer residency only if the
/// resident `layers` are *not* also materialized. Under `Sequential` they are not — the heavy bundle
/// is rebuilt per generate, its blocks are unevaluated lazy handles (SC-15744 measured 1073 handles at
/// 0.0 MiB), and a windowed forward never touches them, so they stay at ~0 bytes. Under `Resident` the
/// same generator serves many requests, so after the first ordinary render the blocks ARE materialized
/// and a window would be *added on top of* them: strictly more memory and strictly slower.
///
/// Rung 1 resolves the same load-time-vs-request-time seam by silently yielding resident behaviour.
/// Rung 4 must not: a silent degradation here writes a rung-4 row into the calibration evidence for a
/// run that saved nothing, which is the false green SC-15750's own mutation check exists to prevent —
/// one layer up, and invisible to it.
pub(crate) fn resolve_block_window(
    req: &GenerationRequest,
    is_sequential: bool,
    id: &str,
) -> Result<Option<usize>> {
    Ok(resolve_window_for(req, is_sequential, id)?
        .filter(|_| block_window_component(req).includes_dit()))
}

/// The rung-4 **component scope** for this request (SC-15794). `None` ⇒ [`TransformerComponent::Dit`],
/// the by-convention scope every request had before the component existed.
pub(crate) fn block_window_component(req: &GenerationRequest) -> TransformerComponent {
    req.memory
        .and_then(|m| m.transformer_window_component)
        .unwrap_or_default()
}

/// [`resolve_block_window`]'s text-encoder twin (SC-15794): the window to apply to the **conditioning**
/// stack, or `None` when this request's component scope excludes the encoder.
///
/// Shares the Sequential precondition for the same reason: under `Resident` the encoder is
/// materialized once and reused across requests, so a window would be added on top of resident weights
/// — more memory, not less.
pub(crate) fn resolve_encoder_window(
    req: &GenerationRequest,
    is_sequential: bool,
    id: &str,
) -> Result<Option<usize>> {
    Ok(resolve_window_for(req, is_sequential, id)?
        .filter(|_| block_window_component(req).includes_text_encoder()))
}

/// The requested window plus the shared residency precondition, before the component scope narrows it.
fn resolve_window_for(
    req: &GenerationRequest,
    is_sequential: bool,
    id: &str,
) -> Result<Option<usize>> {
    let window = block_window_size(req);
    if window.is_some() && !is_sequential {
        return Err(Error::Unsupported(format!(
            "{id}: bounded transformer residency needs a Sequential (staged) load — on a Resident \
             generator the trunk blocks are already materialized, so a window would add memory \
             rather than bound it. Load with OffloadPolicy::Sequential."
        )));
    }
    Ok(window)
}

/// One request's rung-4 text-encoder scope: the window, plus the cancel flag its boundaries are
/// checked against. `Copy`, so it threads through the encode helpers as cheaply as the window alone.
#[derive(Clone, Copy)]
pub(crate) struct EncoderWindow<'a> {
    pub(crate) window: usize,
    pub(crate) cancel: &'a CancelFlag,
}

impl<'a> EncoderWindow<'a> {
    /// The scope for this request, or `None` when the encoder is not in it.
    pub(crate) fn resolve(
        req: &'a GenerationRequest,
        is_sequential: bool,
        id: &str,
    ) -> Result<Option<Self>> {
        Ok(
            resolve_encoder_window(req, is_sequential, id)?.map(|window| Self {
                window,
                cancel: &req.cancel,
            }),
        )
    }

    /// Run `text_encoder` under this scope, or resident when there is none.
    fn encode(
        scope: Option<Self>,
        text_encoder: &TextEncoder,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<Array> {
        match scope {
            Some(s) => text_encoder.forward_windowed(input_ids, attention_mask, s.window, s.cancel),
            None => text_encoder.forward(input_ids, attention_mask),
        }
    }
}

/// Request-local, calibration-only fault injection at a physical phase boundary (SC-15449's
/// [`GenerationMemory::calibration_error_phase`](mlx_gen::gen_core::GenerationMemory), the MLX twin of
/// `candle_gen_krea`'s `maybe_inject_calibration_error`).
///
/// This is what lets the cross-backend conformance harness verify the story's **cleanup on error**
/// requirement the same way it verifies cancellation: fail deterministically at a named boundary, then
/// assert the next request is unaffected. Production selectors leave the field `None` (the default),
/// so every ordinary request is untouched — the check is a `None` comparison.
pub(crate) fn calibration_fault(
    req: &GenerationRequest,
    phase: mlx_gen::gen_core::ImageMemoryPhase,
    id: &str,
) -> Result<()> {
    match req.memory {
        Some(memory) if memory.calibration_error_phase == Some(phase) => Err(Error::Msg(format!(
            "{id}: injected image-memory calibration error at {phase:?}"
        ))),
        _ => Ok(()),
    }
}

/// The attention rung for one request (SC-15615, ladder rung 3).
///
/// [`GenerationMemory::chunk_attention`] is the shared selector's rung-3 signal; the exact score
/// budget is the single provider-declared [`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`],
/// which is also the value published in the provider contract's `attention_chunk_sizes` and
/// re-validated by the request scope's `configure_attention`. Anything else — including a warm
/// request that carried a rung last time — is the unbounded, byte-identical forward.
pub(crate) fn attention_budget(req: &GenerationRequest) -> AttentionBudget {
    match req.memory {
        Some(memory) if memory.chunk_attention => AttentionBudget::CONSTRAINED,
        _ => AttentionBudget::UNBOUNDED,
    }
}

/// Phase 1 of the staged render (sc-13571): the per-image denoise count loop → `Vec` of **evaluated**
/// latents `[16,1,H/8,W/8]`. Each image's latents are `eval`ed before the next so (a) the DiT graph is
/// cut — letting the caller shed the DiT before decode via [`mlx_gen::StagedHeavy`] — and (b) the batch
/// working set stays one image, not all `count`. The seeded bf16 noise + img2img `clean`-blend are
/// unchanged from the old single-pass loop (PARITY-BF16 sc-2609; F-032 defensive `sigma` get).
pub(crate) fn denoise_batch(
    scheduler: &FlowMatchEuler,
    clean: Option<&Array>,
    start_step: usize,
    base_seed: u64,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
    mut denoise: impl FnMut(Array, u64, &mut dyn FnMut(Progress)) -> Result<Array>,
) -> Result<Vec<Array>> {
    // sc-2963/F-036: enable the fusable-elementwise `mx.compile` glue ONCE for the whole denoise batch.
    let _compile_glue = crate::CompileGlueGuard::enable();
    let mut latents = Vec::with_capacity(req.count as usize);
    for i in 0..req.count {
        let seed = base_seed.wrapping_add(i as u64);
        let noise = create_noise(seed, req.width, req.height)?.as_dtype(Dtype::Bfloat16)?;
        let init = match clean {
            Some(clean) => {
                let sigma = scheduler.sigmas.get(start_step).ok_or_else(|| {
                    Error::Msg(format!(
                        "img2img start step {start_step} out of range for {}-element schedule",
                        scheduler.sigmas.len()
                    ))
                })?;
                add_noise_by_interpolation(clean, &noise, *sigma)?
            }
            None => noise,
        };
        let out = denoise(init, seed, on_progress)?;
        // Materialize so the DiT is no longer referenced through MLX's lazy graph (the caller's shed
        // would otherwise free nothing).
        mlx_rs::transforms::eval([&out])?;
        latents.push(out);
    }
    Ok(latents)
}

/// Phase 2 of the staged render (sc-13571): decode each evaluated latent → RGB8 [`Image`]. Uses the
/// memory-bounded [`Vae::decode_tiled`] when `tiling` is `Some` (small-Mac / large-image), else the
/// exact single-pass [`Vae::decode`]; a PiD super-res decoder, when present, takes precedence (its own
/// decode, untiled). `[16,1,H,W] → [1,16,H,W]`; the native VAE + PiD both accept the 4-D latent directly.
pub(crate) fn decode_batch(
    vae: &Vae,
    pid_decoder: Option<&dyn LatentDecoder>,
    tiling: Option<&TilingConfig>,
    latents: Vec<Array>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let mut images = Vec::with_capacity(latents.len());
    for latent in latents {
        on_progress(Progress::Decoding);
        let unpacked = unpack_latents(&latent)?;
        let decoded = match (pid_decoder, tiling) {
            (Some(pid), _) => pid.decode(&unpacked)?,
            (None, Some(cfg)) => vae.decode_tiled(&unpacked, cfg, Some(cancel))?,
            (None, None) => vae.decode(&unpacked)?,
        };
        images.push(decoded_to_image(&decoded.as_dtype(Dtype::Float32)?)?);
    }
    Ok(images)
}

/// Render one preview sample (sc-5637) from the **in-progress training adapter** already installed
/// on `transformer`: seed → noise → flow-match denoise → VAE decode → [`Image`]. A stripped txt2img
/// of [`render_batch`] (no count loop, no img2img blend, no `GenerationRequest`) for the trainer's
/// periodic preview. `cap` is the (already dtype-matched) caption conditioning for the sample prompt;
/// `dtype` is the trainer's compute dtype (bf16/f32) so the noise enters the loop at the same
/// precision as the cast DiT weights. No progress/cancel plumbing — the caller drives the cadence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_sample(
    transformer: &ZImageTransformer,
    vae: &Vae,
    scheduler: &FlowMatchEuler,
    cap: &Array,
    seed: u64,
    edge: u32,
    dtype: Dtype,
    cancel: &CancelFlag,
) -> Result<Image> {
    let _compile_glue = crate::CompileGlueGuard::enable();
    let noise = create_noise(seed, edge, edge)?.as_dtype(dtype)?;
    // F-117: thread the training cancel flag through the preview denoise (was a throwaway
    // `CancelFlag::default()` via the bare `denoise`), so a cancel during a preview's multi-step
    // denoise is honored at the next step boundary rather than only between previews.
    let latents = denoise_with_progress(
        transformer,
        scheduler,
        None,
        0,
        noise,
        cap,
        0,
        // Training previews never select a memory rung; keep the unbounded (byte-identical) forward.
        AttentionBudget::UNBOUNDED,
        None,
        cancel,
        &mut |_| {},
    )?;
    // Honor a cancel that arrived during the denoise before the (lazy) VAE decode.
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // [16,1,H,W] -> [1,16,H,W] -> [1,16,1,H,W] for VAE decode (same as render_batch).
    let unpacked = unpack_latents(&latents)?;
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]])?;
    let decoded = vae.decode(&latent5)?.as_dtype(Dtype::Float32)?;
    decoded_to_image(&decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    fn req(
        w: u32,
        h: u32,
        memory: Option<mlx_gen::gen_core::GenerationMemory>,
    ) -> GenerationRequest {
        GenerationRequest {
            prompt: "a fox".into(),
            width: w,
            height: h,
            memory,
            ..Default::default()
        }
    }

    /// SC-15615: the shared selector's explicit bounded-decode signal is a THIRD trigger alongside the
    /// two historical ones, so a contract-driven rung-2/3 selection tiles even on a `Resident` load.
    #[test]
    fn decode_tiling_honors_the_selector_signal_as_well_as_the_legacy_triggers() {
        use mlx_gen::gen_core::GenerationMemory;

        // Historical triggers, unchanged: Sequential, or an output edge above 2048.
        assert!(decode_tiling(&req(1024, 1024, None), true).is_some());
        assert!(decode_tiling(&req(2560, 1024, None), false).is_some());
        // The default resident 1024² render still decodes EXACTLY (untiled) — no behaviour change.
        assert!(decode_tiling(&req(1024, 1024, None), false).is_none());
        // A staged-only selection does not ask for tiling, so it must not get it.
        assert!(
            decode_tiling(&req(1024, 1024, Some(GenerationMemory::default())), false).is_none()
        );
        // The rung-2 signal does, on a Resident load.
        let tiled = decode_tiling(
            &req(
                1024,
                1024,
                Some(GenerationMemory {
                    tile_vae_decode: true,
                    ..Default::default()
                }),
            ),
            false,
        )
        .expect("the bounded-decode signal must engage tiling");
        // And it is the same 512/64 spatial-only policy the contract advertises — not a new one.
        let spatial = tiled.spatial.expect("spatial tiling");
        assert_eq!(
            (spatial.tile_px, spatial.overlap_px),
            (
                crate::image_memory::DECODE_TILE_EDGE as i32,
                crate::image_memory::DECODE_OVERLAP as i32
            ),
            "the executed tile geometry must match the contract's advertised candidates"
        );
        assert!(tiled.temporal.is_none(), "z-image decode is spatial-only");
    }

    /// Calibration fault injection is inert unless a request explicitly names a phase, and fires at
    /// exactly the named one. This is the hook the conformance harness uses to verify the story's
    /// **cleanup on error** requirement.
    #[test]
    fn calibration_faults_fire_only_at_the_named_phase() {
        use mlx_gen::gen_core::{GenerationMemory, ImageMemoryPhase};

        const PHASES: [ImageMemoryPhase; 3] = [
            ImageMemoryPhase::Conditioning,
            ImageMemoryPhase::Denoise,
            ImageMemoryPhase::Decode,
        ];

        // No memory block, and a memory block with no named phase: every boundary passes.
        for memory in [None, Some(GenerationMemory::default())] {
            let r = req(1024, 1024, memory);
            for phase in PHASES {
                calibration_fault(&r, phase, "z_image_turbo").unwrap();
            }
        }

        // A named phase fails at that phase and only that phase.
        for named in PHASES {
            let r = req(
                1024,
                1024,
                Some(GenerationMemory {
                    calibration_error_phase: Some(named),
                    ..Default::default()
                }),
            );
            for phase in PHASES {
                let result = calibration_fault(&r, phase, "z_image_turbo");
                if phase == named {
                    let err = result.unwrap_err().to_string();
                    assert!(
                        err.contains("injected image-memory calibration error"),
                        "{err}"
                    );
                    assert!(err.contains(&format!("{named:?}")), "{err}");
                    assert!(err.contains("z_image_turbo"), "{err}");
                } else {
                    result.unwrap_or_else(|e| {
                        panic!("fault named at {named:?} must not fire at {phase:?}: {e}")
                    });
                }
            }
        }
    }

    /// The rung-3 request knob maps onto exactly one budget, and nothing else engages it.
    #[test]
    fn attention_budget_engages_only_on_the_rung_three_signal() {
        use mlx_gen::attention::AttentionBudget;
        use mlx_gen::gen_core::GenerationMemory;

        assert_eq!(
            attention_budget(&req(1024, 1024, None)),
            AttentionBudget::UNBOUNDED
        );
        assert_eq!(
            attention_budget(&req(1024, 1024, Some(GenerationMemory::default()))),
            AttentionBudget::UNBOUNDED
        );
        assert_eq!(
            attention_budget(&req(
                1024,
                1024,
                Some(GenerationMemory {
                    tile_vae_decode: true,
                    ..Default::default()
                })
            )),
            AttentionBudget::UNBOUNDED
        );
        assert_eq!(
            attention_budget(&req(
                1024,
                1024,
                Some(GenerationMemory {
                    tile_vae_decode: true,
                    chunk_attention: true,
                    ..Default::default()
                })
            )),
            AttentionBudget::CONSTRAINED
        );
    }

    #[test]
    fn resolve_reference_handles_count_and_strength() {
        // F-035: the single shared img2img resolution (was duplicated in both generators). No
        // Reference → None.
        let none = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert_eq!(resolve_reference(&none, "z_image_turbo").unwrap(), None);

        // A per-reference strength wins over req.strength.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: img(64, 64),
                strength: Some(0.3),
            }],
            ..Default::default()
        };
        let (_, s) = resolve_reference(&req, "z_image_turbo").unwrap().unwrap();
        assert_eq!(s, Some(0.3));

        // Missing per-reference strength falls back to req.strength.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: img(64, 64),
                strength: None,
            }],
            ..Default::default()
        };
        let (_, s) = resolve_reference(&req, "z_image_turbo").unwrap().unwrap();
        assert_eq!(s, Some(0.6));

        // More than one Reference is an error (single img2img init only).
        let req = GenerationRequest {
            prompt: "a fox".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: img(64, 64),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(64, 64),
                    strength: None,
                },
            ],
            ..Default::default()
        };
        let err = resolve_reference(&req, "z_image_turbo")
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple reference images"), "{err}");
    }
}
