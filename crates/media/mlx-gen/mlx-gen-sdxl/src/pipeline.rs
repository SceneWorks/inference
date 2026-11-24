//! SDXL T2I sampling pipeline — composes the dual-CLIP conditioning, the seeded prior, the
//! Euler-Ancestral denoise loop with real classifier-free guidance, and the VAE decode. Port of the
//! vendored `StableDiffusionXL.generate_latents` + `_denoising_loop` + `decode`.
//!
//! The U-Net, text encoders, sampler, and CFG run **fp16**, matching the production reference
//! (`StableDiffusionXL(float16=True)`); the VAE runs f32 (it promotes the f16 latents on decode).
//! The RNG is seeded once per image, then the sampler draws the prior + per-step ancestral noise from
//! the global stream — reproducing the reference's exact noise sequence for a seed.

use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, round};
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::gen_core;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{
    CancelFlag, DiffusionSampler, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverStepInfo, DriverSupport, Error, FamilyDriver, Image, LatentDecoder,
    MlxLatentOps, PreviewSink, Progress, Result, SliceInput, SliceSpec, SliceTransition, StepHooks,
};

use crate::inpaint::InpaintBlend;
use crate::long_prompt::ChunkedTokens;
#[cfg(any(test, feature = "training"))]
use crate::sampler::AncestralEuler;
use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::unet::{ControlNet, ControlResiduals, UNet2DConditionModel};
use crate::vae::{Autoencoder, SdxlLatentDecoder};

/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;
/// Latent channel count.
pub const LATENT_CHANNELS: i32 = 4;

/// The SDXL micro-conditioning `time_ids`, hardcoded `[512, 512, 0, 0, 512, 512]` per row — the
/// vendored `StableDiffusionXL.generate_latents` quirk (it does NOT pass the real
/// original/target sizes). Reproduced verbatim for parity. `batch` rows.
pub fn text_time_ids(batch: i32) -> Array {
    let row = [512.0f32, 512.0, 0.0, 0.0, 512.0, 512.0];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

/// Run both CLIP encoders over the (CFG) token batch and assemble the SDXL conditioning:
/// `concat(te1.hidden[-2], te2.hidden[-2])` and `te2.pooled`. `tokens` is `[B, N]` (B=2 with CFG).
///
/// The **single-window** encode: `N` must fit CLIP's position table. It is unchanged since before
/// sc-20528 and is what [`encode_conditioning_windows`] delegates to for every request that fits, so
/// a ≤77-token render produces the identical conditioning it always did.
pub fn encode_conditioning(
    te1: &ClipTextEncoder,
    te2: &ClipTextEncoder,
    tokens: &Array,
) -> Result<(Array, Array)> {
    let o1 = te1.forward(tokens)?;
    let o2 = te2.forward(tokens)?;
    let h1 = &o1.hidden_states[o1.hidden_states.len() - 2];
    let h2 = &o2.hidden_states[o2.hidden_states.len() - 2];
    let conditioning = concatenate_axis(&[h1, h2], -1)?;
    Ok((conditioning, o2.pooled))
}

/// The production encode (sc-20528): one forward per CLIP window, the windows concatenated on the
/// **sequence** axis — the A1111/compel "long prompt weighting" shape that lets SDXL condition on a
/// prompt past CLIP's architectural 77-token context instead of losing its tail.
///
/// Returns `(conditioning [B, n·77, 2048], pooled [B, 1280])`. Cross-attention takes an arbitrary
/// key/value length, so the grown sequence axis is a drop-in for the U-Net and every ControlNet.
///
/// Two properties the callers depend on:
///
/// - **`n == 1` is the legacy path, structurally.** A request whose rows all fit is a single window
///   — the token batch
///   [`tokenize_batch`](crate::tokenizer::ClipBpeTokenizer::tokenize_batch) has always built — and
///   is handed straight to [`encode_conditioning`]: no re-wrap, no `cat`, nothing to drift.
/// - **The pooled embed is window 0's.** Every window carries its own EOS, so the encoder's
///   `argmax` over a concatenation would be ambiguous; diffusers' pooled text-embed is defined on
///   the first window, and that is what the `add_embedding` micro-conditioning gets. The candle twin
///   pools the same way.
pub fn encode_conditioning_windows(
    te1: &ClipTextEncoder,
    te2: &ClipTextEncoder,
    tokens: &ChunkedTokens,
) -> Result<(Array, Array)> {
    // The ≤77 request: the pre-sc-20528 encode, verbatim.
    if let [only] = tokens.windows() {
        return encode_conditioning(te1, te2, only);
    }

    let mut per_window: Vec<Array> = Vec::with_capacity(tokens.len());
    let mut pooled: Option<Array> = None;
    for window in tokens.windows() {
        let o1 = te1.forward(window)?;
        let o2 = te2.forward(window)?;
        let h1 = &o1.hidden_states[o1.hidden_states.len() - 2];
        let h2 = &o2.hidden_states[o2.hidden_states.len() - 2];
        per_window.push(concatenate_axis(&[h1, h2], -1)?); // [B, 77, 2048]
        if pooled.is_none() {
            pooled = Some(o2.pooled);
        }
    }
    let refs: Vec<&Array> = per_window.iter().collect();
    let conditioning = concatenate_axis(&refs, 1)?; // [B, n·77, 2048]
    let pooled = pooled.ok_or_else(|| {
        Error::Msg("sdxl: tokenized request carried no CLIP windows to encode".into())
    })?;
    Ok((conditioning, pooled))
}

/// Components needed for one denoise run (borrowed from the loaded model). `sampler` is any
/// [`DiffusionSampler`] — SDXL's production ancestral [`crate::sampler::AncestralEuler`] or a
/// few-step acceleration sampler (`mlx_gen::{LcmSampler, LightningSampler, TcdSampler}`, sc-2769).
pub struct Denoiser<'a> {
    pub unet: &'a UNet2DConditionModel,
    pub sampler: &'a dyn DiffusionSampler,
    /// Ladder rungs 3 and 4 for this request (SC-15525).
    /// [`SdxlForwardPlan::UNBOUNDED`](crate::plan::SdxlForwardPlan::UNBOUNDED) is the historical
    /// path, which is what `Denoiser::new` builds — the field is public so the production
    /// generate can hand it a bounded plan without a second constructor.
    pub plan: crate::plan::SdxlForwardPlan<'a>,
}

impl<'a> Denoiser<'a> {
    /// The unbounded denoiser — every pre-SC-15525 construction site's exact behaviour.
    pub fn new(unet: &'a UNet2DConditionModel, sampler: &'a dyn DiffusionSampler) -> Self {
        Self {
            unet,
            sampler,
            plan: crate::plan::SdxlForwardPlan::UNBOUNDED,
        }
    }

    /// The denoiser with a bounded-execution plan.
    pub fn with_plan(
        unet: &'a UNet2DConditionModel,
        sampler: &'a dyn DiffusionSampler,
        plan: crate::plan::SdxlForwardPlan<'a>,
    ) -> Self {
        Self {
            unet,
            sampler,
            plan,
        }
    }
}

/// ControlNet conditioning for the denoise loop (sc-3058): the loaded branch, the preprocessed
/// control image (NHWC `[B, H, W, 3]` in `[0,1]`, already CFG-batched to match the UNet input), and
/// the `conditioning_scale`. Each step runs the branch on the model input and injects its residuals.
pub struct ControlContext<'a> {
    pub controlnet: &'a ControlNet,
    /// The precomputed conditioning embedding for the fixed control image
    /// ([`ControlNet::embed_cond`]) — step-invariant, so it is computed once at construction rather
    /// than re-run every denoise step (F-069).
    pub cond_embed: Array,
    pub scale: f32,
}

// Backward-compatible struct API. Registered generators call the explicit `*_with_preview`
// variants below; direct users (including InstantID) retain the exact inert-preview signatures and
// execution semantics they had before sc-16633.

#[allow(clippy::too_many_arguments)]
pub fn denoise(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_inpaint(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    blend: &InpaintBlend,
) -> Result<Array> {
    denoise_inpaint_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        blend,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    control: &ControlContext,
) -> Result<Array> {
    denoise_control_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        control,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_multi_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
) -> Result<Array> {
    denoise_multi_control_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        controls,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_ip(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_ip_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        tokens,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    control: &ControlContext,
    controlnet_encoder: &Array,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_ip_control_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        control,
        controlnet_encoder,
        tokens,
        scale,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_multi_control(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
    controlnet_encoder: &Array,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_ip_multi_control_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        controls,
        controlnet_encoder,
        tokens,
        scale,
    )
}

/// Run the denoise loop with CFG, driven entirely by the sampler's own schedule
/// (`sampler.num_steps()` iterations). `latents` is the seeded init `[1, h, w, 4]`;
/// `conditioning`/`pooled`/`time_ids` carry the CFG batch (B = 2 when `cfg > 1`). Returns the final
/// latents; progress per step; `cancel` between steps. Each iteration:
/// `x_in = scale_model_input(latents)` → U-Net eps → (CFG) → `latents = sampler.step(eps, latents)`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        None,
        &[],
        None,
        None,
    )
}

/// Like [`denoise`] but applies the legacy inpaint **mask-blend** after each step (sc-3057):
/// `latents = (1-mask)·init_noised + mask·latents`. The blend draws no RNG, so the ancestral noise
/// stream is identical to plain img2img (a full-white mask ⇒ bit-identical to [`denoise`]).
#[allow(clippy::too_many_arguments)]
pub fn denoise_inpaint_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    blend: &InpaintBlend,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        Some(blend),
        &[],
        None,
        None,
    )
}

/// Like [`denoise`] but runs a ControlNet branch each step and injects its residuals into the UNet
/// (sc-3058). Works on the txt2img or img2img init (set up by the caller); `scale = 0` ⇒ identical
/// to [`denoise`] (the residuals vanish).
#[allow(clippy::too_many_arguments)]
pub fn denoise_control_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    control: &ControlContext,
) -> Result<Array> {
    denoise_multi_control_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        std::slice::from_ref(control),
    )
}

/// Like [`denoise_control`] but runs **multiple** ControlNet branches and sums their residuals — the
/// diffusers `MultiControlNetModel` rule (sc-3378). `controls[i]` pairs with the `i`-th branch; all
/// share the text `conditioning` as their cross-attention input. A single-element `controls` is
/// bit-identical to [`denoise_control`]; an empty `controls` reduces to [`denoise`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_multi_control_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    controls: &[ControlContext],
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        None,
        controls,
        None,
        None,
    )
}

/// Like [`denoise`] but injects the IP-Adapter image `tokens` (`[B, N, cross_attention_dim]`,
/// CFG-batched with a zeros uncond row) into every cross-attention at `scale` (sc-3059). Works on
/// the txt2img or img2img init; `scale = 0` ⇒ identical to [`denoise`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        None,
        &[],
        Some((tokens, scale)),
        None,
    )
}

/// Like [`denoise`] but runs the **InstantID** dual conditioning each step (sc-3113/3114): the
/// IdentityNet ControlNet (on the kps `control` image, cross-attended to `controlnet_encoder` = the
/// face tokens) injects its residuals, while the face IP `tokens` are injected into the UNet
/// cross-attention at `scale`. `tokens`/`controlnet_encoder` are typically the same CFG-batched
/// `[B, 16, cross_attention_dim]` face tokens. `scale = 0` + a `0`-scale control ⇒ identical to
/// [`denoise`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_control_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    control: &ControlContext,
    controlnet_encoder: &Array,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_ip_multi_control_with_preview(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        std::slice::from_ref(control),
        controlnet_encoder,
        tokens,
        scale,
    )
}

/// Like [`denoise_ip_control`] but runs **multiple** ControlNet branches and sums their residuals
/// before injection — the diffusers `MultiControlNetModel` rule (sc-3378). This is the engine for
/// InstantID pose mode (sc-3117): `controls = [IdentityNet(kps), OpenPose(skeleton)]`, each with its
/// own `conditioning_scale`, all sharing `controlnet_encoder` (the face tokens) as their
/// cross-attention conditioning — exactly as the vendored InstantID pipeline passes the same
/// `prompt_image_emb` to every sub-ControlNet. A single-element `controls` is bit-identical to
/// [`denoise_ip_control`]; an empty `controls` reduces to [`denoise_ip`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_ip_multi_control_with_preview(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    controls: &[ControlContext],
    controlnet_encoder: &Array,
    tokens: &Array,
    scale: f32,
) -> Result<Array> {
    denoise_core(
        d,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        None,
        controls,
        Some((tokens, scale)),
        Some(controlnet_encoder),
    )
}

/// One denoise-step ε-prediction: run each ControlNet branch and sum its residuals (the diffusers
/// `MultiControlNetModel` rule, sc-3378), then dispatch the U-Net forward on the `(ip, control)`
/// combination — IP+control (InstantID, sc-3113/3114/3117), IP-Adapter (sc-3059), ControlNet
/// (sc-3058), or the plain forward. This per-step body was triplicated verbatim across the
/// ancestral [`denoise_core`], the curated [`denoise_curated`], and the CFG++ [`denoise_cfgpp`]
/// loops; it lives here once (F-082). `cn_enc` is the ControlNet cross-attention conditioning
/// (the text `conditioning`, or the caller's override — InstantID's face tokens).
#[allow(clippy::too_many_arguments)]
fn forward_eps(
    unet: &UNet2DConditionModel,
    x_unet: &Array,
    timestep: f32,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    controls: &[ControlContext],
    cn_enc: &Array,
    ip: Option<(&Array, f32)>,
    plan: crate::plan::SdxlForwardPlan<'_>,
) -> Result<Array> {
    // Preserve the sampler's actual dtype, including promoted F32 img2img/noise state.
    // Header pricing follows this same operand; bounded execution must not change arithmetic.
    let x_cast = ios_unet_input(x_unet)?;
    mlx_gen::sampler::diagnostic_sampling_tensor(
        "sdxl",
        "forward-input-before-cast",
        0,
        x_unet.shape()[0] as u64,
        x_unet,
    );
    mlx_gen::sampler::diagnostic_sampling_tensor(
        "sdxl",
        "forward-input-after-cast-pending",
        0,
        x_cast.shape()[0] as u64,
        &x_cast,
    );
    mlx_gen::ios_quantum::diagnostic_metadata(
        "sdxl",
        "forward-cfg-half",
        &[
            ("schema", 1),
            ("half", plan.cfg_row.map_or(0, |row| row as u64 + 1)),
            ("control_branches", controls.len() as u64),
        ],
    );
    let x_unet = &x_cast;
    if let Some((tokens, _)) = ip {
        crate::quantum::diagnostic_owned_tensors(
            "ip-forward-token-owner",
            0,
            std::iter::once(tokens),
        );
        mlx_gen::sampler::diagnostic_sampling_tensor(
            "sdxl",
            "ip-forward-token-layout",
            0,
            tokens.shape()[0] as u64,
            tokens,
        );
    }
    let mut pending_control = crate::quantum::ControlMergeWork::default();
    let combined: Option<ControlResiduals> = {
        let mut acc: Option<ControlResiduals> = None;
        for (branch, cc) in controls.iter().enumerate() {
            mlx_gen::ios_quantum::diagnostic_metadata(
                "sdxl",
                "control-branch-enter",
                &[
                    ("schema", 1),
                    ("branch", branch as u64),
                    ("cfg_half", plan.cfg_row.map_or(0, |row| row as u64 + 1)),
                    ("rows", x_unet.shape()[0] as u64),
                    ("pending_down_logical_upper_bytes", pending_control.down),
                    ("pending_mid_logical_upper_bytes", pending_control.mid),
                ],
            );
            if let Some(previous) = &acc {
                previous.diagnostic_ownership("control-acc-before-branch", branch as u64);
            }
            let res = cc.controlnet.forward_planned(
                x_unet,
                &cc.cond_embed,
                timestep,
                cn_enc,
                pooled,
                time_ids,
                cc.scale,
                plan,
            )?;
            res.diagnostic_ownership("control-branch-return", branch as u64);
            acc = Some(match acc {
                None => res,
                Some(prev) => {
                    prev.diagnostic_ownership("control-merge-previous", branch as u64);
                    res.diagnostic_ownership("control-merge-current", branch as u64);
                    let summed = prev.add(&res)?;
                    pending_control.record_outputs(&summed.down, &summed.mid)?;
                    // At this point prev/res are still genuinely source-owned, while summed is
                    // a lazy DAG. Do not relabel its future payload as an already-held buffer.
                    summed.diagnostic_ownership("control-merge-pending", branch as u64);
                    summed
                }
            });
        }
        match acc {
            Some(res) => {
                res.diagnostic_ownership("control-before-cast", controls.len() as u64);
                let cast = ControlResiduals {
                    down: res.down.iter().map(ios_unet_input).collect::<Result<_>>()?,
                    mid: ios_unet_input(&res.mid)?,
                };
                cast.diagnostic_ownership("control-cast-pending", controls.len() as u64);
                Some(cast)
            }
            None => None,
        }
    };
    // One planned call for all four (control × ip) combinations. The historical four-way dispatch
    // over `forward_with_*` is preserved *in those wrappers*, which every non-ladder caller still
    // uses; routing the production denoise through the single planned entry point is what makes
    // rungs 3 and 4 reach EVERY advertised route — plain t2i, CFG, control, MultiControlNet, IP,
    // InstantID's ip+control — rather than the subset a per-route wiring would have covered.
    unet.forward_planned_with_control_work(
        x_unet,
        timestep,
        conditioning,
        pooled,
        time_ids,
        combined.as_ref(),
        ip,
        plan,
        pending_control,
    )
}

/// Row `k` of a CFG-batched `[cond, uncond]` tensor, or the tensor itself when it carries a single
/// row shared by both halves (a batch-1 conditioning broadcast across the guided pair).
fn cfg_row(a: &Array, k: i32) -> Result<Array> {
    if a.shape().first() == Some(&2) {
        Ok(a.take_axis(Array::from_slice(&[k], &[1]), 0)?)
    } else {
        Ok(a.clone())
    }
}

/// One guided step's `(eps_text, eps_neg)` pair under the request's CFG batching (sc-26268).
///
/// * [`CfgBatching::Batched`](mlx_gen::gen_core::CfgBatching::Batched) — the historical
///   `[x, x]` doubled-batch forward, split by row afterwards. Byte-identical to every earlier
///   release: it is the same op graph the three denoise loops each spelled inline.
/// * [`CfgBatching::Sequential`](mlx_gen::gen_core::CfgBatching::Sequential) — two batch-1
///   forwards over the one-row latent, conditional then unconditional. Each half receives the
///   pair's conditioning-only inputs whole (`conditioning`, `pooled`, `time_ids`, the ControlNet
///   encoder, the IP-Adapter tokens) and keeps its own row after projecting them
///   ([`SdxlForwardPlan::cfg_row`](crate::plan::SdxlForwardPlan::cfg_row)), so it is
///   byte-identical to its batched row; the ControlNet conditioning embedding, added to the latent
///   rather than projected, is sliced up front. Each half is evaluated before the next starts, so
///   the guided step holds one row's latent activations at a time — the high-water the shared
///   `CfgBatching` domain exists to offer a constrained device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn guided_eps_pair(
    unet: &UNet2DConditionModel,
    x_in: &Array,
    timestep: f32,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    controls: &[ControlContext],
    cn_enc: &Array,
    ip: Option<(&Array, f32)>,
    plan: crate::plan::SdxlForwardPlan<'_>,
) -> Result<(Array, Array)> {
    if plan.cfg_batching.is_batched() {
        let x_unet = concatenate_axis(&[x_in, x_in], 0)?;
        let eps = forward_eps(
            unet,
            &x_unet,
            timestep,
            conditioning,
            pooled,
            time_ids,
            controls,
            cn_enc,
            ip,
            plan,
        )?;
        let row = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0);
        return Ok((row(0)?, row(1)?));
    }
    let half = |k: i32| -> Result<Array> {
        let row_controls = controls
            .iter()
            .map(|cc| {
                Ok(ControlContext {
                    controlnet: cc.controlnet,
                    cond_embed: cfg_row(&cc.cond_embed, k)?,
                    scale: cc.scale,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The conditioning-only inputs go in WHOLE: the forward projects them at the pair's batch
        // and keeps row `k` (`SdxlForwardPlan::cfg_row`), so every kernel sees the row count the
        // batched forward gave it. Only `cond_embed` — added to the latent stem, never projected —
        // is sliced here.
        let eps = forward_eps(
            unet,
            x_in,
            timestep,
            conditioning,
            pooled,
            time_ids,
            &row_controls,
            cn_enc,
            ip,
            plan.with_cfg_row(Some(k)),
        )?;
        // Materialize this half before the other one is built, so the two forwards' activations
        // are never live together — the point of the sequential mode.
        eps.eval()?;
        Ok(eps)
    };
    let eps_text = half(0)?;
    let eps_neg = half(1)?;
    Ok((eps_text, eps_neg))
}

/// Forward evaluation seam for native discrete SDXL-family denoising (`PIPE-7`).
///
/// Generic over model prediction so Tier 0 `DriverSupport` conformance tests can drive
/// [`SdxlDiscreteDriver`] without loading weights or executing a learned forward.
pub trait SdxlEpsPredictor {
    /// Predict the guided (or unguided) noise residual ε for scaled model input `x_in` at
    /// `timestep` and 0-based `step_index`.
    fn predict_eps(&self, x_in: &Array, timestep: f32, step_index: usize) -> Result<Array>;
}

enum SamplerSlot<'a> {
    Borrowed(&'a dyn DiffusionSampler),
    Owned(Box<dyn DiffusionSampler + 'a>),
}

impl SamplerSlot<'_> {
    fn as_sampler(&self) -> &dyn DiffusionSampler {
        match self {
            Self::Borrowed(s) => *s,
            Self::Owned(s) => s.as_ref(),
        }
    }
}

/// Typed [`FamilyDriver`] for native MLX SDXL (ancestral, inpaint, IP, control, InstantID,
/// LCM, Lightning, Hyper) and native MLX Kolors (`KolorsEulerSampler`) discrete trajectories
/// (`SW-6966`, `PIPE-7`).
pub struct SdxlDiscreteDriver<'a, F> {
    sampler: SamplerSlot<'a>,
    predictor: F,
    support: DriverSupport,
    inpaint_quantum: bool,
}

impl<'a, F: SdxlEpsPredictor> SdxlDiscreteDriver<'a, F> {
    fn build_support(driver_id: &'static str, uses_global_rng: bool) -> DriverSupport {
        DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng,
            refusal_reasons: DriverRefusalReasons {
                renoise_from_clean:
                    "native discrete SDXL/Kolors driver does not re-noise from clean estimates; use curated shared driver",
                geometry_change:
                    "native discrete SDXL/Kolors driver does not support geometry-changing low-res start; use curated shared driver",
                read_only_boundary_estimate:
                    "native discrete SDXL/Kolors driver does not expose read-only boundary clean estimates",
                ..DriverRefusalReasons::default()
            },
        }
    }

    /// Build a discrete SDXL-family driver with explicit `driver_id` and `uses_global_rng` flag.
    pub fn new(
        driver_id: &'static str,
        sampler: &'a dyn DiffusionSampler,
        predictor: F,
        uses_global_rng: bool,
    ) -> Self {
        Self {
            sampler: SamplerSlot::Borrowed(sampler),
            predictor,
            support: Self::build_support(driver_id, uses_global_rng),
            inpaint_quantum: false,
        }
    }

    /// Build a discrete SDXL-family driver owning a boxed [`DiffusionSampler`] (used by `'static`
    /// Tier 0 conformance harnesses).
    pub fn new_owned(
        driver_id: &'static str,
        sampler: Box<dyn DiffusionSampler + 'a>,
        predictor: F,
        uses_global_rng: bool,
    ) -> Self {
        Self {
            sampler: SamplerSlot::Owned(sampler),
            predictor,
            support: Self::build_support(driver_id, uses_global_rng),
            inpaint_quantum: false,
        }
    }

    pub fn with_inpaint_quantum(mut self, inpaint_quantum: bool) -> Self {
        self.inpaint_quantum = inpaint_quantum;
        self
    }
}

impl<F: SdxlEpsPredictor> FamilyDriver<Array> for SdxlDiscreteDriver<'_, F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> gen_core::Result<gen_core::BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        state.enter_slice(&input)?;
        let mut latents = match input.state {
            gen_core::BoundaryState::XSigma(x)
            | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
            | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        let sampler = self.sampler.as_sampler();
        let steps = sampler.num_steps();
        for i in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            // Scale the latents into the model's input space: identity for the ancestral sampler
            // (which folds the renormalization into its step -> bit-identical to the pre-trait
            // loop), `x/sqrt(sigma^2+1)` for Lightning and Kolors Euler.
            let x_in = sampler
                .scale_model_input(&latents, i)
                .map_err(gen_core::Error::from)?;
            mlx_gen::sampler::diagnostic_sampling_tensor(
                "sdxl",
                "discrete-state",
                i as u64,
                1,
                &latents,
            );
            mlx_gen::sampler::diagnostic_sampling_tensor(
                "sdxl",
                "discrete-scaled-input",
                i as u64,
                1,
                &x_in,
            );
            let timestep = sampler.timestep(i);
            let next_timestep = if i + 1 < steps {
                sampler.timestep(i + 1)
            } else {
                0.0
            };
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: i,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: timestep,
                next_sigma: next_timestep,
            };
            hooks.observe(&info, &x_in)?;
            let eps = self
                .predictor
                .predict_eps(&x_in, timestep, i)
                .map_err(gen_core::Error::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            if self.support.uses_global_rng {
                state.next_draw_index();
            }
            mlx_gen::sampler::diagnostic_sampling_tensor(
                "sdxl",
                "discrete-epsilon",
                i as u64,
                1,
                &eps,
            );
            // sc-26270: the CFG combine above, the sampler update and the inpaint blend materialize
            // together in the per-step eval below — one guarded iOS quantum, checked before the
            // update's graph is built. A pass-through off device.
            mlx_gen::ios_quantum::before(
                crate::quantum::SAMPLER_STEP,
                crate::quantum::sampler_step_bytes(&latents, self.inpaint_quantum),
            )
            .map_err(gen_core::Error::from)?;
            latents = sampler
                .step(&eps, &latents, i)
                .map_err(gen_core::Error::from)?;
            mlx_gen::sampler::diagnostic_sampling_tensor(
                "sdxl",
                "discrete-update-pending",
                i as u64,
                1,
                &latents,
            );
            latents = hooks.post_step(&info, latents)?;
            hooks.materialize(&info, &latents)?;
            state.record_outer_step(i);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);
        Ok(gen_core::BoundaryState::XSigma(latents))
    }
}

struct ProductionSdxlPredictor<'a> {
    unet: &'a UNet2DConditionModel,
    conditioning: &'a Array,
    pooled: &'a Array,
    time_ids: &'a Array,
    controls: &'a [ControlContext<'a>],
    cn_enc: &'a Array,
    ip: Option<(&'a Array, f32)>,
    cfg: f32,
    plan: crate::plan::SdxlForwardPlan<'a>,
}

impl SdxlEpsPredictor for ProductionSdxlPredictor<'_> {
    fn predict_eps(&self, x_in: &Array, timestep: f32, _step_index: usize) -> Result<Array> {
        if self.cfg > 1.0 {
            let (eps_text, eps_neg) = guided_eps_pair(
                self.unet,
                x_in,
                timestep,
                self.conditioning,
                self.pooled,
                self.time_ids,
                self.controls,
                self.cn_enc,
                self.ip,
                self.plan,
            )?;
            // `eps_neg + cfg·(eps_text − eps_neg)` via shared gen-core `cfg` over `MlxLatentOps`
            // (`PIPE-8`, sc-7443): preserves fp16 compute dtype bit-identically.
            Ok(gen_core::guidance::cfg(
                &MlxLatentOps,
                &eps_text,
                &eps_neg,
                self.cfg,
            )?)
        } else {
            forward_eps(
                self.unet,
                x_in,
                timestep,
                self.conditioning,
                self.pooled,
                self.time_ids,
                self.controls,
                self.cn_enc,
                self.ip,
                self.plan,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn denoise_core(
    d: &Denoiser,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    inpaint: Option<&InpaintBlend>,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
) -> Result<Array> {
    denoise_core_planned(
        d,
        None,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        inpaint,
        controls,
        ip,
        control_encoder,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_core_planned(
    d: &Denoiser,
    run: Option<&gen_core::RenderRun<'_, '_>>,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    inpaint: Option<&InpaintBlend>,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
) -> Result<Array> {
    let steps = d.sampler.num_steps();
    let execution_schedule = d.sampler.execution_schedule();
    observe_denoise_policy(d.plan, cfg, controls, ip, inpaint.is_some());
    // A zero-step denoise (img2img at strength <= 1/steps) is a no-op: return the init latents
    // unchanged, matching the reference's `int(num_steps * strength)` loop count. Guards the
    // degenerate schedule and the sigma=0 ancestral step that would otherwise NaN.
    if steps == 0 {
        if let Some(run) = run {
            for phase in 0..run.plan().denoise_spans().len() {
                run.begin_denoise_phase(phase)?;
                run.end_denoise_phase(phase)?;
            }
        }
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "discrete-zero-denoise",
            &[
                ("schema", 1),
                ("steps", 0),
                (
                    "state_dtype",
                    mlx_gen::sampler::sampling_dtype_code(latents.dtype()),
                ),
                ("state_elements", latents.size() as u64),
            ],
        );
        observe_native_sampling(execution_schedule.as_ref(), latents.dtype());
        return Ok(latents);
    }
    // sc-2963 (rollout of sc-2957): fuse the UNet's SiLU activations via `mx.compile` — bit-exact in
    // fp16 (`max|Δ|=0`, compile_parity.rs), so it does not move the precision-load-bearing fp16
    // golden. Scoped + restored on drop by the RAII guard (F-006/F-007).
    let _compile_glue = crate::CompileGlueGuard::enable();
    let cfg_on = cfg > 1.0;
    let preview_sigmas: Vec<f32> = (0..=steps).rev().map(|step| step as f32).collect();
    let preview_counter = mlx_gen::preview::PreviewCounter::new(&preview_sigmas);
    let cn_enc = control_encoder.unwrap_or(conditioning);

    let driver = SdxlDiscreteDriver::new(
        "mlx_sdxl_discrete",
        d.sampler,
        ProductionSdxlPredictor {
            unet: d.unet,
            conditioning,
            pooled,
            time_ids,
            controls,
            cn_enc,
            ip,
            cfg,
            plan: d.plan,
        },
        // Serialize whole trajectory across threads when using MLX global RNG (B3 / PIPE-14).
        true,
    )
    .with_inpaint_quantum(inpaint.is_some());

    let spans: Vec<SliceSpec> = match run {
        Some(run) => {
            let plan_spans = run.plan().denoise_spans();
            let covered: usize = plan_spans.iter().map(|s| s.len()).sum();
            if covered == steps && !plan_spans.is_empty() {
                plan_spans
                    .into_iter()
                    .enumerate()
                    .map(|(phase, span)| SliceSpec {
                        pass: run.plan().pass_index(),
                        phase,
                        span,
                        transition: if phase == 0 {
                            SliceTransition::Start
                        } else {
                            SliceTransition::Continue
                        },
                        history: if phase == 0 {
                            DriverHistoryPolicy::Reset
                        } else {
                            DriverHistoryPolicy::Carry
                        },
                        geometry: Some(run.plan().phase_output_size(phase)),
                        request_boundary_estimate: false,
                    })
                    .collect()
            } else {
                vec![SliceSpec::full(steps)]
            }
        }
        None => vec![SliceSpec::full(steps)],
    };

    let mut state = DriverRunState::new();
    let mut on_observe = |info: &DriverStepInfo, x_in: &Array| -> gen_core::Result<()> {
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "discrete-epoch",
            &[
                ("schema", 1),
                ("evaluation", info.outer_step as u64),
                ("steps", steps as u64),
                ("cfg_on", u64::from(cfg_on)),
                ("cfg_batched", u64::from(d.plan.cfg_batching.is_batched())),
                ("inpaint", u64::from(inpaint.is_some())),
                ("control_branches", controls.len() as u64),
                ("ip", u64::from(ip.is_some())),
            ],
        );
        crate::preview::emit_nhwc_preview(
            preview,
            &preview_counter,
            &preview_sigmas,
            preview_sigmas[info.outer_step],
            x_in,
        );
        Ok(())
    };
    let mut on_post_step = |info: &DriverStepInfo, mut latents: Array| -> gen_core::Result<Array> {
        // Legacy inpaint blend (sc-3057): pin the kept region to the init noised to this step's
        // sigma, keep the repaint region freely denoised. No RNG draw -> ancestral stream intact.
        if let Some(b) = inpaint {
            latents = b
                .blend(&latents, info.outer_step)
                .map_err(gen_core::Error::from)?;
            mlx_gen::sampler::diagnostic_sampling_tensor(
                "sdxl",
                "discrete-blend-pending",
                info.outer_step as u64,
                1,
                &latents,
            );
        }
        Ok(latents)
    };
    let mut on_materialize = |info: &DriverStepInfo, latents: &Array| -> gen_core::Result<()> {
        // Force evaluation each step (the reference's per-step `mx.eval`). Beyond bounding the
        // lazy graph, this materializes the global-RNG state split between steps so the ancestral
        // noise stream is byte-identical to the reference (sc-2400 S5).
        latents
            .eval()
            .map_err(|e| gen_core::Error::from(Error::from(e)))?;
        mlx_gen::ios_quantum::finish(crate::quantum::SAMPLER_STEP, &[latents])
            .map_err(gen_core::Error::from)?;
        mlx_gen::sampler::diagnostic_sampling_tensor(
            "sdxl",
            "discrete-update-after-barrier",
            info.outer_step as u64,
            1,
            latents,
        );
        Ok(())
    };
    let mut hooks = StepHooks::new(on_progress);
    hooks.on_observe = Some(&mut on_observe);
    hooks.on_post_step = Some(&mut on_post_step);
    hooks.on_materialize = Some(&mut on_materialize);

    let seed = run.map_or(0, |r| r.plan().base_seed());
    let out = mlx_gen::run_mlx_driver_slices(
        &driver,
        spans,
        gen_core::BoundaryState::XSigma(latents),
        steps,
        seed,
        cancel,
        run,
        &mut state,
        &mut hooks,
        |_, boundary, _| Ok(boundary),
    )?;
    let latents = match out {
        gen_core::BoundaryState::XSigma(x)
        | gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
    };
    observe_native_sampling(execution_schedule.as_ref(), latents.dtype());
    Ok(latents)
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_curated(
    unet: &UNet2DConditionModel,
    sampler_name: Option<&str>,
    ms: &mlx_gen::DiscreteModelSampling,
    sigmas: &[f32],
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
) -> Result<Array> {
    denoise_curated_with_preview(
        unet,
        sampler_name,
        ms,
        sigmas,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        seed,
        cancel,
        on_progress,
        &PreviewSink::default(),
        controls,
        ip,
        control_encoder,
        crate::plan::SdxlForwardPlan::UNBOUNDED,
    )
}

/// Curated unified-sampler denoise (epic 7114, sc-7121) — the **additive** k-diffusion alternative to
/// SDXL's bespoke ancestral default. Drives any [`mlx_gen::Solver`] over a `DiscreteModelSampling`
/// (ε-prediction) and an [`mlx_gen::Scheduler`]-built σ schedule, through the shared
/// [`mlx_gen::run_curated_sampler`]. The ancestral default path (`denoise_core`) is left untouched —
/// this is selected only when the request names a curated sampler/scheduler, so the N1 default-parity
/// gate is byte-exact (the legacy loop is not entered).
///
/// Supports txt2img / img2img / ControlNet / IP-Adapter (the `controls` / `ip` dispatch mirrors
/// `denoise_core`). Inpaint is **not** offered here: its per-step mask blend has no post-step hook in
/// the callback-form `Sampler`, so it stays on the ancestral path (the same architectural boundary that
/// keeps Ideogram's interleaved inpaint bespoke).
///
/// The latents live in RAW k-diffusion σ-space (`x = ε·σ_max` at the start, `x₀ + ε·σ_start` for
/// img2img — built by the caller), and the U-Net input is `x/√(σ²+1)`
/// ([`mlx_gen::DiscreteModelSampling`]'s
/// `input_scale`), cast to the fp16 compute dtype inside the predict closure. The conditioning timestep
/// is the nearest training index (`mlx_gen::DiscreteModelSampling::timestep`) — ComfyUI's behaviour for a
/// discrete model under a curated solver.
#[allow(clippy::too_many_arguments)]
pub fn denoise_curated_with_preview(
    unet: &UNet2DConditionModel,
    sampler_name: Option<&str>,
    ms: &mlx_gen::DiscreteModelSampling,
    sigmas: &[f32],
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
    // Ladder rungs 3 and 4 for this request (SC-15525). The curated and CFG++ routes are production
    // denoise routes, so a bounded rung that reached only the ancestral loop would be
    // declared-but-unreachable on them.
    plan: crate::plan::SdxlForwardPlan<'_>,
) -> Result<Array> {
    let span = 0..sigmas.len().saturating_sub(1);
    denoise_curated_planned(
        unet,
        sampler_name,
        ms,
        sigmas,
        std::slice::from_ref(&span),
        None,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        seed,
        cancel,
        on_progress,
        preview,
        controls,
        ip,
        control_encoder,
        plan,
    )
}

fn transition_nhwc_clean_estimate(
    x0_low: &Array,
    source: gen_core::EstimateSource,
    low_res_plan: &gen_core::ResolvedLowResStartPlan,
    sigma_k: f32,
    ms: &dyn gen_core::sampling::ModelSampling,
    run: &gen_core::RenderRun<'_, '_>,
    seed: u64,
) -> Result<Array> {
    let target_dtype = x0_low.dtype();
    let shape = x0_low.shape();
    if shape.len() != 4 || shape[3] != 4 {
        return Err(Error::from(gen_core::Error::Unsupported(format!(
            "SDXL low_res_start transition expects NHWC [B, H, W, 4], got {shape:?}"
        ))));
    }
    let b_sz = shape[0] as usize;
    let low_h = shape[1] as usize;
    let nchw_low = mlx_gen::array::contiguous(
        &x0_low
            .transpose_axes(&[0, 3, 1, 2])?
            .as_dtype(mlx_rs::Dtype::Float32)?,
    )?;
    nchw_low.eval()?;
    let low_slice = nchw_low.as_slice::<f32>();
    let vae_spatial_scale = ((low_res_plan.low_height as usize) / low_h.max(1)).max(1);
    let base_seed = run.plan().base_seed();
    let output_index = seed.wrapping_sub(base_seed) as u32;
    let pass_index = run.plan().pass_index();
    let noised = gen_core::transition_clean_estimate_f32(
        gen_core::BoundaryState::CleanEstimate {
            latent: low_slice,
            source,
        },
        b_sz,
        4,
        vae_spatial_scale as u32,
        low_res_plan,
        sigma_k,
        ms,
        base_seed,
        output_index,
        pass_index,
        0,
    )?;
    let full_h = ((low_res_plan.full_height as usize) / vae_spatial_scale) as i32;
    let full_w = ((low_res_plan.full_width as usize) / vae_spatial_scale) as i32;
    let nchw_full = Array::from_slice(&noised, &[b_sz as i32, 4, full_h, full_w]);
    let nhwc_full = mlx_gen::array::contiguous(&nchw_full.transpose_axes(&[0, 2, 3, 1])?)?;
    if target_dtype == mlx_rs::Dtype::Float32 {
        Ok(nhwc_full)
    } else {
        Ok(nhwc_full.as_dtype(target_dtype)?)
    }
}

/// Planned multi-span curated unified-sampler denoise with NHWC-aware `low_res_start` transition.
#[allow(clippy::too_many_arguments)]
pub fn denoise_curated_planned(
    unet: &UNet2DConditionModel,
    sampler_name: Option<&str>,
    ms: &mlx_gen::DiscreteModelSampling,
    sigmas: &[f32],
    spans: &[std::ops::Range<usize>],
    phase_scope: Option<&gen_core::RenderRun<'_, '_>>,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
    plan: crate::plan::SdxlForwardPlan<'_>,
) -> Result<Array> {
    let _compile_glue = crate::CompileGlueGuard::enable();
    let cfg_on = cfg > 1.0;
    let cn_enc = control_encoder.unwrap_or(conditioning);
    let preview_counter = mlx_gen::preview::PreviewCounter::new(sigmas);
    let evaluation = std::cell::Cell::new(0u64);
    let predict_eps = |x_in: &Array, timestep: f32| -> Result<Array> {
        let x16 = x_in.as_dtype(mlx_rs::Dtype::Float16)?;
        let current = evaluation.get();
        evaluation.set(current + 1);
        diagnostic_curated_input(current, timestep, cfg_on, false, plan, x_in, &x16);
        let eps = if cfg_on {
            let (eps_text, eps_neg) = guided_eps_pair(
                unet,
                &x16,
                timestep,
                conditioning,
                pooled,
                time_ids,
                controls,
                cn_enc,
                ip,
                plan,
            )?;
            gen_core::guidance::cfg(&MlxLatentOps, &eps_text, &eps_neg, cfg)?
        } else {
            forward_eps(
                unet,
                &x16,
                timestep,
                conditioning,
                pooled,
                time_ids,
                controls,
                cn_enc,
                ip,
                plan,
            )?
        };
        mlx_gen::sampler::diagnostic_sampling_tensor("sdxl", "curated-epsilon", current, 1, &eps);
        Ok(eps)
    };
    let mut transition_fn = |_phase: usize,
                             boundary_step: usize,
                             ops: &MlxLatentOps,
                             ms_dyn: &dyn gen_core::sampling::ModelSampling,
                             out: &Array,
                             state: &mut gen_core::sampling::SolverRunState<MlxLatentOps>|
     -> Result<Array> {
        let Some(run) = phase_scope else {
            return Ok(out.clone());
        };
        let Some(low_res_plan) = run.plan().low_res_start().copied() else {
            return Ok(out.clone());
        };
        if boundary_step != low_res_plan.k {
            return Ok(out.clone());
        }
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let sigma_k = sigmas[boundary_step];
        let boundary_state = match low_res_plan.estimate_source {
            gen_core::EstimateSource::EndpointEvaluation => {
                let x0_low = gen_core::sampling::denoise(ops, ms_dyn, out, sigma_k, |xin, t| {
                    predict_eps(xin, t).map_err(gen_core::Error::from)
                })?;
                gen_core::BoundaryState::CleanEstimate {
                    latent: x0_low,
                    source: gen_core::EstimateSource::EndpointEvaluation,
                }
            }
            gen_core::EstimateSource::PreviousEvaluation => {
                let captured = state.take_captured_primary_estimate(boundary_step - 1)?;
                gen_core::BoundaryState::CleanEstimate {
                    latent: captured.latent,
                    source: gen_core::EstimateSource::PreviousEvaluation,
                }
            }
        };
        let (x0_low, source) = boundary_state.into_clean_estimate()?;
        let transitioned = transition_nhwc_clean_estimate(
            &x0_low,
            source,
            &low_res_plan,
            sigma_k,
            ms_dyn,
            run,
            seed,
        )?;
        state.note_geometry_change(
            (low_res_plan.low_width, low_res_plan.low_height),
            (low_res_plan.full_width, low_res_plan.full_height),
        );
        state.note_renoise_transition();
        state.apply_phase_boundary(gen_core::sampling::PhaseBoundary::Reset, true)?;
        Ok(transitioned)
    };
    let on_phase_boundary: Option<&mut mlx_gen::sampler::MlxPhaseTransitionFn<'_>> =
        if spans.len() > 1 {
            Some(&mut transition_fn)
        } else {
            None
        };
    let out = mlx_gen::run_curated_sampler_with_phase_transition_spans(
        mlx_gen::SamplerRun {
            sampler_name,
            sigmas,
            seed,
            cancel,
            on_progress,
        },
        ms,
        latents,
        spans,
        phase_scope,
        |latents, sigma| match phase_scope {
            Some(run) => crate::preview::emit_nhwc_ve_preview_for_plan(
                preview,
                run.plan(),
                &preview_counter,
                sigmas,
                sigma,
                latents,
            ),
            None => crate::preview::emit_nhwc_ve_preview(
                preview,
                &preview_counter,
                sigmas,
                sigma,
                latents,
            ),
        },
        predict_eps,
        on_phase_boundary,
    )?;
    observe_denoise_policy(plan, cfg, controls, ip, false);
    mlx_gen::image_execution::sampled_native_from_primary_seed(
        mlx_gen::sampler::canonical_solver(sampler_name, mlx_gen::Solver::Euler).name(),
        "discrete-epsilon-ve-sigma",
        out.dtype(),
        sigmas,
        "mlx-global-normal-initial-v1;curated-step=seed+golden-ratio*(step+1),wrapping-u64",
    );
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_cfgpp(
    unet: &UNet2DConditionModel,
    base_sampler_name: Option<&str>,
    ms: &mlx_gen::DiscreteModelSampling,
    sigmas: &[f32],
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
) -> Result<Array> {
    denoise_cfgpp_with_preview(
        unet,
        base_sampler_name,
        ms,
        sigmas,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        &PreviewSink::default(),
        controls,
        ip,
        control_encoder,
        crate::plan::SdxlForwardPlan::UNBOUNDED,
    )
}

/// CFG++ denoise (epic 7434 P3, sc-8256) — the [`denoise_curated`] twin that routes through
/// [`mlx_gen::run_cfgpp_sampler`]. Same U-Net forward + CFG combine, but it surfaces the unconditional
/// branch (`eps_neg`) alongside the guided combine so the CFG++ solver can land on the guided `x0` while
/// renoising from the unconditional one (Chung et al.). Selected only when `guidance_method == "cfg_pp"`,
/// a CFG++-compatible base sampler is chosen, AND CFG is active (`cfg > 1`) — CFG++ is meaningless
/// without a guidance gap (the caller guarantees `cfg_on`). `base_sampler_name` is the base solver
/// (`euler`/`ddim`/`dpmpp_2m`); the guided combine reuses the shared `gen_core::guidance::cfg`, so at
/// matched `cfg` the guided trajectory anchor is byte-identical to the plain path.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfgpp_with_preview(
    unet: &UNet2DConditionModel,
    base_sampler_name: Option<&str>,
    ms: &mlx_gen::DiscreteModelSampling,
    sigmas: &[f32],
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
    // Ladder rungs 3 and 4 for this request (SC-15525). The curated and CFG++ routes are production
    // denoise routes, so a bounded rung that reached only the ancestral loop would be
    // declared-but-unreachable on them.
    plan: crate::plan::SdxlForwardPlan<'_>,
) -> Result<Array> {
    let span = 0..sigmas.len().saturating_sub(1);
    denoise_cfgpp_planned(
        unet,
        base_sampler_name,
        ms,
        sigmas,
        std::slice::from_ref(&span),
        None,
        0,
        latents,
        conditioning,
        pooled,
        time_ids,
        cfg,
        cancel,
        on_progress,
        preview,
        controls,
        ip,
        control_encoder,
        plan,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn denoise_cfgpp_planned(
    unet: &UNet2DConditionModel,
    base_sampler_name: Option<&str>,
    ms: &mlx_gen::DiscreteModelSampling,
    sigmas: &[f32],
    spans: &[std::ops::Range<usize>],
    phase_scope: Option<&gen_core::RenderRun<'_, '_>>,
    seed: u64,
    latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    preview: &PreviewSink,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    control_encoder: Option<&Array>,
    plan: crate::plan::SdxlForwardPlan<'_>,
) -> Result<Array> {
    let _compile_glue = crate::CompileGlueGuard::enable();
    let cn_enc = control_encoder.unwrap_or(conditioning);
    let preview_counter = mlx_gen::preview::PreviewCounter::new(sigmas);
    let evaluation = std::cell::Cell::new(0u64);
    let predict_pair = |x_in: &Array, timestep: f32| -> Result<(Array, Array)> {
        let x16 = x_in.as_dtype(mlx_rs::Dtype::Float16)?;
        let current = evaluation.get();
        evaluation.set(current + 1);
        diagnostic_curated_input(current, timestep, true, true, plan, x_in, &x16);
        let (eps_text, eps_neg) = guided_eps_pair(
            unet,
            &x16,
            timestep,
            conditioning,
            pooled,
            time_ids,
            controls,
            cn_enc,
            ip,
            plan,
        )?;
        let guided = gen_core::guidance::cfg(&MlxLatentOps, &eps_text, &eps_neg, cfg)?;
        mlx_gen::sampler::diagnostic_sampling_tensor(
            "sdxl",
            "curated-epsilon",
            current,
            1,
            &guided,
        );
        mlx_gen::sampler::diagnostic_sampling_tensor(
            "sdxl",
            "cfgpp-unconditional-epsilon",
            current,
            1,
            &eps_neg,
        );
        Ok((guided, eps_neg))
    };
    let mut transition_fn = |_phase: usize,
                             boundary_step: usize,
                             ops: &MlxLatentOps,
                             ms_dyn: &dyn gen_core::sampling::ModelSampling,
                             out: &Array,
                             state: &mut gen_core::sampling::SolverRunState<MlxLatentOps>|
     -> Result<Array> {
        let Some(run) = phase_scope else {
            return Ok(out.clone());
        };
        let Some(low_res_plan) = run.plan().low_res_start().copied() else {
            return Ok(out.clone());
        };
        if boundary_step != low_res_plan.k {
            return Ok(out.clone());
        }
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let sigma_k = sigmas[boundary_step];
        let x0_low = gen_core::sampling::denoise(ops, ms_dyn, out, sigma_k, |xin, t| {
            Ok(predict_pair(xin, t)?.0)
        })?;
        let transitioned = transition_nhwc_clean_estimate(
            &x0_low,
            gen_core::EstimateSource::EndpointEvaluation,
            &low_res_plan,
            sigma_k,
            ms_dyn,
            run,
            seed,
        )?;
        state.note_geometry_change(
            (low_res_plan.low_width, low_res_plan.low_height),
            (low_res_plan.full_width, low_res_plan.full_height),
        );
        state.note_renoise_transition();
        state.apply_phase_boundary(gen_core::sampling::PhaseBoundary::Reset, true)?;
        Ok(transitioned)
    };
    let on_phase_boundary: Option<&mut mlx_gen::sampler::MlxPhaseTransitionFn<'_>> =
        if spans.len() > 1 {
            Some(&mut transition_fn)
        } else {
            None
        };
    let out = mlx_gen::run_cfgpp_sampler_with_phase_transition_spans(
        mlx_gen::SamplerRun {
            sampler_name: base_sampler_name,
            sigmas,
            seed,
            cancel,
            on_progress,
        },
        ms,
        latents,
        spans,
        phase_scope,
        |latents, sigma| match phase_scope {
            Some(run) => crate::preview::emit_nhwc_ve_preview_for_plan(
                preview,
                run.plan(),
                &preview_counter,
                sigmas,
                sigma,
                latents,
            ),
            None => crate::preview::emit_nhwc_ve_preview(
                preview,
                &preview_counter,
                sigmas,
                sigma,
                latents,
            ),
        },
        predict_pair,
        on_phase_boundary,
    )?;
    observe_denoise_policy(plan, cfg, controls, ip, false);
    mlx_gen::image_execution::sampled_native_from_primary_seed(
        &format!(
            "{}_cfg++",
            base_sampler_name
                .and_then(mlx_gen::Solver::from_name)
                .filter(|solver| mlx_gen::gen_core::sampling::base_supports_cfgpp(*solver))
                .unwrap_or(mlx_gen::Solver::Euler)
                .name()
        ),
        "discrete-epsilon-ve-sigma",
        out.dtype(),
        sigmas,
        "mlx-global-normal-initial-v1;curated-step=seed+golden-ratio*(step+1),wrapping-u64",
    );
    Ok(out)
}

fn observe_native_sampling(schedule: Option<&(&str, Vec<f32>)>, dtype: mlx_rs::Dtype) {
    match schedule {
        Some((encoding, values)) => mlx_gen::image_execution::sampled_native_from_primary_seed(
            encoding,
            "native-sampler-scalar-state",
            dtype,
            values,
            "mlx-global-normal-initial-v1;subsequent-draws=resolved-sampler-policy",
        ),
        None => mlx_gen::image_execution::unavailable(
            "native_sampler_schedule_unobserved",
            "the instantiated native sampler does not expose its consumed scalar state",
            "schedule",
        ),
    }
}

fn observe_denoise_policy(
    plan: crate::plan::SdxlForwardPlan<'_>,
    cfg: f32,
    controls: &[ControlContext],
    ip: Option<(&Array, f32)>,
    inpaint: bool,
) {
    mlx_gen::image_execution::policy_debug("sdxl-family.attention", &plan.attention.budget);
    mlx_gen::image_execution::policy_debug(
        "sdxl-family.window",
        &plan.window.map(|window| window.size),
    );
    mlx_gen::image_execution::policy_debug("sdxl-family.cfg_batching", &plan.cfg_batching);
    mlx_gen::image_execution::policy_debug("sdxl-family.inpaint", &inpaint);
    mlx_gen::image_execution::scalar("sdxl-family.guidance", cfg);
    mlx_gen::image_execution::policy_debug(
        "sdxl-family.control_scales",
        &controls
            .iter()
            .map(|control| control.scale.to_bits())
            .collect::<Vec<_>>(),
    );
    if let Some((_, scale)) = ip {
        mlx_gen::image_execution::scalar("sdxl-family.ip_scale", scale);
    }
}

/// Actual callback arguments on either curated route. Evaluation is the real callback ordinal,
/// including correctors/midpoints; solver-owned history/outer holders come from the existing
/// shared solver observations. These borrowed logical facts never establish storage disjointness.
fn diagnostic_curated_input(
    evaluation: u64,
    timestep: f32,
    cfg_on: bool,
    cfg_pp: bool,
    plan: crate::plan::SdxlForwardPlan<'_>,
    raw: &Array,
    compute: &Array,
) {
    mlx_gen::ios_quantum::diagnostic_metadata(
        "sdxl",
        "curated-callback",
        &[
            ("schema", 1),
            ("evaluation", evaluation),
            ("timestep_bits", timestep.to_bits() as u64),
            ("cfg_on", u64::from(cfg_on)),
            ("cfg_pp", u64::from(cfg_pp)),
            ("cfg_batched", u64::from(plan.cfg_batching.is_batched())),
        ],
    );
    mlx_gen::sampler::diagnostic_sampling_tensor("sdxl", "curated-model-input", evaluation, 1, raw);
    mlx_gen::sampler::diagnostic_sampling_tensor(
        "sdxl",
        "curated-compute-input",
        evaluation,
        1,
        compute,
    );
}

/// Seed the global RNG and sample the prior latents `[1, height/8, width/8, 4]` (NHWC, f32).
pub fn seeded_prior(sampler: &EulerSampler, seed: u64, width: u32, height: u32) -> Result<Array> {
    random::seed(seed)?;
    sampler.sample_prior(&[
        1,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
        LATENT_CHANNELS,
    ])
}

/// Shared resize/validate/layout for the SDXL image preprocessors: PIL-LANCZOS resize to the target
/// dims (no-op when already sized) via the core PIL-exact resampler (`resize_lanczos_u8`), apply the
/// per-pixel `normalize`, and lay out NHWC `[1, H, W, 3]` f32. `kind` names the image in the
/// buffer-size error. The init (`[-1,1]`) and control (`[0,1]`) preprocessors differ ONLY in
/// `normalize`, so the resize/validation logic lives here once (F-071).
fn preprocess_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
    kind: &str,
    normalize: impl Fn(f32) -> f32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    // A 0×0 image with an empty buffer passes the length check but reaches the core resize
    // `assert!` — give it the typed error `preprocess_clip_image_sized` already produces (F-084).
    if iw == 0 || ih == 0 {
        return Err(Error::Msg(format!(
            "sdxl {kind} image has a zero dimension ({iw}x{ih})"
        )));
    }
    if image.pixels.len()
        != mlx_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(Error::Msg(format!(
            "sdxl {kind} image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)?
    };
    let norm: Vec<f32> = resized.iter().map(|&v| normalize(v)).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// Preprocess an init image for img2img: PIL-LANCZOS resize to the target dims (no-op when already
/// sized), normalize `[0,255] → [-1,1]`, lay out NHWC `[1, H, W, 3]` f32 — the input the VAE encoder
/// expects.
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    preprocess_image(image, target_width, target_height, "init", |v| {
        2.0 * (v / 255.0) - 1.0
    })
}

/// Preprocess a ControlNet control image (sc-3058): LANCZOS resize to the target dims, normalize
/// `[0,255] → [0,1]` (the diffusers control image processor uses `do_normalize=False` ⇒ `[0,1]`, NOT
/// the `[-1,1]` of a VAE init), lay out NHWC `[1, H, W, 3]` f32.
pub fn preprocess_control_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    preprocess_image(image, target_width, target_height, "control", |v| v / 255.0)
}

/// img2img init latents: preprocess the image → VAE-encode mean `[1, h, w, 4]` (NHWC). The fork's
/// `generate_latents_from_image` uses the encoder mean (not a sample) as `x_0`.
pub fn encode_init_latents(
    vae: &Autoencoder,
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    encode_init_latents_tiled(vae, image, target_width, target_height, None, None)
}

/// [`encode_init_latents`] under the request's bounded-decode rung (sc-26270): when the rung
/// selected a native spatial tiling, the encode runs [`Autoencoder::encode_tiled`] at the same
/// pixel tile edge (the encoder twin of the tiled decode); without one it is the whole-image encode
/// — which on iOS is itself the bounded encode ([`Autoencoder::encode_mean`]).
pub fn encode_init_latents_tiled(
    vae: &Autoencoder,
    image: &Image,
    target_width: u32,
    target_height: u32,
    tiling: Option<&mlx_gen::tiling::TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Array> {
    let nhwc = preprocess_init_image(image, target_width, target_height)?;
    let latent = match tiling.and_then(|cfg| cfg.spatial.as_ref()) {
        Some(spatial) => vae.encode_tiled(&nhwc, spatial.tile_px, cancel)?,
        None => vae.encode_mean(&nhwc)?,
    };
    ios_unet_input(&latent)
}

/// Preserve the historical init/sampler/ControlNet arithmetic on every platform. A boundary
/// must not narrow a promoted F32 owner: latent rounding is amplified by the nonlinear decode.
/// Curated samplers still perform their original pre-CFG compute cast at their own call sites.
pub(crate) fn ios_unet_input(value: &Array) -> Result<Array> {
    Ok(value.clone())
}

/// Convert a VAE-decoded NHWC tensor `[1, H, W, 3]` (≈`[-1, 1]`) to an RGB8 [`Image`]:
/// `clip(x·0.5 + 0.5, 0, 1) · 255` (the vendored `StableDiffusion.decode` + txt2image recipe).
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    // sc-26270: the full-resolution clamp/scale/round chain is one guarded iOS quantum; a
    // pass-through off device. Counted: ONE f32 image-sized buffer — the first scale's output (or
    // a half-precision VAE's lazy f32 output cast, which that scale then donates into); every
    // later op of the chain donates its single-consumer operand (measured). Plus its page.
    let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
    let x = mlx_gen::ios_quantum::run(
        crate::quantum::DECODED_TO_IMAGE,
        crate::quantum::decoded_to_image_bytes(decoded),
        || {
            let half = scalar(0.5);
            let x = add(&multiply(decoded, &half)?, &half)?;
            let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
            Ok(round(&multiply(&x, scalar(255.0))?, 0)?)
        },
    )?;
    let sh = x.shape();
    // One image per call (the pipeline loops with B==1): reject B>1 instead of silently keeping only
    // batch 0, and size in usize / flatten via -1 to avoid the u32/i32 product overflow (F-053).
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "sdxl decoded_to_image: expected batch size 1, got {}",
            sh[0]
        )));
    }
    let (h, w, c) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
    let n = h * w * c;
    let flat = x.reshape(&[-1])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Decode final latents `[1, h, w, 4]` to an RGB8 image.
///
/// When `pid` is `Some` (epic 7840, sc-7848 — the model was loaded with `LoadSpec::pid` and the
/// request set `use_pid`), the latent is decoded by the **PiD super-resolving student** (4× SR)
/// instead of the SDXL VAE. The SDXL `latents` (NHWC `[1,h,w,4]`) are already the
/// `0.13025`-normalized tensor the `sdxl` student trained on — the exact tensor `vae.decode`
/// consumes (zero-transform handoff), so we just relayout: NHWC `[1,h,w,4]` → NCHW `[1,4,h,w]` for
/// the LQ adapter; the student returns NCHW `[1,3,4H,4W]`, transposed back to NHWC for
/// [`decoded_to_image`] (which, like the VAE output, expects channels-last). Both decoders return
/// pixels in `≈[-1,1]`, so the `x·0.5+0.5` mapping in [`decoded_to_image`] is identical.
pub fn decode_image(
    vae: &Autoencoder,
    latents: &Array,
    pid: Option<&dyn LatentDecoder>,
) -> Result<Image> {
    decode_image_tiled(vae, latents, pid, None, None)
}

/// [`decode_image`] with ladder rung 2's optional native-VAE tiling (SC-15525).
///
/// `tiling` is the **native** decode geometry resolved by
/// [`memory_strategy::decode_tiling`](crate::memory_strategy) — `None` for the exact single-pass
/// decode, which is what every pre-SC-15525 caller gets through [`decode_image`].
///
/// The **PiD branch deliberately ignores `tiling`**, and that is not a gap: the student plans and
/// executes its own tiling inside `mlx_gen_pid::mint_planned_decoder_with_tiling`, against its own
/// disjoint edge domain. Applying a *native* edge on top would tile the same decode twice at two
/// different geometries.
///
/// **On SDXL that branch is unreachable with `tiling` set, and the reason is worth stating rather
/// than implying.** `memory_strategy::decode_tiling_for_contract` validates a bounded decode against
/// the route it runs (sc-26268, `mlx_gen_pid::DecodeRoutes`): a `use_pid` request reaches here with
/// `tiling` set only at the student's own measured geometry, which the student's `decode_tiled`
/// executes; a native edge on the PiD route is refused before generate. A PiD request that does
/// **not** set the flag is completely unaffected and still gets the student's auto-planning, which is
/// the pre-SC-15525 behaviour.
///
/// `cancel` gives the decode a cancellation point it has never had: it is a dominant fraction of an
/// SDXL render's wall clock, and the shared tile loop checks between tiles.
pub fn decode_image_tiled(
    vae: &Autoencoder,
    latents: &Array,
    pid: Option<&dyn LatentDecoder>,
    tiling: Option<&mlx_gen::tiling::TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Image> {
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(Error::Canceled);
    }
    let native = SdxlLatentDecoder::new(vae);
    let decoder: &dyn LatentDecoder = pid.unwrap_or(&native);
    mlx_gen::ensure_decoder_compatible(Some(&mlx_gen::gen_core::SDXL_LATENT_SPACE), decoder)?;
    let nchw = latents.transpose_axes(&[0, 3, 1, 2])?;
    let decoded = match tiling {
        Some(cfg) => decoder.decode_tiled(&nchw, cfg, cancel)?,
        None => decoder.decode(&nchw)?,
    };
    decoded_to_image(&decoded.transpose_axes(&[0, 2, 3, 1])?)
}

/// Render one preview sample (sc-5637) from the **in-progress training adapter** already installed
/// on `unet`: seeded txt2img prior → Euler-Ancestral CFG denoise → VAE decode → [`Image`]. A stripped
/// [`Sdxl::generate_impl`](crate::model) txt2img: builds the ancestral sampler + seeded prior exactly
/// as inference does, runs the plain [`denoise`] with the configured `guidance`. `conditioning`/`pooled`
/// are the pre-encoded **CFG batch** (`[2, …]` = positive then empty-negative), so the denoise loop's
/// classifier-free guidance has both streams. No progress/cancel plumbing — the caller drives cadence.
#[cfg(feature = "training")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_sample(
    unet: &UNet2DConditionModel,
    vae: &Autoencoder,
    base_sampler: &EulerSampler,
    conditioning: &Array,
    pooled: &Array,
    guidance: f32,
    seed: u64,
    (width, height): (u32, u32),
    steps: usize,
) -> Result<Image> {
    random::seed(seed)?;
    let latent_shape = [1, (height / 8) as i32, (width / 8) as i32, 4];
    let prior = base_sampler.sample_prior(&latent_shape)?;
    let sampler = AncestralEuler::new(base_sampler, steps.max(1), base_sampler.max_time())?;
    let time_ids = text_time_ids(pooled.shape()[0]);
    let d = Denoiser::new(unet, &sampler);
    let latents = denoise(
        &d,
        prior,
        conditioning,
        pooled,
        &time_ids,
        guidance,
        &CancelFlag::default(),
        &mut |_| {},
    )?;
    // Training preview — always the native VAE decode (no PiD overlay in the trainer's render path).
    decode_image(vae, &latents, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{AlphaSchedule, LightningSampler};

    /// `cfg_row` takes row `k` of a CFG-batched `[cond, uncond]` tensor and passes a single shared
    /// row through untouched — the slicing every sequential-CFG input (conditioning, pooled,
    /// time ids, ControlNet embedding/encoder, IP tokens) goes through.
    #[test]
    fn cfg_row_splits_a_guided_pair_and_shares_a_single_row() {
        let _cpu = crate::test_device::cpu_only();
        let pair = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(cfg_row(&pair, 0).unwrap().as_slice::<f32>(), &[1.0, 2.0]);
        assert_eq!(cfg_row(&pair, 1).unwrap().as_slice::<f32>(), &[3.0, 4.0]);
        assert_eq!(cfg_row(&pair, 0).unwrap().shape(), &[1, 2]);
        let shared = Array::from_slice(&[5.0f32, 6.0], &[1, 2]);
        assert_eq!(cfg_row(&shared, 1).unwrap().as_slice::<f32>(), &[5.0, 6.0]);
    }

    /// The tiny-U-Net inputs for one guided step: a latent and a distinct `[cond, uncond]` pair in
    /// every CFG-batched input, so a mode that mixed up rows would fail.
    fn guided_inputs() -> (Array, Array, Array, Array) {
        use crate::test_support::{TINY_CONTEXT_DIM, TINY_POOLED_DIM};
        let ramp = |shape: &[i32], phase: f32| {
            let n = shape.iter().product::<i32>();
            let v = (0..n)
                .map(|i| ((i as f32 + phase) * 0.173).sin() * 0.5)
                .collect::<Vec<_>>();
            Array::from_slice(&v, shape)
        };
        (
            ramp(&[1, 8, 8, 4], 0.0),
            ramp(&[2, 5, TINY_CONTEXT_DIM], 11.0),
            ramp(&[2, TINY_POOLED_DIM], 29.0),
            text_time_ids(2),
        )
    }

    /// The devices × dtypes the sequential-CFG identity is held on: the CPU at f32 (where a lone
    /// `[1, K]` GEMV and row 0 of the `[2, K]` GEMM differ in the last ulps) and the default
    /// device at f16 (Metal: `[1, K]` vs `[2, K]` and the 77- vs 154-row text-memory projections
    /// pick different kernels, ~1e-3 of peak apart). Both differed before sc-26268's fix (M5 Max:
    /// 1.8e-6 and 3.2e-3 of peak; 0 and 0 after).
    fn cfg_identity_cells() -> [(bool, mlx_rs::Dtype); 2] {
        [
            (true, mlx_rs::Dtype::Float32),
            (false, mlx_rs::Dtype::Float16),
        ]
    }

    /// The tiny U-Net at `dtype` (weights cast before construction, like a real f16 load).
    fn tiny_unet_at(dtype: mlx_rs::Dtype) -> UNet2DConditionModel {
        use crate::test_support::{tiny_unet_config, tiny_unet_weights};
        let mut w = tiny_unet_weights(7).unwrap();
        w.cast_all(dtype).unwrap();
        UNet2DConditionModel::from_weights(&w, &tiny_unet_config()).unwrap()
    }

    /// Run `f` on the CPU (`cpu = true`) or the process default device.
    fn on_device<T>(cpu: bool, f: impl FnOnce() -> T) -> T {
        let _guard = cpu.then(crate::test_device::cpu_only);
        f()
    }

    fn rel(a: &[u32], b: &[u32]) -> f32 {
        let (a, b): (Vec<f32>, Vec<f32>) = (
            a.iter().map(|v| f32::from_bits(*v)).collect(),
            b.iter().map(|v| f32::from_bits(*v)).collect(),
        );
        let peak = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        a.iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
            / peak
    }

    /// Whether this backend computes every **latent-path** op of the tiny U-Net row-exactly
    /// between a one-row and a two-row batch — each dense projection of its 4×4 (16-token)
    /// attention level and each 3×3/1×1 convolution of its two levels, with random data at
    /// `dtype`. It probes MLX itself, never the code under test, so a defect in the CFG split
    /// cannot lower the bar it is held to. True for both cells on Apple M5; a CPU BLAS that blocks
    /// by row count (hosted CI's M1) makes it false.
    fn latent_path_is_row_exact(dtype: mlx_rs::Dtype) -> bool {
        let rnd = |shape: &[i32], seed: u64| {
            let key = random::key(seed).unwrap();
            random::normal::<f32>(shape, None, None, Some(&key))
                .unwrap()
                .as_dtype(dtype)
                .unwrap()
        };
        let pair_row0_matches = |one: Array, two: Array| {
            let two = two.take_axis(Array::from_slice(&[0], &[1]), 0).unwrap();
            bits(&one) == bits(&two)
        };
        let linears = [(64, 64), (64, 256), (256, 64)].into_iter().all(|(k, n)| {
            let (x1, w) = (rnd(&[1, 16, k], 1), rnd(&[n, k], 2));
            let x2 = concatenate_axis(&[&x1, &rnd(&[1, 16, k], 3)], 0).unwrap();
            let wt = w.transpose().unwrap();
            pair_row0_matches(
                mlx_rs::ops::matmul(&x1, &wt).unwrap(),
                mlx_rs::ops::matmul(&x2, &wt).unwrap(),
            )
        });
        // (spatial, in, out, kernel, stride): conv_in/out, the resnets' 3×3s and 1×1 shortcuts
        // over both levels' (skip-concatenated) widths, and the stride-2 downsampler.
        let convs = [
            (8, 4, 32, 3, 1),
            (8, 32, 32, 3, 1),
            (8, 32, 32, 3, 2),
            (8, 32, 4, 3, 1),
            (4, 32, 64, 3, 1),
            (4, 64, 64, 3, 1),
            (4, 128, 64, 3, 1),
            (4, 96, 64, 3, 1),
            (8, 96, 32, 3, 1),
            (8, 64, 32, 3, 1),
            (4, 32, 64, 1, 1),
            (4, 128, 64, 1, 1),
            (4, 96, 64, 1, 1),
            (8, 96, 32, 1, 1),
            (8, 64, 32, 1, 1),
        ]
        .into_iter()
        .all(|(hw, cin, cout, k, stride)| {
            let (x1, w) = (rnd(&[1, hw, hw, cin], 4), rnd(&[cout, k, k, cin], 5));
            let x2 = concatenate_axis(&[&x1, &rnd(&[1, hw, hw, cin], 6)], 0).unwrap();
            let pad = k / 2;
            let conv = |x: &Array| {
                mlx_rs::ops::conv2d(x, &w, (stride, stride), (pad, pad), None, None).unwrap()
            };
            pair_row0_matches(conv(&x1), conv(&x2))
        });
        linears && convs
    }

    /// The sequential-CFG identity check: **byte for byte** wherever the backend's latent path is
    /// row-exact ([`latent_path_is_row_exact`] — both cells on Apple M5, which is where the pre-fix
    /// code was measured RED). Elsewhere the latent GEMMs legitimately differ by row count (the
    /// reassociation `CfgBatching` admits), so the halves are held to that class instead: 1e-5 of
    /// peak at f32, 1e-2 at f16 — the backend then cannot tell a conditioning-path deviation from
    /// its own reassociation, and the eprintln says so.
    fn assert_cfg_identity(
        label: &str,
        dtype: mlx_rs::Dtype,
        exact: bool,
        batched: &[u32],
        sequential: &[u32],
    ) {
        let delta = rel(batched, sequential);
        if exact {
            assert!(batched == sequential, "{label}: moved by {delta:.3e}");
        } else {
            let bound = if dtype == mlx_rs::Dtype::Float32 {
                1e-5
            } else {
                1e-2
            };
            eprintln!(
                "{label}: this backend's latent path is not row-exact; held to {bound:.0e} of peak \
                 (measured {delta:.3e}) instead of byte identity"
            );
            assert!(
                delta <= bound,
                "{label}: moved by {delta:.3e} (bound {bound:.0e})"
            );
        }
    }

    fn bits(a: &Array) -> Vec<u32> {
        let a = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
    }

    /// One guided step's pair under `batching`, with optional IP-Adapter tokens.
    fn guided_pair_bits(
        unet: &UNet2DConditionModel,
        dtype: mlx_rs::Dtype,
        ip_tokens: Option<&Array>,
        batching: gen_core::CfgBatching,
    ) -> (Vec<u32>, Vec<u32>) {
        let (x, conditioning, pooled, time_ids) = guided_inputs();
        let at = |a: &Array| a.as_dtype(dtype).unwrap();
        let (x, conditioning, pooled) = (at(&x), at(&conditioning), at(&pooled));
        let tokens = ip_tokens.cloned();
        let ip = tokens.as_ref().map(|t| (t, 0.7_f32));
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(batching);
        let (text, neg) = (
            crate::test_device::record_model_inference_device(),
            guided_eps_pair(
                unet,
                &x,
                500.0,
                &conditioning,
                &pooled,
                &time_ids,
                &[],
                &conditioning,
                ip,
                plan,
            ),
        )
            .1
            .unwrap();
        assert_eq!(text.shape(), x.shape());
        assert_eq!(neg.shape(), x.shape());
        (bits(&text), bits(&neg))
    }

    /// sc-26268: **sequential CFG** (two batch-1 forwards) is **byte-identical** to the batched
    /// `[cond, uncond]` forward — each half is exactly its batched row: the per-row split is right
    /// (the conditional half is row 0, the unconditional half row 1), nothing couples the rows,
    /// and every conditioning-only projection (time/`text_time` embedding, each resnet's
    /// `time_emb_proj`, each cross-attention `to_k`/`to_v`) runs at the pair's row count
    /// ([`SdxlForwardPlan::cfg_row`](crate::plan::SdxlForwardPlan::cfg_row)).
    ///
    /// Before that fix the halves sliced their conditioning to one row first, so those projections
    /// ran a `[1, K]` (and 77-row) product where the batched pair ran `[2, K]` (154 rows): ~1e-6 of
    /// peak on the CPU at f32, ~3e-3 at f16 on Metal — which CFG × 7 and the ancestral noise grew
    /// into a 255/255 pixel on the real SDXL q4 render (`low_memory_compositions_q4`).
    #[ignore = "model inference; explicit model-proof workflow"]
    #[test]
    fn sequential_cfg_is_byte_identical_to_the_batched_guided_pair() {
        for (cpu, dtype) in cfg_identity_cells() {
            on_device(cpu, || {
                let unet = tiny_unet_at(dtype);
                let exact = latent_path_is_row_exact(dtype);
                let pair = |batching| guided_pair_bits(&unet, dtype, None, batching);
                let (bt, bn) = pair(gen_core::CfgBatching::Batched);
                let (st, sn) = pair(gen_core::CfgBatching::Sequential);
                let label = |half: &str| format!("cpu={cpu} {dtype:?}: {half} half");
                assert_cfg_identity(&label("conditional"), dtype, exact, &bt, &st);
                assert_cfg_identity(&label("unconditional"), dtype, exact, &bn, &sn);
                // Non-vacuity: the two halves genuinely differ, so a mode that returned one row
                // twice (or swapped them) cannot pass the comparisons above.
                assert!(
                    bt != bn,
                    "the guided pair must carry two distinct conditionings"
                );
            });
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED pipeline::tests::sequential_cfg_is_byte_identical_to_the_batched_guided_pair");
    }

    /// The same identity with the **IP-Adapter** decoupled cross-attention installed: its
    /// `to_k_ip`/`to_v_ip` over the `[image tokens, zeros]` pair are conditioning-only projections
    /// too, and a sequential half must run them at the pair's batch as well.
    #[ignore = "model inference; explicit model-proof workflow"]
    #[test]
    fn sequential_cfg_is_byte_identical_with_the_ip_adapter() {
        use crate::test_support::TINY_CONTEXT_DIM;
        for (cpu, dtype) in cfg_identity_cells() {
            on_device(cpu, || {
                let mut unet = tiny_unet_at(dtype);
                // One `[hidden, ctx]` pair per cross-attention, in install order (down → up →
                // mid): four, all at the tiny U-Net's 64-wide attention level.
                let ramp = |rows: i32, phase: f32| {
                    let n = rows * TINY_CONTEXT_DIM;
                    let v = (0..n)
                        .map(|i| ((i as f32 + phase) * 0.311).cos() * 0.2)
                        .collect::<Vec<_>>();
                    Array::from_slice(&v, &[rows, TINY_CONTEXT_DIM])
                        .as_dtype(dtype)
                        .unwrap()
                };
                let pairs = [64; 4]
                    .iter()
                    .enumerate()
                    .map(|(i, &h)| (ramp(h, i as f32), ramp(h, 50.0 + i as f32)))
                    .collect();
                unet.install_ip_adapter(pairs).unwrap();
                // A single image token: its per-half projection would be a `[1, ctx]` GEMV where
                // the pair runs a `[2, ctx]` GEMM — the row count whose kernels round differently
                // on both devices, so a half that projected only its own row is observable.
                let tokens = concatenate_axis(
                    &[
                        &ramp(1, 3.0).reshape(&[1, 1, TINY_CONTEXT_DIM]).unwrap(),
                        &Array::zeros::<f32>(&[1, 1, TINY_CONTEXT_DIM])
                            .unwrap()
                            .as_dtype(dtype)
                            .unwrap(),
                    ],
                    0,
                )
                .unwrap();
                let exact = latent_path_is_row_exact(dtype);
                let pair = |batching| guided_pair_bits(&unet, dtype, Some(&tokens), batching);
                let batched = pair(gen_core::CfgBatching::Batched);
                let sequential = pair(gen_core::CfgBatching::Sequential);
                let label = |half: &str| format!("cpu={cpu} {dtype:?}: IP {half} half");
                assert_cfg_identity(
                    &label("conditional"),
                    dtype,
                    exact,
                    &batched.0,
                    &sequential.0,
                );
                assert_cfg_identity(
                    &label("unconditional"),
                    dtype,
                    exact,
                    &batched.1,
                    &sequential.1,
                );
                // Non-vacuity: the IP branch is live (it moves the conditional half).
                let plain = guided_pair_bits(&unet, dtype, None, gen_core::CfgBatching::Batched);
                assert!(plain.0 != batched.0, "the IP branch must change the output");
            });
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED pipeline::tests::sequential_cfg_is_byte_identical_with_the_ip_adapter");
    }

    /// The whole ancestral denoise loop under sequential CFG reproduces the batched loop's final
    /// latent **byte for byte** — the production route (`denoise_core`) actually consumes the
    /// plan's batching, and the identity survives the guidance combine and the ancestral noise.
    #[ignore = "model inference; explicit model-proof workflow"]
    #[test]
    fn sequential_cfg_denoise_is_byte_identical_to_the_batched_denoise() {
        for (cpu, dtype) in cfg_identity_cells() {
            on_device(cpu, || {
                let unet = tiny_unet_at(dtype);
                let (x, conditioning, pooled, time_ids) = guided_inputs();
                let at = |a: &Array| a.as_dtype(dtype).unwrap();
                let (x, conditioning, pooled) = (at(&x), at(&conditioning), at(&pooled));
                let inner = EulerSampler::new_with_dtype(
                    &crate::config::DiffusionConfig::sdxl_base(),
                    true,
                    dtype,
                )
                .unwrap();
                let run = |batching| {
                    random::seed(42).unwrap();
                    let sampler = AncestralEuler::new(&inner, 2, inner.max_time()).unwrap();
                    let plan = crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(batching);
                    let d = Denoiser::with_plan(&unet, &sampler, plan);
                    let out = (
                        crate::test_device::record_model_inference_device(),
                        denoise(
                            &d,
                            x.clone(),
                            &conditioning,
                            &pooled,
                            &time_ids,
                            5.0,
                            &CancelFlag::new(),
                            &mut |_| {},
                        ),
                    )
                        .1
                        .unwrap();
                    bits(&out)
                };
                let batched = run(gen_core::CfgBatching::Batched);
                let sequential = run(gen_core::CfgBatching::Sequential);
                let exact = latent_path_is_row_exact(dtype);
                let label = format!("cpu={cpu} {dtype:?}: denoise");
                assert_cfg_identity(&label, dtype, exact, &batched, &sequential);
            });
        }

        println!("\nMODEL_INFERENCE_BODY_COMPLETED pipeline::tests::sequential_cfg_denoise_is_byte_identical_to_the_batched_denoise");
    }

    /// F-071: the init and control preprocessors share the resize/validate/layout helper and differ
    /// only in normalization — init maps `[0,255]→[-1,1]`, control maps `[0,255]→[0,1]`. Use a
    /// same-size image so the resampler is a no-op and the per-pixel normalization is what's asserted.
    #[test]
    fn preprocess_normalizations_and_layout() {
        // 2×1 RGB: pixel A = (255,255,255), pixel B = (0,0,0).
        let img = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };

        let init = preprocess_init_image(&img, 2, 1).unwrap();
        assert_eq!(init.shape(), &[1, 1, 2, 3]); // NHWC [1, H, W, 3]
        let s = init.as_slice::<f32>();
        assert_eq!(s[0], 1.0); // 255 → +1
        assert_eq!(s[3], -1.0); // 0 → −1

        let control = preprocess_control_image(&img, 2, 1).unwrap();
        let c = control.as_slice::<f32>();
        assert_eq!(c[0], 1.0); // 255 → 1
        assert_eq!(c[3], 0.0); // 0 → 0
    }

    /// The shared buffer-size validation rejects a mismatched pixel buffer and names the image kind
    /// (so the two wrappers stay distinguishable in the error).
    #[test]
    fn preprocess_rejects_wrong_buffer_with_kind() {
        let bad = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 8], // not 4·4·3
        };
        let e = preprocess_init_image(&bad, 4, 4).unwrap_err().to_string();
        assert!(e.contains("init"), "init error should name the kind: {e}");
        let e = preprocess_control_image(&bad, 4, 4)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("control"),
            "control error should name the kind: {e}"
        );
    }

    /// F-084: a 0×0 request image (empty buffer passes the length check) must be a typed error
    /// instead of reaching the core resize `assert!`.
    #[test]
    fn preprocess_rejects_zero_dimension_image() {
        let zero = Image {
            width: 0,
            height: 0,
            pixels: vec![],
        };
        let e = preprocess_init_image(&zero, 8, 8).unwrap_err().to_string();
        assert!(e.contains("zero dimension"), "unexpected error: {e}");
        let e = preprocess_control_image(&zero, 8, 8)
            .unwrap_err()
            .to_string();
        assert!(e.contains("zero dimension"), "unexpected error: {e}");
    }

    /// The acceleration change must not be a validation-only admission. This CPU-only test crosses
    /// the narrow production seam used by `UNet2DConditionModel::forward_planned` to inject a real
    /// [`ControlResiduals::mid`] tensor, then hands the resulting prediction to the actual
    /// [`LightningSampler`]. A live residual must move the next latent; absent and zero residuals
    /// must be exact identities. It therefore fails if the production injection seam drops its
    /// `control` argument, while needing neither SDXL weights nor accelerator hardware.
    #[test]
    fn production_control_injection_materially_changes_lightning_step() {
        let schedule = AlphaSchedule::scaled_linear(1_000, 0.00085, 0.012);
        let sampler = LightningSampler::new(&schedule, 1_000, 4, mlx_rs::Dtype::Float16);
        let latents = Array::from_slice(&[1.0f32, -2.0], &[1, 2]);
        let plain_epsilon = Array::from_slice(&[0.25f32, -0.5], &[1, 2]);
        let live = ControlResiduals {
            down: vec![Array::from_slice(&[0.125f32, -0.25], &[1, 2])],
            mid: Array::from_slice(&[0.75f32, 0.25], &[1, 2]),
        };
        let zero = ControlResiduals {
            down: vec![Array::from_slice(&[0.0f32, 0.0], &[1, 2])],
            mid: Array::from_slice(&[0.0f32, 0.0], &[1, 2]),
        };

        let base_skip = Array::from_slice(&[2.0f32, -1.0], &[1, 2]);
        let mut absent_skip = vec![base_skip.clone()];
        crate::unet::inject_control_down_residuals(&mut absent_skip, None).unwrap();
        let mut zero_skip = vec![base_skip.clone()];
        crate::unet::inject_control_down_residuals(&mut zero_skip, Some(&zero)).unwrap();
        let mut controlled_skip = vec![base_skip.clone()];
        crate::unet::inject_control_down_residuals(&mut controlled_skip, Some(&live)).unwrap();
        assert_eq!(
            base_skip.as_slice::<f32>(),
            absent_skip[0].as_slice::<f32>()
        );
        assert_eq!(base_skip.as_slice::<f32>(), zero_skip[0].as_slice::<f32>());
        assert_ne!(
            base_skip.as_slice::<f32>(),
            controlled_skip[0].as_slice::<f32>(),
            "a nonzero ControlNet down residual must alter the U-Net skip stream"
        );

        let absent_epsilon =
            crate::unet::inject_control_mid_residual(plain_epsilon.clone(), None).unwrap();
        let zero_epsilon =
            crate::unet::inject_control_mid_residual(plain_epsilon.clone(), Some(&zero)).unwrap();
        let controlled_epsilon =
            crate::unet::inject_control_mid_residual(plain_epsilon.clone(), Some(&live)).unwrap();

        let plain = sampler.step(&plain_epsilon, &latents, 0).unwrap();
        let absent = sampler.step(&absent_epsilon, &latents, 0).unwrap();
        let zero = sampler.step(&zero_epsilon, &latents, 0).unwrap();
        let controlled = sampler.step(&controlled_epsilon, &latents, 0).unwrap();
        assert_eq!(plain.as_slice::<f32>(), absent.as_slice::<f32>());
        assert_eq!(plain.as_slice::<f32>(), zero.as_slice::<f32>());
        assert_ne!(
            plain.as_slice::<f32>(),
            controlled.as_slice::<f32>(),
            "a nonzero ControlNet residual must alter the Lightning trajectory"
        );
    }

    #[test]
    fn transition_nhwc_clean_estimate_materializes_transposed_nhwc_planes() {
        let req = mlx_gen::GenerationRequest {
            prompt: "nhwc low_res_start transition".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            seed: Some(42),
            sampler: Some("euler".into()),
            scheduler: Some("karras".into()),
            low_res_start: Some(gen_core::LowResStartSpec::new(2, Some(0.5))),
            ..Default::default()
        };
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        let plan = crate::model::compile_sdxl(&spec, &req)
            .unwrap()
            .expect("compiled sdxl plan");
        let low_res_plan = *plan.low_res_start().unwrap();
        let sigma_k = plan.denoise_sigmas()[low_res_plan.k];
        let cfg = crate::DiffusionConfig::sdxl_base();
        let alpha = AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
        let ms = mlx_gen::gen_core::sampling::DiscreteModelSampling::sdxl(&alpha);

        // Build a spatially varying, channel-distinct [1, 64, 64, 4] NHWC f32 fixture and its
        // matching planar [1, 4, 64, 64] NCHW vector. Without `contiguous`, `transpose_axes`
        // followed by `as_dtype(Float32)` and `as_slice` reads the physical NHWC layout as NCHW.
        let low_h = 64usize;
        let low_w = 64usize;
        let mut nhwc_vec = vec![0.0f32; low_h * low_w * 4];
        let mut nchw_vec = vec![0.0f32; 4 * low_h * low_w];
        for y in 0..low_h {
            for x in 0..low_w {
                for c in 0..4 {
                    let val = (c as f32) * 10.0 + (y as f32) * 0.1 + (x as f32) * 0.01;
                    nhwc_vec[(y * low_w + x) * 4 + c] = val;
                    nchw_vec[c * (low_h * low_w) + y * low_w + x] = val;
                }
            }
        }
        let x0_nhwc = Array::from_slice(&nhwc_vec, &[1, low_h as i32, low_w as i32, 4]);
        let run = gen_core::RenderRun::new(&plan, &req.cancel, None);
        let got_nhwc = transition_nhwc_clean_estimate(
            &x0_nhwc,
            gen_core::EstimateSource::EndpointEvaluation,
            &low_res_plan,
            sigma_k,
            &ms,
            &run,
            plan.base_seed(),
        )
        .unwrap();
        assert_eq!(got_nhwc.shape(), &[1, 128, 128, 4]);
        let got_nchw =
            mlx_gen::array::contiguous(&got_nhwc.transpose_axes(&[0, 3, 1, 2]).unwrap()).unwrap();
        got_nchw.eval().unwrap();

        let expected_nchw = gen_core::transition_clean_estimate_f32(
            gen_core::BoundaryState::CleanEstimate {
                latent: &nchw_vec,
                source: gen_core::EstimateSource::EndpointEvaluation,
            },
            1,
            4,
            8,
            &low_res_plan,
            sigma_k,
            &ms,
            plan.base_seed(),
            0,
            plan.pass_index(),
            0,
        )
        .unwrap();
        assert_eq!(got_nchw.as_slice::<f32>(), expected_nchw.as_slice());
    }

    #[test]
    fn cfgpp_dpmpp_2m_preserves_multistep_history_across_inert_split() {
        use mlx_gen::gen_core::sampling::{cfgpp_denoise, cfgpp_sampler_for, CpuLatentOps, Solver};

        let req = mlx_gen::GenerationRequest {
            prompt: "a lighthouse in a storm".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            seed: Some(7),
            guidance: Some(7.0),
            sampler: Some("dpmpp_2m".into()),
            scheduler: Some("karras".into()),
            guidance_method: Some("cfg_pp".into()),
            ..Default::default()
        };
        let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        let unsplit_plan = crate::model::compile_sdxl(&spec, &req)
            .unwrap()
            .expect("compiled sdxl cfg_pp plan");
        assert_eq!(unsplit_plan.denoise_spans(), vec![0..8]);
        let split_plan = unsplit_plan.clone().with_inert_split_at(4).unwrap();
        assert_eq!(split_plan.denoise_spans(), vec![0..4, 4..8]);
        assert!(split_plan.low_res_start().is_none());

        let cfg = crate::DiffusionConfig::sdxl_base();
        let alpha = AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
        let ms = mlx_gen::gen_core::sampling::DiscreteModelSampling::sdxl(&alpha);
        let init_vals = [0.4f32, -0.25, 0.15, 0.8];
        let init = Array::from_slice(&init_vals, &[1, 1, 1, 4]);

        let run_for_plan = |plan: &gen_core::RenderPlan| -> Vec<f32> {
            let cancel = mlx_gen::CancelFlag::default();
            let run = gen_core::RenderRun::new(plan, &cancel, None);
            run.encode(|| Ok::<_, Error>(())).unwrap();
            run.begin_rendering().unwrap();
            let mut progress = |_| {};
            let mut transition_fn = |_phase: usize,
                                     boundary_step: usize,
                                     _ops: &MlxLatentOps,
                                     _ms_dyn: &dyn gen_core::sampling::ModelSampling,
                                     out: &Array,
                                     state: &mut gen_core::sampling::SolverRunState<
                MlxLatentOps,
            >|
             -> Result<Array> {
                if run
                    .plan()
                    .low_res_start()
                    .is_some_and(|lr| boundary_step == lr.k)
                {
                    state.note_renoise_transition();
                    state.apply_phase_boundary(gen_core::sampling::PhaseBoundary::Reset, true)?;
                }
                Ok(out.clone())
            };
            let spans = plan.denoise_spans();
            let on_phase_boundary: Option<&mut mlx_gen::sampler::MlxPhaseTransitionFn<'_>> =
                if spans.len() > 1 {
                    Some(&mut transition_fn)
                } else {
                    None
                };
            let out = mlx_gen::run_cfgpp_sampler_with_phase_transition_spans(
                mlx_gen::SamplerRun {
                    sampler_name: Some(plan.sampler()),
                    sigmas: plan.denoise_sigmas(),
                    seed: plan.base_seed(),
                    cancel: &cancel,
                    on_progress: &mut progress,
                },
                &ms,
                init.clone(),
                &spans,
                Some(&run),
                |_, _| {},
                |x_in, t| {
                    x_in.eval()?;
                    let xs = x_in.as_slice::<f32>();
                    let tf = (t * 1e-3).sin();
                    let guided: Vec<f32> = xs
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| 0.65 * v + 0.12 * tf + (i as f32) * 0.03)
                        .collect();
                    let uncond: Vec<f32> = xs
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| 0.35 * v - 0.07 * tf - (i as f32) * 0.02)
                        .collect();
                    Ok((
                        Array::from_slice(&guided, &[1, 1, 1, 4]),
                        Array::from_slice(&uncond, &[1, 1, 1, 4]),
                    ))
                },
                on_phase_boundary,
            )
            .unwrap();
            let out = run
                .decode_phase(|| {
                    out.eval()?;
                    Ok::<_, Error>(out.as_slice::<f32>().to_vec())
                })
                .unwrap();
            run.finish_rendering();
            out
        };

        let unsplit_out = run_for_plan(&unsplit_plan);
        let split_out = run_for_plan(&split_plan);
        assert_eq!(
            split_out, unsplit_out,
            "inert phase split must preserve CFG++ DPM++ 2M multistep history"
        );

        // Sanity check: resetting CFG++ solver state at step 4 (as per-span `sample` did) diverges.
        let sigmas = split_plan.denoise_sigmas();
        let ops = CpuLatentOps;
        let sampler = cfgpp_sampler_for::<CpuLatentOps>(Solver::Dpmpp2m).unwrap();
        let predict_cpu = |x_in: &Vec<f32>, t: f32| -> gen_core::Result<(Vec<f32>, Vec<f32>)> {
            let tf = (t * 1e-3).sin();
            let guided = x_in
                .iter()
                .enumerate()
                .map(|(i, &v)| 0.65 * v + 0.12 * tf + (i as f32) * 0.03)
                .collect();
            let uncond = x_in
                .iter()
                .enumerate()
                .map(|(i, &v)| 0.35 * v - 0.07 * tf - (i as f32) * 0.02)
                .collect();
            Ok((guided, uncond))
        };
        let mut denoise_fn =
            |x: &Vec<f32>, sigma: f32| cfgpp_denoise(&ops, &ms, x, sigma, predict_cpu);
        let mid = sampler
            .sample(&ops, &mut denoise_fn, init_vals.to_vec(), &sigmas[0..=4])
            .unwrap();
        let reset_out = sampler
            .sample(&ops, &mut denoise_fn, mid, &sigmas[4..=8])
            .unwrap();
        assert_ne!(
            reset_out, unsplit_out,
            "per-span reset must differ from preserved multistep history"
        );
    }

    #[test]
    fn sdxl_discrete_driver_passes_tier0_conformance_without_weights() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        struct FakeLinearPredictor;
        impl SdxlEpsPredictor for FakeLinearPredictor {
            fn predict_eps(&self, x_in: &Array, timestep: f32, step_index: usize) -> Result<Array> {
                let scale = scalar(0.05 + 0.0001 * timestep + 0.01 * (step_index as f32));
                Ok(x_in.multiply(&scale)?)
            }
        }

        // Pure host/MLX arithmetic DiffusionSampler (no learned weights) mirroring deterministic
        // discrete step progression so the conformance harness checks split-parity bit-for-bit.
        struct FakeDiscreteSampler {
            timesteps: Vec<f32>,
        }
        impl DiffusionSampler for FakeDiscreteSampler {
            fn num_steps(&self) -> usize {
                self.timesteps.len()
            }
            fn timestep(&self, i: usize) -> f32 {
                self.timesteps[i]
            }
            fn scale_initial_noise(&self, noise: &Array) -> Result<Array> {
                Ok(noise.clone())
            }
            fn step(
                &self,
                model_output: &Array,
                latents: &Array,
                step_index: usize,
            ) -> Result<Array> {
                let dt = scalar(0.1 / ((step_index + 1) as f32));
                Ok(latents.subtract(&model_output.multiply(&dt)?)?)
            }
        }

        let report = verify_driver_conformance(ConformanceCase {
            total_steps: 6,
            split_at: 3,
            seed: 42,
            base_geometry: (1024, 1024),
            alt_geometry: (512, 512),
            make_driver: || {
                let sampler = FakeDiscreteSampler {
                    timesteps: vec![999.0, 833.0, 666.0, 500.0, 333.0, 166.0],
                };
                Box::new(SdxlDiscreteDriver::new_owned(
                    "sdxl_discrete_tier0_test",
                    Box::new(sampler),
                    FakeLinearPredictor,
                    false,
                ))
            },
            make_initial: || Array::from_slice(&[0.5f32, -0.25, 0.75, -0.1], &[1, 1, 1, 4]),
            latents_equal: |a, b| {
                a.eval().unwrap();
                b.eval().unwrap();
                a.as_slice::<f32>() == b.as_slice::<f32>()
            },
        })
        .expect("SdxlDiscreteDriver must pass Tier 0 conformance");
        assert!(report.verified_inert_split);
        assert!(report.verified_carry_refusal_or_support);
        assert!(report.verified_reset_preserves_draws);
        assert!(report.verified_boundary_estimate_readonly_or_refusal);
        assert!(report.verified_renoise_and_resample_guards);
        assert!(report.verified_cancellation);
    }
}
