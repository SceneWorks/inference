//! sc-7781 — the parameterized **SDXL-family** LoRA/LoKr training backbone, shared by
//! [`SdxlTrainer`](super::SdxlTrainer) (dual-CLIP) and the Kolors trainer (ChatGLM3). Kolors *is* an
//! SDXL-base U-Net under a different text encoder, so its trainer was an ~85–90% line-for-line clone
//! of the SDXL one — same private-fn roster, same `train_impl` control flow, same tests — differing
//! only at a handful of injection points. This module collapses that duplication: the whole
//! prepare→cache→train→save lifecycle lives once in [`train_family`], and each crate supplies its
//! deltas through the [`SdxlFamilyHooks`] trait.
//!
//! Everything generic here was **moved verbatim** from the two trainers (no behavioral change — the
//! committed grad-parity gates in each crate are the safety case), with the per-family pieces routed
//! through hooks:
//!   1. **Prompt encoding** — SDXL dual-CLIP vs Kolors ChatGLM3 ([`SdxlFamilyHooks::encode_prompt`]);
//!      the preview-sample CFG batch via [`SdxlFamilyHooks::encode_sample_cfg`].
//!   2. **Micro-conditioning `time_ids`** — SDXL's hardcoded `[512,512,0,0,512,512]` vs Kolors'
//!      real-resolution `(H,W,0,0,H,W)` ([`SdxlFamilyHooks::time_ids`]).
//!   3. **Noise / objective** — SDXL's renormalized k-diffusion sigma noising (float table-index `t`)
//!      vs Kolors' direct DDPM `√ᾱ_t·x0 + √(1−ᾱ_t)·noise` (integer `t`). The float-vs-integer split is
//!      surfaced explicitly as [`TrainTimestep`]; the noise call hides behind
//!      [`SdxlFamilyHooks::add_noise`] and the draw behind [`SdxlFamilyHooks::sample_timestep`].
//!   4. **Memory-fit coefficients** — each family's fitted peak-GB curve
//!      ([`SdxlFamilyHooks::peak_gb`]) + the error-string family label ([`SdxlFamilyHooks::label`]).
//!   5. **Preview render** — SDXL's Euler-Ancestral vs Kolors' leading-Euler render
//!      ([`SdxlFamilyHooks::render_sample`]); freeing the text encoder(s) after caching via
//!      [`SdxlFamilyHooks::free_text_encoders`].
//!
//! The U-Net and VAE are the **same** concrete SDXL types in both crates (Kolors reuses
//! `mlx_gen_sdxl::{UNet2DConditionModel, Autoencoder}`), so they are threaded as plain parameters
//! rather than hidden behind the trait.

use std::path::Path;

use mlx_gen::gen_core::BucketSchedule;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, LoraParams, TrainAdapter,
};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxDriver, Parameterization, PerceptualPath, StepPlan,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::tae::TinyDecoderSpec;
use mlx_gen::{
    Image, NetworkType, Result, TrainOptimizer, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{broadcast_to, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::pipeline::encode_init_latents;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// The SDXL-family reconstructs its LoKr delta at **f32** (the f32-everywhere merge path); training
/// must match so the adapter round-trips through the inference loader. Shared by both families.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// PEFT save-key prefix for the LoRA adapter — what `peft.save_pretrained()` / the SceneWorks
/// backends emit, and what the SDXL loader's PEFT key classifier (`adapters::classify_key`) expects.
/// The Kolors U-Net IS the SDXL `UNet2DConditionModel`, so it shares this prefix.
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// The default SDXL-family attention LoRA targets — the suffixes `to_q`/`to_k`/`to_v`/`to_out.0` the
/// torch trainers use, suffix-matched across the UNet attention modules exactly as PEFT's
/// `LoraConfig(target_modules=…)` does.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// The sampled training timestep, surfacing the one fiddly cross-family split: SDXL noises in the
/// vendored **sigma** space at a *float* table-index, while Kolors noises with direct DDPM
/// `alphas_cumprod` at an *integer* index. [`SdxlFamilyHooks::add_noise`] consumes this verbatim; the
/// U-Net's sinusoidal time embedding consumes [`Self::unet_time`] (Kolors' integer cast to f32).
#[derive(Clone, Copy, Debug)]
pub enum TrainTimestep {
    /// SDXL: a float sigma-table index in `[1, max_time]`; fed verbatim to both the noiser and the
    /// U-Net's time embedding.
    Sigma(f32),
    /// Kolors: an integer `alphas_cumprod` index in `[0, num_train_timesteps)`; the U-Net consumes it
    /// as `t as f32`.
    Index(usize),
}

impl TrainTimestep {
    /// The f32 timestep the U-Net's sinusoidal embedding consumes.
    pub fn unet_time(self) -> f32 {
        match self {
            TrainTimestep::Sigma(s) => s,
            TrainTimestep::Index(i) => i as f32,
        }
    }
}

/// The per-family injection points the generic [`train_family`] backbone routes through. Everything
/// *not* on this trait is identical across the SDXL family and lives in [`train_family`].
pub trait SdxlFamilyHooks {
    /// Family id prefix for error strings + log lines, e.g. `"sdxl"` / `"kolors"`.
    fn label(&self) -> &'static str;

    /// Encode one caption → `(conditioning, pooled)` for a single (B=1) cached training item.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)>;

    /// Encode one preview-sample prompt into the **CFG batch** (`[2, …]` = positive then
    /// empty-negative) the preview render's classifier-free guidance needs. Returns the f32 (or
    /// encoder-native) conditioning/pooled; [`train_family`] casts to the compute dtype.
    fn encode_sample_cfg(&self, prompt: &str) -> Result<(Array, Array)>;

    /// Free the text encoder(s) after the caching phase (SDXL frees both CLIPs; Kolors frees the
    /// ChatGLM3) — the 32 GB-Mac headroom lever. After this, [`Self::encode_prompt`] /
    /// [`Self::encode_sample_cfg`] must not be called again.
    fn free_text_encoders(&mut self);

    /// Micro-conditioning `time_ids` for the given latent `edge`, `batch` rows.
    fn time_ids(&self, batch: i32, edge: u32) -> Array;

    /// Sample a training timestep for this step, deterministic in `seed`.
    fn sample_timestep(&self, seed: u64) -> Result<TrainTimestep>;

    /// Add noise at the sampled timestep — hides the VE-sigma vs DDPM-`alpha_bar` convention (and the
    /// f32/usize split carried by [`TrainTimestep`]).
    fn add_noise(&self, x0: &Array, noise: &Array, t: TrainTimestep) -> Result<Array>;

    /// Fitted dense first-step peak-GB curve vs the latent pixel count `p = (edge/8)²`.
    fn peak_gb(&self, p: f64, bf16: bool) -> f64;

    /// The cumulative signal fraction `ᾱ` [`Self::add_noise`] uses at `t`
    /// (`noisy = √ᾱ·x0 + √(1−ᾱ)·noise`) — what recovers the ε-prediction's x0 estimate for the
    /// decoded-x0 auxiliary losses (epic 2123 E8).
    fn alpha_bar(&self, t: TrainTimestep) -> Result<f32>;

    /// `t` as the unit noise level the auxiliary-loss timestep window speaks (`0` clean … `1` pure
    /// noise).
    fn noise_level(&self, t: TrainTimestep) -> f32;

    /// The family timestep at unit noise level `level` (the inverse of [`Self::noise_level`],
    /// rounded onto the family's discrete table) — where an aux-only step trains.
    fn timestep_at(&self, level: f32) -> TrainTimestep;

    /// Render one preview sample from the **in-progress adapter** already installed on `unet`:
    /// seeded prior → CFG denoise → VAE decode. `conditioning`/`pooled` are the pre-encoded CFG batch;
    /// `dtype` is the trainer compute dtype (used by Kolors' sampler; SDXL ignores it).
    #[allow(clippy::too_many_arguments)]
    fn render_sample(
        &self,
        unet: &UNet2DConditionModel,
        vae: &Autoencoder,
        conditioning: &Array,
        pooled: &Array,
        guidance: f32,
        seed: u64,
        edge: u32,
        steps: usize,
        dtype: Dtype,
    ) -> Result<Image>;
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted UNet paths by suffix-matching them against the routable Linear surface — the same match
/// PEFT's `LoraConfig(target_modules=…)` does over the UNet attention modules.
///
/// The surface is chosen to match each adapter kind's **inference consumption** (so nothing trains
/// that no inference path reads, and the adapter round-trips cleanly):
///   * **LoRA** → the **complete** surface ([`UNet2DConditionModel::lora_target_paths_complete`]),
///     which `LoraCoverage::Complete` (the SDXL-family `model::load` default) merges — down / **mid** /
///     up attention. Matches the torch PEFT suffix-match (which hits `mid_block` too).
///   * **LoKr** → the **vendored** surface ([`UNet2DConditionModel::lora_target_paths`]), down / up
///     attention only: the SDXL LoKr loader keeps `mid_block` out (sc-2640), so a `mid_block` LoKr
///     factor would be skipped at load. Training to the vendored surface keeps train/inference in
///     lock-step.
pub fn resolve_target_paths(unet: &UNet2DConditionModel, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        DEFAULT_TARGET_SUFFIXES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    let surface = match cfg.network_type {
        NetworkType::Lora => unet.lora_target_paths_complete(),
        NetworkType::Lokr => unet.lora_target_paths(),
    };
    surface
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// One forward+backward over the trainable adapter factors: build the noisy input at the sampled
/// timestep (via [`SdxlFamilyHooks::add_noise`]), inject `params` (LoRA or LoKr), run the U-Net,
/// regress the predicted `eps` toward the unit `noise`, return `(loss, grads)`. The plain diffusion
/// step — [`compute_step_loss_grads`] with no auxiliary loss.
///
/// `dtype` is the training compute dtype (sc-4941): for bf16 the noisy latent / conditioning / pooled
/// are cast to bf16 at entry (the U-Net weights were cast once in [`train_family`]) and the LoRA
/// factors / LoKr delta are reconstructed at bf16 inside the traced install — so the whole U-Net graph
/// runs bf16 with no silent f32 re-promotion. The noise target, loss, and grads stay f32
/// (master-weights pattern).
#[allow(clippy::too_many_arguments)]
pub fn compute_loss_grads<H: SdxlFamilyHooks>(
    hooks: &H,
    unet: &mut UNet2DConditionModel,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Array,
    pooled: &Array,
    time_ids: &Array,
    t: TrainTimestep,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    checkpoint_targets: Option<Vec<String>>,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        hooks,
        unet,
        params,
        adapter,
        alpha,
        rank,
        x0,
        cond,
        pooled,
        time_ids,
        t,
        noise,
        mae,
        mask_weight,
        dtype,
        checkpoint_targets,
        None,
    )?;
    Ok((losses.total, grads))
}

/// The per-step loss breakdown [`compute_step_loss_grads`] returns (epic 2123 E8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLosses {
    /// The differentiated step loss.
    pub total: f32,
    /// The diffusion (ε-regression) term, `None` on an aux-only step (it contributed zero).
    pub diffusion: Option<f32>,
    /// The weighted aux-loss term, `None` when no aux loss contributed this step.
    pub aux: Option<f32>,
}

/// One aux-loss step's view of the trainer's [`PerceptualPath`] (epic 2123 E8).
pub struct AuxStep<'a> {
    pub path: &'a PerceptualPath,
    pub plan: &'a StepPlan,
    /// The step's reference key — its (item, bucket) cache entry.
    pub entry: usize,
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed (it contributes zero) and the loss is the weighted perceptual
/// term on the ε-prediction's x0 estimate `x0 = (x_t − √(1−ᾱ)·ε)/√ᾱ` (ᾱ from
/// [`SdxlFamilyHooks::alpha_bar`] at `t`; the NHWC SDXL latent transposed to the decoder's NCHW, in
/// the same `scaling_factor`-normalized space TAESDXL decodes). With `aux = None` (or a
/// diffusion-only plan with no aux loss) the traced graph is exactly the pre-epic-2123 one. Both the
/// dense and the block-checkpointed forwards carry the aux term (the x0 recovery reads the shared
/// `eps` output).
#[allow(clippy::too_many_arguments)]
pub fn compute_step_loss_grads<H: SdxlFamilyHooks>(
    hooks: &H,
    unet: &mut UNet2DConditionModel,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Array,
    pooled: &Array,
    time_ids: &Array,
    t: TrainTimestep,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    checkpoint_targets: Option<Vec<String>>,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    // The renormalized DDPM noisy input at the sampled timestep — the family hook hides the SDXL
    // sigma-space vs Kolors `alphas_cumprod` convention. The epsilon target is the unit `noise`.
    let noisy_f32 = hooks.add_noise(x0, noise, t)?;
    let noisy = noisy_f32.as_dtype(dtype)?;
    let t_f = t.unet_time();
    let target = noise.clone(); // f32 — the loss is computed in f32 (eps promotes on subtract)
    let mask_weight = mask_weight.cloned();
    let (cond, pooled, time_ids) = (
        cond.as_dtype(dtype)?,
        pooled.as_dtype(dtype)?,
        time_ids.clone(),
    );
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    // Reconstruct the LoKr delta at the compute dtype so its residual matches the bf16 activation
    // stream; the SAVED factors stay f32 (the inference round-trip dtype) — `save` writes the raw
    // factor arrays, not this delta.
    let lokr_dtype = if dtype == Dtype::Float32 {
        LOKR_DTYPE
    } else {
        dtype
    };
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    // ᾱ only matters for the x0 recovery of an aux term.
    let alpha_bar = if aux_on { hooks.alpha_bar(t)? } else { 1.0 };
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL adapters: under block checkpointing the mid block + embedders train through
        // these on the (non-checkpointed) dense path, while each down/up block's adapters are replaced
        // inside its checkpoint segment by the explicit-input factors — so installing them here costs
        // nothing on the checkpointed path.
        adapter.install_as(unet, &p, alpha, rank, lora_dtype, lokr_dtype)?;
        let eps = match &checkpoint_targets {
            Some(tp) => unet
                .forward_block_checkpointed(&noisy, t_f, &cond, &pooled, &time_ids, tp, &p, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => unet
                .forward(&noisy, t_f, &cond, &pooled, &time_ids)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diffusion = if diffusion_on {
            let diff = subtract(&eps, &target)?;
            // MSE / MAE on the ε residual, subject-mask weighted when on (sc-24828) — reduces to a
            // 0-d scalar (grad requires a scalar cotangent). Both the dense and block-checkpointed
            // forwards land here.
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // x0 estimate in f32 from the ε prediction, NHWC → the decoder's NCHW.
                let x0_hat = Parameterization::Epsilon { alpha_bar }
                    .recover_x0(&noisy_f32, &eps.as_dtype(Dtype::Float32)?)
                    .and_then(|x| nchw(&x))
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.entry, &x0_hat)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        // Only the first output is differentiated; the other two are reported terms.
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: val[0].item::<f32>(),
        diffusion: diffusion_on.then(|| val[1].item::<f32>()),
        aux: aux_on.then(|| val[2].item::<f32>()),
    };
    Ok((losses, grads))
}

/// The SDXL family's x0 decoder for the shared aux-loss builder (epic 2123 E8): TAESDXL
/// (`madebyollin/taesdxl`, the SDXL 4-channel latent API — SDXL, Illustrious and Kolors all use the
/// SDXL VAE, and their cached latent is the `scaling_factor`-normalized one TAESDXL decodes).
pub fn taesdxl_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Tiny {
        name: "TAESDXL",
        config: TinyDecoderSpec::taesdxl(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder
/// ([`mlx_gen_perceptual::build_perceptual_path`]): `None` when no aux loss is enabled (the default
/// — nothing is loaded and every step is the plain diffusion step); otherwise TAESDXL + every
/// enabled loss. A missing or unreadable checkpoint is a named error prefixed `"{label} trainer"`.
pub fn load_perceptual_path(label: &str, cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    let label = format!("{label} trainer");
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label: &label,
            decoder: taesdxl_decoder(),
            // SDXL, Illustrious and Kolors share the SDXL VAE latent space.
            latent_lpips: Some(mlx_gen::gen_core::train::LatentLpipsFamily::Sdxl),
        },
    )
}

/// Extra training memory (GB) the enabled aux losses add at the bucketed `edge` — TAESDXL (when a
/// loss decodes pixels) + every enabled loss's model, plus `entries` cached references (epic 2123
/// E7). `0` when no aux loss is enabled.
pub fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &taesdxl_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// An NHWC `[B, h, w, 4]` SDXL latent → the decoder's NCHW `[B, 4, h, w]`.
fn nchw(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 3, 1, 2])?)
}

/// The dense first-step peak projection a run over `edges` must fit (epic 2123 E7, sc-2127): the
/// largest of the family's fitted `peak_gb` curve over every bucketed training edge, with the edge
/// that produces it. `p = ⌈edge/8⌉²` (the SDXL VAE downscales /8). The latent cache is not part of
/// the curve (it models the per-step U-Net working set, and the cached latents are a few hundred KB
/// each), so caching one latent per bucket does not move it.
pub fn dense_peak_for_edges(
    peak_gb: impl Fn(f64, bool) -> f64,
    edges: &[u32],
    bf16: bool,
) -> (u32, f64) {
    edges
        .iter()
        .map(|&edge| {
            let latent_side = (edge as f64 / 8.0).ceil();
            (edge, peak_gb(latent_side * latent_side, bf16))
        })
        .fold((0, f64::NEG_INFINITY), |best, cur| {
            if cur.1 > best.1 {
                cur
            } else {
                best
            }
        })
}

/// Refuse a run whose dense first step would exceed this machine's memory budget, returning a
/// catchable, actionable error instead of risking an uncatchable SIGKILL (sc-4874/sc-4941).
/// Consulted when gradient checkpointing is OFF, and — whenever the training-time auxiliary models
/// add memory (`extra_gb`, [`perceptual_footprint_gb`], epic 2123 E7) — when it is on too. `edges`
/// are the bucketed training edges; the dense guard sizes for the most expensive one
/// ([`dense_peak_for_edges`] over the family's fitted [`SdxlFamilyHooks::peak_gb`] curve). With
/// `checkpointed`, the U-Net projection is the curve's resident base (`peak_gb(0)`: no fitted
/// checkpointed curve exists, so the resident U-Net + VAE is the lower bound the aux models stack on).
fn preflight_memory_guard<H: SdxlFamilyHooks>(
    hooks: &H,
    cfg: &TrainingConfig,
    edges: &[u32],
    bf16: bool,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    preflight_memory_guard_with_budget(
        hooks,
        cfg,
        edges,
        bf16,
        extra_gb,
        checkpointed,
        get_memory_limit(),
    )
}

/// The pre-flight memory guard against an explicit memory budget (`budget_bytes`, the live MLX limit
/// in production) — so the guard's arithmetic is testable on any host. A checkpointed refusal goes
/// through the shared [`mlx_gen_perceptual::check_aux_memory`], naming `cfg`'s enabled aux losses.
#[doc(hidden)]
pub fn preflight_memory_guard_with_budget<H: SdxlFamilyHooks>(
    hooks: &H,
    cfg: &TrainingConfig,
    edges: &[u32],
    bf16: bool,
    extra_gb: f64,
    checkpointed: bool,
    budget_bytes: usize,
) -> Result<()> {
    if checkpointed && extra_gb <= 0.0 {
        // No fitted checkpointed curve: a checkpointed run is only guarded for its aux models.
        return Ok(());
    }
    let (edge, dense) = dense_peak_for_edges(|p, b| hooks.peak_gb(p, b), edges, bf16);
    let projected = if checkpointed {
        hooks.peak_gb(0.0, bf16)
    } else {
        dense
    } + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if checkpointed {
        return mlx_gen_perceptual::check_aux_memory(
            &format!("{} trainer", hooks.label()),
            cfg,
            &format!("a checkpointed training step at resolution {edge}"),
            extra_gb,
            projected,
            safe,
            &format!("{budget_gb:.0} GB MLX limit × 0.85"),
        );
    }
    if projected > safe {
        return Err(mlx_gen_perceptual::name_aux_losses(
            format!(
                "{} trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
                 (the forward working set materializes in one allocation), exceeding this machine's \
                 ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
                 could hard-kill the worker (SIGKILL) at the first step with no recoverable error. Enable \
                 Gradient Checkpointing or reduce the training resolution.",
                hooks.label()
            )
            .into(),
            cfg,
            extra_gb,
        ));
    }
    Ok(())
}

/// One cached training sample: an item's clean latent at ONE bucket edge, its (shared, refcounted)
/// conditioning/pooled, and the family micro-conditioning `time_ids` built for that same edge. Keeping
/// `time_ids` in the entry means a step can never pair a latent with ids built for another bucket
/// (sc-2127 — Kolors' ids are the real `(H, W, 0, 0, H, W)`). Likewise the subject-masked loss
/// weight (sc-24828) is resampled for, and stored with, the latent of that same edge.
pub(crate) struct CachedSample {
    pub(crate) x0: Array,
    pub(crate) cond: Array,
    pub(crate) pooled: Array,
    pub(crate) time_ids: Array,
    /// Subject-mask loss weight, `x0`'s exact shape — `None` when the technique is off.
    pub(crate) mask_weight: Option<Array>,
}

/// Push one dataset item's cache entries — one per bucket edge, in `edges` order (the item-major
/// layout [`BucketSchedule::cache_index`] indexes) — encoding the latent per edge via
/// `encode_latent`, pairing it with `time_ids[bucket]` (built for that edge) and with
/// `mask_weight(x0.shape())` (sc-24828: the item's subject mask resampled onto **that** latent's
/// grid). The item's conditioning is encoded once by the caller and cloned (refcounted) into each
/// entry.
fn push_bucket_entries(
    cache: &mut Vec<CachedSample>,
    edges: &[u32],
    time_ids: &[Array],
    cond: &Array,
    pooled: &Array,
    mut encode_latent: impl FnMut(u32) -> Result<Array>,
    mut mask_weight: impl FnMut(&[i32]) -> Result<Option<Array>>,
) -> Result<()> {
    debug_assert_eq!(edges.len(), time_ids.len());
    for (&edge, ids) in edges.iter().zip(time_ids) {
        let x0 = encode_latent(edge)?;
        let mask_weight = mask_weight(x0.shape())?;
        eval(std::iter::once(&x0).chain(mask_weight.as_ref()))?;
        cache.push(CachedSample {
            x0,
            cond: cond.clone(),
            pooled: pooled.clone(),
            time_ids: ids.clone(),
            mask_weight,
        });
    }
    Ok(())
}

/// The cache entry the 1-based training `step` reads: the schedule's `(step - 1)`-th sample. For a
/// single bucket this is the pre-bucket round-robin `(step - 1) % n_items`.
fn step_cache_index(schedule: &BucketSchedule, step: u32) -> usize {
    schedule.cache_index((step - 1) as usize)
}

/// The loop's [`AuxDriver`] (epic 2123 E8): every cache entry's perceptual reference computed once
/// (its clean NHWC latent, transposed to the decoder's NCHW), keyed per (item, bucket) entry — each
/// bucket's latent decodes to its own size — the alternation over the schedule's epochs with
/// `accum` micro-steps per update.
pub(crate) fn aux_driver(
    path: PerceptualPath,
    cache: &[CachedSample],
    schedule: &BucketSchedule,
    accum: u32,
    cancel: &mlx_gen::gen_core::runtime::CancelFlag,
) -> Result<AuxDriver> {
    AuxDriver::prepare(
        path,
        cache.len(),
        |i| nchw(&cache[i].x0),
        schedule,
        accum,
        cancel,
    )
}

/// The timestep a planned step trains at (epic 2123 E8): the plan's level whenever it moved off
/// the sampled `t`'s — an aux-only step, and also a claimed step that `StepPlan::without_skipped`
/// reverted to diffusion (it still trains at the remapped level, as on Candle) — else `t` exactly.
pub(crate) fn trained_timestep<H: SdxlFamilyHooks>(
    hooks: &H,
    t: TrainTimestep,
    plan: &StepPlan,
) -> TrainTimestep {
    if plan.noise_level != hooks.noise_level(t) {
        hooks.timestep_at(plan.noise_level)
    } else {
        t
    }
}

/// One training micro-step on the 1-based `step`: pick the step's cached (item, bucket) entry,
/// sample its timestep and noise (seeded, exactly as before epic 2123), plan the step's loss terms
/// through the perceptual path (when one is configured: the alternation key comes from the step's
/// optimizer window, and an aux-only step trains at the sampled noise level remapped into the loss
/// window, mapped back onto the family's timestep table), and run [`compute_step_loss_grads`]. With
/// no perceptual path every step is the plain diffusion step, bit-identical to the pre-epic-2123
/// loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_train_step<H: SdxlFamilyHooks>(
    hooks: &H,
    unet: &mut UNet2DConditionModel,
    params: &LoraParams,
    adapter: &TrainAdapter,
    cfg: &TrainingConfig,
    cache: &[CachedSample],
    schedule: &BucketSchedule,
    perceptual: Option<&mut AuxDriver>,
    step: u32,
    mae: bool,
    compute_dtype: Dtype,
    checkpoint_targets: Option<Vec<String>>,
) -> Result<(StepLosses, LoraParams)> {
    let entry = step_cache_index(schedule, step);
    let CachedSample {
        x0,
        cond,
        pooled,
        time_ids,
        mask_weight,
    } = &cache[entry];
    let mut t =
        hooks.sample_timestep(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
    let noise = random::normal::<f32>(
        x0.shape(),
        None,
        None,
        Some(&random::key(
            cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
        )?),
    )?;
    let planned = match perceptual {
        Some(d) => d.sample(step, schedule).plan(hooks.noise_level(t))?,
        None => None,
    };
    if let Some(p) = &planned {
        t = trained_timestep(hooks, t, &p.plan);
    }
    let aux = planned.as_ref().map(|p| AuxStep {
        path: p.path,
        plan: &p.plan,
        entry: p.entry,
    });
    compute_step_loss_grads(
        hooks,
        unet,
        params,
        adapter,
        cfg.alpha,
        cfg.rank as f32,
        x0,
        cond,
        pooled,
        time_ids,
        t,
        &noise,
        mae,
        mask_weight.as_ref(),
        compute_dtype,
        checkpoint_targets,
        aux,
    )
}

/// Decode a dataset image file (PNG/JPEG) into the core RGB8 [`Image`](mlx_gen::media::Image).
fn decode_image(path: &Path) -> Result<mlx_gen::media::Image> {
    let dynimg = image::open(path)
        .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(mlx_gen::media::Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// The subject-masked loss weight (sc-24828) for one cached latent whose shape is the family's
/// **NHWC** `[1, h, w, 4]` (from `encode_init_latents` of the [`center_crop_square`] image at one
/// bucket edge): the item's prepared mask is cropped with [`CropBox::center_square`],
/// area-averaged onto **that latent's** `(h, w)` grid and laid out by [`nhwc_weight`] to broadcast
/// over the channel axis — `x0`'s exact shape. `None` when off (`mask` is `None`).
fn nhwc_subject_weight(
    label: &str,
    mask: Option<&PreparedSubjectMask>,
    x0_shape: &[i32],
) -> Result<Option<Array>> {
    if mask.is_none() {
        return Ok(None);
    }
    if x0_shape.len() != 4 {
        return Err(mlx_gen::Error::Msg(format!(
            "{label}: subject mask expects an NHWC latent, got shape {x0_shape:?}"
        )));
    }
    prepared_subject_mask_weight(label, mask, CropBox::center_square, &x0_shape[..3])?
        .map(|w| nhwc_weight(&w, x0_shape))
        .transpose()
}

/// A `[B, h, w]` weight map → `x0_shape` (`[B, h, w, C]`): the latent cell `(y, x)` weight on every
/// channel of that cell.
fn nhwc_weight(weight_bhw: &Array, x0_shape: &[i32]) -> Result<Array> {
    Ok(broadcast_to(&weight_bhw.expand_dims(-1)?, x0_shape)?)
}

/// The shared SDXL-family LoRA/LoKr training lifecycle: prepare → load → cache (VAE-latents +
/// conditioning/pooled) → functional-autograd loop (LR schedule, gradient accumulation, checkpoint
/// cadence, cancel, preview samples) → save an adapter that round-trips through the family's inference
/// loader. The per-family deltas are supplied by `hooks`; `unet`/`vae` are the (shared) SDXL types the
/// trainer owns. `validate` runs in the caller's `Trainer::train` before this is entered.
pub fn train_family<H: SdxlFamilyHooks>(
    hooks: &mut H,
    unet: &mut UNet2DConditionModel,
    vae: &Autoencoder,
    req: &TrainingRequest,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainingOutput> {
    let cfg = &req.config;
    let label = hooks.label();
    on_progress(TrainingProgress::Preparing);
    // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are off).
    // The memory guard and preview renders size for the largest (epic 2123 E7).
    let edges = bucket_edges(cfg);
    let max_edge = edges.iter().copied().max().unwrap_or(0);

    // sc-4941 — training compute dtype. bf16 (the worker default, passed through since sc-4881) halves
    // the activation working set and is the ecosystem-standard mixed precision; the trainable factors /
    // loss / grads / optimizer stay f32 (master-weights). The U-Net f32→bf16 cast is destructive, so a
    // trainer already cast to bf16 cannot honor a later f32 request — reload instead of silently
    // training at the wrong precision.
    let use_bf16 = cfg.train_dtype.trim().eq_ignore_ascii_case("bf16")
        || cfg.train_dtype.trim().eq_ignore_ascii_case("bfloat16");
    let compute_dtype = if use_bf16 {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    };
    if !use_bf16 && unet.compute_dtype() == Some(Dtype::Bfloat16) {
        return Err(format!(
            "{label} trainer: this trainer instance was already cast to bf16 by a previous run; \
             reload the trainer for f32 training"
        )
        .into());
    }

    // sc-4941 — opt-in gradient checkpointing (the SceneWorks "Gradient Checkpointing" toggle). When
    // on, each down/up macro-block recomputes its activations in the backward
    // (`forward_block_checkpointed`) instead of retaining them — the lever that makes 1280+ training fit
    // a 32 GB Mac (1024 already fits dense bf16). LoRA-only: LoKr falls back to the dense path (a
    // distinct Kronecker reconstruction), where the pre-flight guard refuses a run that would exceed the
    // memory budget. The block recompute already covers attention, so the standalone SDPA-segment
    // checkpoint stays off (nesting = double recompute).
    let use_checkpoint =
        matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
    // Epic 2123 E7: the training-time auxiliary models (TAESDXL + Depth-Anything-V2) count against
    // the budget on BOTH paths; one cached depth reference per (item, bucket) entry, sized at the
    // largest edge.
    let aux_gb = perceptual_footprint_gb(cfg, max_edge, req.items.len() * edges.len());
    // (A checkpointed run with no aux models is not guarded — see the guard.)
    preflight_memory_guard(hooks, cfg, &edges, use_bf16, aux_gb, use_checkpoint)?;
    unet.set_sdpa_checkpoint(false);
    if use_bf16 {
        unet.cast_weights(Dtype::Bfloat16)?;
    }

    // Epic 2123 depth anchoring: load the frozen TAESDXL decoder + Depth-Anything-V2 before the
    // caching pass, so a missing/corrupt aux checkpoint fails fast.
    let mut perceptual = load_perceptual_path(label, cfg)?;

    // --- prepare → load → cache: VAE-latents + (conditioning, pooled) into memory ---
    on_progress(TrainingProgress::LoadingModel); // base already resident from load_trainer
    let total = req.items.len() as u32;
    // Family micro-conditioning `time_ids` per bucket edge (B=1) — matches the inference path so the
    // LoRA trains under the conditioning it is applied under, at the size each latent was cached at.
    let time_ids: Vec<Array> = edges.iter().map(|&e| hooks.time_ids(1, e)).collect();
    // Item-major: `cache[item * edges.len() + bucket]` (sc-2127); each entry carries its latent's
    // subject-mask loss weight (sc-24828) — `None` when the technique is off.
    let mut cache: Vec<CachedSample> = Vec::with_capacity(req.items.len() * edges.len());
    for (i, item) in req.items.iter().enumerate() {
        if req.cancel.is_cancelled() {
            break;
        }
        on_progress(TrainingProgress::Caching {
            current: i as u32 + 1,
            total,
        });
        let img = center_crop_square(&decode_image(&item.image_path)?);
        // sc-24828: the item's mask is read and checked once, resampled per bucket below.
        let mask =
            PreparedSubjectMask::load_if_enabled(label, item, cfg.subject_mask_loss.as_ref())?;
        let (cond, pooled) = hooks.encode_prompt(&item.caption)?;
        eval([&cond, &pooled])?;
        // scaled latent [1,h,w,4] per bucket edge, with its own subject-mask weight
        push_bucket_entries(
            &mut cache,
            &edges,
            &time_ids,
            &cond,
            &pooled,
            |edge| encode_init_latents(vae, &img, edge, edge),
            |shape| nhwc_subject_weight(label, mask.as_ref(), shape),
        )?;
    }
    if cache.is_empty() {
        // sc-4895 — a cancel tripped during caching is a genuine cancellation → typed
        // `Error::Canceled` (bridged 1:1 to `gen_core::Error::Canceled`); an empty cache with no
        // cancel is a real "no usable dataset items" error.
        if req.cancel.is_cancelled() {
            return Err(mlx_gen::Error::Canceled);
        }
        return Err(format!("{label} trainer: no usable dataset items").into());
    }

    // Epic 2123 E8: each (item, bucket) entry's perceptual reference (TAESDXL decode of its cached
    // clean latent → DA2 depth) is computed exactly once per job, by the `AuxDriver` before the
    // loop.
    if let Some(path) = perceptual.as_mut() {
        // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
        // cropped like the image and resampled onto its decoded size.
        path.attach_subject_masks(mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
            hooks.label(),
            &req.items,
            cfg,
            edges.len(),
            CropBox::center_square,
        )?);
    }

    // sc-5637 — pre-encode the preview-sample prompts as a **CFG batch** (`[2, …]` = positive then
    // empty-negative) while the text encoder(s) are still resident (freed just below). The family
    // renders previews with real classifier-free guidance, so the denoise needs both streams.
    let sample_caps: Vec<(String, Array, Array)> =
        if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled() {
            let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let (cond, pooled) = hooks.encode_sample_cfg(prompt)?;
                let cond = if compute_dtype == Dtype::Float32 {
                    cond
                } else {
                    cond.as_dtype(compute_dtype)?
                };
                let pooled = if compute_dtype == Dtype::Float32 {
                    pooled
                } else {
                    pooled.as_dtype(compute_dtype)?
                };
                eval([&cond, &pooled])?;
                caps.push((prompt.clone(), cond, pooled));
            }
            caps
        } else {
            Vec::new()
        };
    let sampling_enabled = !sample_caps.is_empty();

    // sc-4941 (32 GB-Mac headroom) — the prompts are all encoded into `cache`, so the text encoder(s)
    // are dead weight for the rest of the run. Drop them and evict their buffers before the train loop,
    // reclaiming their footprint for the U-Net working set.
    hooks.free_text_encoders();
    mlx_rs::memory::clear_cache();

    // --- adapter targets + params (LoRA or LoKr) + optimizer ---
    let target_paths = resolve_target_paths(unet, cfg);
    // When block checkpointing is on, the per-step forward threads these target paths' LoRA factors
    // through the block checkpoints; `None` selects the dense forward.
    let checkpoint_targets: Option<Vec<String>> = use_checkpoint.then(|| target_paths.clone());
    let rank = cfg.rank as f32;
    let (adapter, mut params) = match cfg.network_type {
        NetworkType::Lora => {
            let (targets, params) =
                build_lora_targets(unet, &target_paths, cfg.rank as i32, cfg.seed)?;
            (TrainAdapter::Lora { targets }, params)
        }
        NetworkType::Lokr => {
            let (targets, params) = build_lokr_targets(
                unet,
                &target_paths,
                cfg.rank as i32,
                cfg.decompose_factor,
                cfg.seed,
            )?;
            (TrainAdapter::Lokr { targets }, params)
        }
    };
    let alpha = cfg.alpha;
    let mae = {
        let lt = cfg.loss_type.to_ascii_lowercase();
        lt == "mae" || lt == "l1"
    };
    // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
    let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
        0.0
    } else {
        cfg.weight_decay
    };
    let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;

    let accum = cfg.gradient_accumulation.max(1);
    let (total_updates, warmup_updates) = schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
    let stem = Path::new(&req.file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("lora")
        .to_string();

    // --- resume (F-125): continue from the latest snapshot of THIS adapter in output_dir, if any ---
    let mut update_idx: u32 = 0;
    let mut start_step: u32 = 0;
    if cfg.resume {
        if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
            let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
            params = loaded;
            start_step = meta.step;
            update_idx = meta.update_idx;
            eprintln!(
                "[F-125] {label} resuming from step {start_step} (optimizer update {update_idx})"
            );
        }
    }

    // --- train loop ---
    // sc-2127: which cached (item, bucket) sample each step trains on (round-robin over items for a
    // single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
    let schedule =
        BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
    // Epic 2123 E8: the perceptual alternation interleaves optimizer windows over the schedule's
    // epochs (sc-2124) — a pure function of the step, so a resumed run needs no replay.
    let mut aux_driver = match perceptual {
        Some(path) => Some(aux_driver(path, &cache, &schedule, accum, &req.cancel)?),
        None => None,
    };
    let mut accumulated: Option<LoraParams> = None;
    let mut last_loss = 0.0f32;
    let mut steps_run = start_step;
    for step in start_step + 1..=cfg.steps {
        if req.cancel.is_cancelled() {
            break;
        }
        let (losses, grads) = run_train_step(
            hooks,
            unet,
            &params,
            &adapter,
            cfg,
            &cache,
            &schedule,
            aux_driver.as_mut(),
            step,
            mae,
            compute_dtype,
            checkpoint_targets.clone(),
        )?;
        let loss = losses.total;
        last_loss = loss;
        steps_run = step;
        accumulate_grads(&mut accumulated, grads)?;

        if step % accum == 0 || step == cfg.steps {
            let mult = lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
            opt.set_lr_scaled(mult);
            // F-017: average by the ACTUAL in-window count, not the full `accum`. The final-step
            // flush is usually a partial window (cfg.steps % accum != 0); dividing by `accum`
            // down-scaled that update (halved effective LR on the tail) for BOTH the SDXL and
            // Kolors trainers. Mirrors z-image/lens F-069. (When step%accum==0 the window is the
            // full `accum`.)
            let window = if step % accum == 0 {
                accum
            } else {
                step % accum
            };
            let avg = average_grads(
                accumulated
                    .take()
                    .expect("an update fires only after accumulation"),
                window,
            )?;
            // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
            adapter_optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx, cfg.seed)?;
            update_idx += 1;
        }

        on_progress(TrainingProgress::Training {
            step,
            total: cfg.steps,
            loss: last_loss,
        });

        if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
            std::fs::create_dir_all(&req.output_dir)?;
            let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
            adapter.save(
                &params,
                alpha,
                rank,
                cfg.decompose_factor,
                PEFT_PREFIX,
                &ckpt,
            )?;
            // F-125: the resume bundle (raw factors + optimizer state + step/update index) siblings the
            // PEFT checkpoint, so a later `config.resume` run continues this schedule from here.
            checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
            on_progress(TrainingProgress::Checkpoint { step });
        }

        // sc-5637 — periodic best-effort previews from the in-progress adapter (mirrors z-image).
        // Install the current factors as concrete adapters for the forward-only render; the next step's
        // traced `loss_fn` re-installs them, so no teardown is needed. A render failure must NOT abort
        // the long training run — log and continue.
        if sampling_enabled && step % cfg.sample_every == 0 {
            let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
            adapter.install_as(unet, &params, alpha, rank, lora_dtype, LOKR_DTYPE)?;
            let total = sample_caps.len() as u32;
            for (i, (prompt, cond, pooled)) in sample_caps.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                let sample_seed = cfg
                    .seed
                    .wrapping_add(step as u64)
                    .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                    .wrapping_add(i as u64);
                match hooks.render_sample(
                    unet,
                    vae,
                    cond,
                    pooled,
                    cfg.sample_guidance_scale,
                    sample_seed,
                    max_edge,
                    cfg.sample_steps.max(1) as usize,
                    compute_dtype,
                ) {
                    Ok(image) => on_progress(TrainingProgress::Sample {
                        step,
                        index: i as u32 + 1,
                        total,
                        prompt: prompt.clone(),
                        image,
                    }),
                    Err(e) => eprintln!(
                        "[sc-5637] {label} preview sample failed at step {step} \
                         (prompt {}): {e} — skipping this preview, training continues",
                        i + 1
                    ),
                }
            }
        }
    }

    // Cancelled before completing a single step (`steps == 0` is rejected upstream by `validate`): the
    // LoRA factors are still freshly initialized with `B = 0`, a no-op adapter. Surface the typed
    // `Error::Canceled` (sc-4895, bridged 1:1 to `gen_core::Error::Canceled`) rather than writing a
    // valid-looking `.safetensors` and returning `Ok` — downstream tooling would otherwise ship an
    // identity LoRA as a trained artifact (F-040).
    if steps_run == 0 {
        return Err(mlx_gen::Error::Canceled);
    }

    // --- save final adapter ---
    on_progress(TrainingProgress::Saving);
    std::fs::create_dir_all(&req.output_dir)?;
    let adapter_path = req.output_dir.join(&req.file_name);
    adapter.save(
        &params,
        alpha,
        rank,
        cfg.decompose_factor,
        PEFT_PREFIX,
        &adapter_path,
    )?;
    Ok(TrainingOutput {
        adapter_path,
        steps: steps_run,
        final_loss: last_loss,
    })
}

#[cfg(test)]
mod subject_mask_tests {
    use super::*;

    /// sc-24828: the NHWC weight lines up element-for-element with the NHWC latent — cell `(y, x)`'s
    /// value (encoded as `y·10 + x`) sits on every channel of `[0, y, x, :]`.
    #[test]
    fn nhwc_weight_lines_up_with_the_nhwc_latent() {
        let (h, w, c) = (2usize, 3usize, 4i32);
        let values: Vec<f32> = (0..h * w).map(|i| ((i / w) * 10 + i % w) as f32).collect();
        let bhw =
            mlx_gen::train::loss::subject_mask_weight(&values, h, w, &[1, h as i32, w as i32])
                .unwrap();
        let x0_shape = [1, h as i32, w as i32, c];
        let nhwc = nhwc_weight(&bhw, &x0_shape).unwrap();
        assert_eq!(nhwc.shape(), &x0_shape);
        // Flatten row-major (a broadcast is a strided view; the reshape copies it out in order).
        let flat = nhwc.reshape(&[-1]).unwrap();
        let flat = flat.as_slice::<f32>();
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c as usize {
                    let at = (y * w + x) * c as usize + ch;
                    assert_eq!(flat[at], (y * 10 + x) as f32, "(y={y}, x={x}, c={ch})");
                }
            }
        }
    }

    /// Mask off ⇒ no weight and no file read (the item names a mask path that does not exist).
    #[test]
    fn mask_off_builds_no_weight() {
        let mut item = mlx_gen::TrainingItem::captioned("/nonexistent/a.png".into(), "a".into());
        item.subject_mask_path = Some("/nonexistent/a.mask.png".into());
        let cfg = TrainingConfig::default();
        let mask =
            PreparedSubjectMask::load_if_enabled("sdxl", &item, cfg.subject_mask_loss.as_ref())
                .unwrap();
        assert!(mask.is_none());
        assert!(nhwc_subject_weight("sdxl", mask.as_ref(), &[1, 4, 4, 4])
            .unwrap()
            .is_none());
    }
}

#[cfg(test)]
mod bucket_tests {
    use super::*;
    use mlx_gen::gen_core::ResolutionBucket;

    fn bucket(resolution: u32, repeats: u32) -> ResolutionBucket {
        ResolutionBucket {
            resolution,
            repeats,
        }
    }

    /// sc-2127: with one bucket the step → cache-entry walk is exactly the pre-bucket
    /// `(step - 1) % n_items` round-robin, so a bucket-less run trains in today's order.
    #[test]
    fn one_bucket_step_index_is_the_pre_bucket_round_robin() {
        let n_items = 7;
        for repeats in [1, 3] {
            let schedule = BucketSchedule::new(n_items, &[bucket(1024, repeats)], 42);
            for step in 1..=200u32 {
                assert_eq!(
                    step_cache_index(&schedule, step),
                    ((step - 1) as usize) % n_items,
                    "step {step} (repeats {repeats})"
                );
            }
        }
    }

    /// sc-2127: over one epoch of a `[512×16, 768×4, 1024×1]` schedule every item is visited
    /// 16:4:1 across its three cached buckets (cache stride 3, item-major).
    #[test]
    fn multi_bucket_steps_mix_16_4_1_per_item() {
        let n_items = 3;
        let schedule = BucketSchedule::new(
            n_items,
            &[bucket(512, 16), bucket(768, 4), bucket(1024, 1)],
            7,
        );
        let epoch = n_items * 21;
        let mut counts = vec![[0u32; 3]; n_items];
        for step in 1..=epoch as u32 {
            let idx = step_cache_index(&schedule, step);
            counts[idx / 3][idx % 3] += 1;
        }
        for (item, c) in counts.iter().enumerate() {
            assert_eq!(*c, [16, 4, 1], "item {item}");
        }
    }

    /// sc-2127: each item is cached once per bucket edge, item-major, and every entry's
    /// micro-conditioning `time_ids` is the one built for the edge its latent was encoded at (the
    /// Kolors `(H, W, 0, 0, H, W)` shape), so a step never pairs a latent with another bucket's ids.
    #[test]
    fn bucket_entries_pair_each_latent_with_its_edges_time_ids() {
        let edges = [512u32, 768, 1024];
        let time_ids: Vec<Array> = edges
            .iter()
            .map(|&e| {
                let e = e as f32;
                Array::from_slice(&[e, e, 0.0, 0.0, e, e], &[1, 6])
            })
            .collect();
        let cond = Array::from_slice(&[1.0f32], &[1, 1]);
        let pooled = Array::from_slice(&[2.0f32], &[1, 1]);
        let mut cache = Vec::new();
        for _item in 0..2 {
            push_bucket_entries(
                &mut cache,
                &edges,
                &time_ids,
                &cond,
                &pooled,
                |edge| {
                    let side = (edge / 8) as i32;
                    Ok(mlx_rs::ops::zeros::<f32>(&[1, side, side, 4])?)
                },
                |_| Ok(None),
            )
            .unwrap();
        }
        assert_eq!(cache.len(), 2 * edges.len());
        for (k, entry) in cache.iter().enumerate() {
            let latent_edge = entry.x0.shape()[1] as u32 * 8;
            assert_eq!(
                latent_edge,
                edges[k % edges.len()],
                "entry {k} is item-major"
            );
            let ids: Vec<f32> = entry.time_ids.as_slice::<f32>().to_vec();
            let e = latent_edge as f32;
            assert_eq!(ids, vec![e, e, 0.0, 0.0, e, e], "entry {k} time_ids");
        }
    }

    /// sc-2127 / epic 2123 E7: the projection is the most expensive bucket, independent of order.
    #[test]
    fn dense_peak_sizes_for_the_largest_edge() {
        let curve = |p: f64, _bf16: bool| 1.0 + p;
        let (edge, gb) = dense_peak_for_edges(curve, &[1024, 512, 768], true);
        assert_eq!(edge, 1024);
        assert_eq!(gb, 1.0 + 128.0 * 128.0);
    }

    /// sc-24828 × sc-2127: with subject-masked loss on and three buckets, every cache entry's
    /// weight is the item's mask resampled onto **that entry's** latent grid (centre-square crop of
    /// a non-square image) — `x0`'s exact shape — with the masked-out (right-half-of-crop) cells
    /// zero and the subject cells at the subject weight.
    ///
    /// *Mutation that reds this:* `push_bucket_entries` resampling the mask once at the first
    /// bucket's latent shape and reusing it for every bucket.
    #[test]
    fn bucket_entries_resample_the_subject_mask_per_edge() {
        let tmp = tempfile::tempdir().unwrap();
        // 96×64 image: the centre square is x ∈ [16, 80). The subject is x < 48 — exactly the
        // crop's left half.
        let image_path = tmp.path().join("a.png");
        image::RgbImage::from_pixel(96, 64, image::Rgb([90, 120, 150]))
            .save(&image_path)
            .unwrap();
        let mask_path = tmp.path().join("a.mask.png");
        image::GrayImage::from_fn(96, 64, |x, _| image::Luma([if x < 48 { 255 } else { 0 }]))
            .save(&mask_path)
            .unwrap();
        let mut item = mlx_gen::TrainingItem::captioned(image_path, "a".into());
        item.subject_mask_path = Some(mask_path);
        let cfg = mlx_gen::gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask = PreparedSubjectMask::load_if_enabled("sdxl", &item, Some(&cfg))
            .unwrap()
            .expect("mask loss on");
        let edges = [256u32, 512, 768];
        let time_ids: Vec<Array> = edges
            .iter()
            .map(|_| Array::from_slice(&[0.0f32; 6], &[1, 6]))
            .collect();
        let cond = Array::from_slice(&[1.0f32], &[1, 1]);
        let pooled = Array::from_slice(&[2.0f32], &[1, 1]);
        let mut cache = Vec::new();
        push_bucket_entries(
            &mut cache,
            &edges,
            &time_ids,
            &cond,
            &pooled,
            |edge| {
                let side = (edge / 8) as i32;
                Ok(mlx_rs::ops::zeros::<f32>(&[1, side, side, 4])?)
            },
            |shape| nhwc_subject_weight("sdxl", Some(&mask), shape),
        )
        .unwrap();
        assert_eq!(cache.len(), edges.len());
        for (k, entry) in cache.iter().enumerate() {
            let weight = entry.mask_weight.as_ref().expect("mask loss on ⇒ a weight");
            assert_eq!(weight.shape(), entry.x0.shape(), "entry {k} weight shape");
            let side = entry.x0.shape()[2] as usize;
            let flat = weight.reshape(&[-1]).unwrap();
            for (cell, chans) in flat.as_slice::<f32>().chunks(4).enumerate() {
                let want = if cell % side < side / 2 { 1.0 } else { 0.0 };
                assert!(
                    chans.iter().all(|&v| (v - want).abs() < 1e-5),
                    "entry {k} cell {cell}: {chans:?} want {want}"
                );
            }
        }
    }
}

/// A tiny random-init SDXL-family U-Net for weights-free trainer tests (epic 2123 sc-24830) — this
/// crate's and `mlx-gen-kolors`'s (the Kolors U-Net IS this U-Net). Not a stable API.
#[doc(hidden)]
pub mod test_support {
    use mlx_gen::weights::Weights;
    use mlx_rs::ops::multiply;
    use mlx_rs::{random, Array};

    use crate::config::UNetConfig;
    use crate::unet::UNet2DConditionModel;
    use mlx_gen::Result;

    /// Cross-attention context width of [`tiny_unet_config`].
    pub const TINY_CONTEXT_DIM: i32 = 16;
    /// Pooled text-embedding width of [`tiny_unet_config`].
    pub const TINY_POOLED_DIM: i32 = 16;

    /// Two blocks (`[32, 64]` channels — the U-Net's GroupNorm is a fixed 32 groups), one resnet per
    /// block, a cross-attention transformer on the inner block + the mid block, 4 latent channels,
    /// SDXL's `text_time` added conditioning (6 time ids × 8 + the 16-wide pooled text).
    pub fn tiny_unet_config() -> UNetConfig {
        UNetConfig {
            in_channels: 4,
            out_channels: 4,
            conv_in_kernel: 3,
            conv_out_kernel: 3,
            block_out_channels: vec![32, 64],
            layers_per_block: vec![1, 1],
            transformer_layers_per_block: vec![1, 1],
            num_attention_heads: vec![1, 2],
            cross_attention_dim: vec![TINY_CONTEXT_DIM, TINY_CONTEXT_DIM],
            norm_num_groups: 32,
            down_block_types: vec!["DownBlock2D".into(), "CrossAttnDownBlock2D".into()],
            // Stored already-reversed, like `UNetConfig::sdxl_base`.
            up_block_types: vec!["UpBlock2D".into(), "CrossAttnUpBlock2D".into()],
            addition_embed_type: Some("text_time".into()),
            addition_time_embed_dim: Some(8),
            projection_class_embeddings_input_dim: Some(TINY_POOLED_DIM + 6 * 8),
        }
    }

    struct Gen {
        w: Weights,
        seed: u64,
        n: u64,
    }

    impl Gen {
        fn rnd(&mut self, shape: &[i32], scale: f32) -> Result<Array> {
            self.n += 1;
            let key = random::key(self.seed.wrapping_mul(1_000_003).wrapping_add(self.n))?;
            Ok(multiply(
                &random::normal::<f32>(shape, None, None, Some(&key))?,
                Array::from_f32(scale),
            )?)
        }
        fn put(&mut self, key: String, shape: &[i32], scale: f32) -> Result<()> {
            let t = self.rnd(shape, scale)?;
            self.w.insert(key, t);
            Ok(())
        }
        /// A dense layer `[out, in]` (+ bias), fan-in scaled.
        fn linear(&mut self, p: &str, out: i32, inp: i32, bias: bool) -> Result<()> {
            self.put(
                format!("{p}.weight"),
                &[out, inp],
                (1.0 / inp as f32).sqrt(),
            )?;
            if bias {
                self.put(format!("{p}.bias"), &[out], 0.02)?;
            }
            Ok(())
        }
        /// A torch OIHW conv `[out, in, k, k]` + bias.
        fn conv(&mut self, p: &str, out: i32, inp: i32, k: i32) -> Result<()> {
            self.put(
                format!("{p}.weight"),
                &[out, inp, k, k],
                (1.0 / (inp * k * k) as f32).sqrt(),
            )?;
            self.put(format!("{p}.bias"), &[out], 0.02)
        }
        /// A norm's affine pair (weight ≈ 1).
        fn norm(&mut self, p: &str, c: i32) -> Result<()> {
            let ones = Array::ones::<f32>(&[c])?;
            let jitter = self.rnd(&[c], 0.02)?;
            self.w
                .insert(format!("{p}.weight"), mlx_rs::ops::add(&ones, &jitter)?);
            self.put(format!("{p}.bias"), &[c], 0.02)
        }
        fn resnet(&mut self, p: &str, inp: i32, out: i32, temb: i32) -> Result<()> {
            self.norm(&format!("{p}.norm1"), inp)?;
            self.conv(&format!("{p}.conv1"), out, inp, 3)?;
            self.linear(&format!("{p}.time_emb_proj"), out, temb, true)?;
            self.norm(&format!("{p}.norm2"), out)?;
            self.conv(&format!("{p}.conv2"), out, out, 3)?;
            if inp != out {
                self.conv(&format!("{p}.conv_shortcut"), out, inp, 1)?;
            }
            Ok(())
        }
        fn transformer(&mut self, p: &str, c: i32, ctx: i32, layers: i32) -> Result<()> {
            self.norm(&format!("{p}.norm"), c)?;
            self.linear(&format!("{p}.proj_in"), c, c, true)?;
            self.linear(&format!("{p}.proj_out"), c, c, true)?;
            for i in 0..layers {
                let b = format!("{p}.transformer_blocks.{i}");
                for n in ["norm1", "norm2", "norm3"] {
                    self.norm(&format!("{b}.{n}"), c)?;
                }
                for (attn, kv) in [("attn1", c), ("attn2", ctx)] {
                    self.linear(&format!("{b}.{attn}.to_q"), c, c, false)?;
                    self.linear(&format!("{b}.{attn}.to_k"), c, kv, false)?;
                    self.linear(&format!("{b}.{attn}.to_v"), c, kv, false)?;
                    self.linear(&format!("{b}.{attn}.to_out.0"), c, c, true)?;
                }
                self.linear(&format!("{b}.ff.net.0.proj"), 8 * c, c, true)?;
                self.linear(&format!("{b}.ff.net.2"), c, 4 * c, true)?;
            }
            Ok(())
        }
    }

    /// Random weights for every key [`UNet2DConditionModel::from_weights`] reads under
    /// [`tiny_unet_config`] (diffusers key layout, torch OIHW convs). Deterministic in `seed`.
    pub fn tiny_unet_weights(seed: u64) -> Result<Weights> {
        let cfg = tiny_unet_config();
        let boc = cfg.block_out_channels.clone();
        let n = boc.len();
        let temb = cfg.time_embed_dim();
        let ctx = TINY_CONTEXT_DIM;
        let mut g = Gen {
            w: Weights::empty(),
            seed,
            n: 0,
        };
        g.conv("conv_in", boc[0], cfg.in_channels, 3)?;
        g.linear("time_embedding.linear_1", temb, boc[0], true)?;
        g.linear("time_embedding.linear_2", temb, temb, true)?;
        g.linear(
            "add_embedding.linear_1",
            temb,
            cfg.projection_class_embeddings_input_dim.unwrap(),
            true,
        )?;
        g.linear("add_embedding.linear_2", temb, temb, true)?;
        for i in 0..n {
            let p = format!("down_blocks.{i}");
            let inp = if i == 0 { boc[0] } else { boc[i - 1] };
            for j in 0..cfg.layers_per_block[i] {
                let rin = if j == 0 { inp } else { boc[i] };
                g.resnet(&format!("{p}.resnets.{j}"), rin, boc[i], temb)?;
                if cfg.down_block_types[i].contains("CrossAttn") {
                    g.transformer(
                        &format!("{p}.attentions.{j}"),
                        boc[i],
                        ctx,
                        cfg.transformer_layers_per_block[i],
                    )?;
                }
            }
            if i < n - 1 {
                g.conv(&format!("{p}.downsamplers.0.conv"), boc[i], boc[i], 3)?;
            }
        }
        let c = boc[n - 1];
        g.resnet("mid_block.resnets.0", c, c, temb)?;
        g.transformer(
            "mid_block.attentions.0",
            c,
            ctx,
            cfg.transformer_layers_per_block[n - 1],
        )?;
        g.resnet("mid_block.resnets.1", c, c, temb)?;
        for k in 0..n {
            let ci = n - 1 - k;
            let p = format!("up_blocks.{k}");
            let out = boc[ci];
            let prev = if k == 0 { boc[n - 1] } else { boc[ci + 1] };
            let input = boc[ci.saturating_sub(1)];
            let layers = cfg.layers_per_block[ci] + 1;
            for j in 0..layers {
                let skip = if j < layers - 1 { out } else { input };
                let rin = if j == 0 { prev } else { out };
                g.resnet(&format!("{p}.resnets.{j}"), rin + skip, out, temb)?;
                if cfg.up_block_types[ci].contains("CrossAttn") {
                    g.transformer(
                        &format!("{p}.attentions.{j}"),
                        out,
                        ctx,
                        cfg.transformer_layers_per_block[ci],
                    )?;
                }
            }
            if ci > 0 {
                g.conv(&format!("{p}.upsamplers.0.conv"), out, out, 3)?;
            }
        }
        g.norm("conv_norm_out", boc[0])?;
        g.conv("conv_out", cfg.out_channels, boc[0], 3)?;
        Ok(g.w)
    }

    /// The tiny U-Net built from [`tiny_unet_weights`]; every generated key is consumed.
    pub fn tiny_unet(seed: u64) -> Result<UNet2DConditionModel> {
        let w = tiny_unet_weights(seed)?;
        let unet = UNet2DConditionModel::from_weights(&w, &tiny_unet_config())?;
        let unused = w.unused_keys();
        if !unused.is_empty() {
            return Err(mlx_gen::Error::Msg(format!(
                "tiny U-Net fixture: unconsumed keys {unused:?}"
            )));
        }
        Ok(unet)
    }
}
