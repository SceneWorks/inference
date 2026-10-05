//! The candle **Krea 2 LoRA/LoKr trainer** (sc-7577) — the candle twin of `mlx-gen-krea`'s
//! `KreaRawTrainer`, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"` and reusing the shared [`candle_gen::train`] harness the SDXL/Z-Image
//! stories established. It trains on the **Krea-2-Raw** 12B base (the undistilled checkpoint; sc-7576)
//! and the adapter cross-applies to **Krea-2-Turbo** at inference by the family-match policy
//! (`baseModel: krea_2_raw`, `family: krea_2`).
//!
//! Since sc-7787 the cache → loop → save scaffolding lives in the shared single-model flow-match
//! driver ([`candle_gen::train::flow_match`]); this module supplies the Krea-specific hooks via
//! [`FlowMatchTrainer`] — caching, DiT construction, the parity-critical [`compute_loss_grads`], and the
//! provenance-stamping [`save`](FlowMatchTrainer::save).
//!
//! ## Cache → loop → save, on the flow-match objective
//!
//!  1. **Cache** — for each captioned image: decode/crop/resize to a VAE-input tensor
//!     ([`decode_square`] + [`square_image_tensor`]), encode the **deterministic latent mean** through the Qwen-Image
//!     [`QwenVaeEncoder`] (the `(mean − latents_mean)/latents_std` the DiT consumes — `encode` already
//!     skips the `DiagonalGaussian` draw), and encode the caption through the Qwen3-VL-4B text encoder
//!     with the *exact* tokenizer + select-layer stack inference uses → `(L, num_text_layers,
//!     text_hidden)`. The VAE encoder + text encoder are dropped after caching.
//!  2. **Loop** (driver-owned) — sample a flow-match `σ ∈ [1e-3, 1−1e-3]`
//!     ([`sample_unit_timestep`](candle_gen::train::flow_match::sample_unit_timestep)), form
//!     `x_t = (1−σ)·x0 + σ·noise`, predict the velocity through the vendored trainable DiT
//!     ([`KreaTrainDit`]) at timestep `σ` (the raw flow time the DiT's `temb` scales ×1000 — the
//!     [`TimestepConvention::Sigma`](candle_gen::gen_core::sampling) inference uses), and regress it
//!     toward `noise − x0`.
//!  3. **Save** — a PEFT `.safetensors` (`save_lora_peft` with the DiT's **bare** key prefix /
//!     `save_lokr`) stamped with `baseModel`/`family` provenance, the on-disk format the Turbo
//!     inference-side merge (sc-7578) reads back.
//!
//! **Velocity sign.** Krea's inference pipeline consumes the **raw** DiT velocity (`x + v·Δσ`,
//! [`crate::pipeline`]) — unlike Z-Image it does not negate — so [`KreaTrainDit::forward`] returns the
//! raw velocity and the trainer regresses it toward `noise − x0` directly (the Lens convention). The
//! timestep fed to the DiT is the raw `σ` (NOT `1−σ`), matching the inference `TimestepConvention::Sigma`
//! — which (with the absence of a pre-main adapter stitch) is why [`compute_loss_grads`] stays local.
//!
//! **The eager-`Var` simplification** (inherited from the SDXL/Z-Image harness): the adapter factors
//! are storage-sharing `Var`s installed once; each forward re-reads the current factor storage and
//! `loss.backward()` attributes grads straight to the `Var`s.
//!
//! **Gradient checkpointing** (`config.gradient_checkpointing`) routes the backward through
//! [`checkpointed_backward`] over the DiT's single-stream `blocks`. Because the default surface is the
//! 28 blocks' attention, **every** adapter lives in that checkpointed stack — there is no
//! retained-pre-main adapter to stitch (the Z-Image complication), so the frozen front-end is simply
//! run once and detached at the joint-sequence boundary. Both paths yield the same grads (the
//! `dense_and_checkpoint_grads_match` gate pins this).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, IndexOp, Tensor, Var};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, Image, LoadSpec, Modality, Progress, WeightsSource};
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor};
use candle_gen::train::flow_match::{
    self, combine_terms, prepared_subject_mask_weight, run_flow_match_training, step_terms,
    validate_flow_match_request, weighted_velocity_loss, AuxStep, FlowMatchTrainer, SamplePlan,
    StepLosses,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::LoraSet;
use candle_gen::train::perceptual::{Parameterization, PerceptualPath};
use candle_gen::train::taehv::TaehvConfig;
use candle_gen::{CandleError, LatentDecoder, Result};

use candle_gen_qwen_image::vae::{QwenVae, QwenVaeEncoder};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::pipeline::krea_cfg_combine;
use crate::schedule::{dynamic_mu, krea_sigmas};
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::{KreaTrainDit, KREA_ATTN_TARGETS};

/// VAE spatial downscale (the latent is image/8 per side) and latent channel count — the Turbo
/// inference constants ([`crate::pipeline`]) the preview-sample render mirrors.
const SPATIAL_SCALE: u32 = 8;
const LATENT_CHANNELS: usize = 16;

/// Max preview prompts pre-encoded + rendered per sample cadence (sc-8650). Matches the
/// `SAMPLE_PROMPT_CAP` the shared preview contract documents.
const SAMPLE_PROMPT_CAP: usize = 4;

/// Registry id for the trainable Krea 2 **Raw** base (the undistilled 12B checkpoint LoRAs train on),
/// distinct from the `krea_2_turbo` inference id — mirrors the MLX trainer (sc-7577).
pub const KREA_2_RAW_ID: &str = "krea_2_raw";

/// Error-message prefix shared by [`validate_flow_match_request`] and the driver's `no usable dataset
/// items` guard.
const LABEL: &str = "krea trainer";

// The prompt-token cap for caption caching is the crate's single canonical
// [`crate::pipeline::MAX_TEXT_TOKENS`] (sc-11205 / F-120) — no longer a second definition here, so the
// training/control lanes can never drift from the inference cap.
use crate::pipeline::MAX_TEXT_TOKENS;

/// `(x_t, target, timestep)` for one sample at flow-match `σ`: delegates the latent mix
/// (`x_t = (1−σ)·x0 + σ·noise`, `target = noise − x0`) to the shared
/// [`flow_match::build_batch`](candle_gen::train::flow_match::build_batch) and appends Krea's raw-σ
/// timestep convention `timestep = σ` (the inference `TimestepConvention::Sigma`, NOT Z-Image's `1 − σ`).
fn build_batch(x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<(Tensor, Tensor, f32)> {
    let (x_t, target) = flow_match::build_batch(x0, noise, sigma as f64)?;
    Ok((x_t, target, sigma))
}

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at
/// `sigma`, predict the **raw** velocity through the trainable DiT, regress it toward `noise − x0`, and
/// return `(loss, grads)` keyed by the adapter `Var`s. A free function so the parity tests can drive it
/// against a tiny DiT.
///
/// `x0`/`noise` are the cached `[1, 16, h, w]` clean latent + the per-step noise (f32); `cap` is the
/// cached `(L, num_text_layers, text_hidden)` caption stack — unsqueezed to the DiT's batched `context`.
///
/// `use_checkpoint` selects the gradient-checkpointed backward over the dense `loss.backward()`. Because
/// every adapter lives in the checkpointed `blocks` stack, the frozen front-end is run once via
/// [`KreaTrainDit::forward_pre_main`] and detached at the joint-sequence boundary — no pre-main stitch.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &KreaTrainDit,
    lora_vars: &[Var],
    x0: &Tensor,
    cap: &Tensor,
    sigma: f32,
    noise: &Tensor,
    mae: bool,
    mask_weight: Option<&Tensor>,
    compute_dtype: DType,
    use_checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (losses, grads) = compute_step_loss_grads(
        dit,
        lora_vars,
        x0,
        cap,
        sigma,
        noise,
        mae,
        mask_weight,
        compute_dtype,
        use_checkpoint,
        None,
    )?;
    Ok((losses.total, grads))
}

/// The step's loss terms on the DiT's raw velocity `v` (epic 2123 E8): the (subject-mask weighted)
/// velocity regression when the diffusion term contributes, and — on a planned step with aux losses
/// — the weighted perceptual term on the x0 estimate `x_t − σ·v` (`[1, 16, h, w]`, already the
/// decoder's NCHW layout), decoded by TAEW2.1. Summed by [`combine_terms`]; no aux step ⇒ exactly the
/// legacy loss tensor.
#[allow(clippy::too_many_arguments)]
fn step_loss(
    v: &Tensor,
    target: &Tensor,
    x_t_f32: &Tensor,
    sigma: f32,
    mask_weight: Option<&Tensor>,
    mae: bool,
    aux: Option<&AuxStep<'_>>,
) -> Result<(Tensor, StepLosses)> {
    let (diffusion_on, aux_on) = step_terms(aux);
    let diffusion = if diffusion_on {
        Some(weighted_velocity_loss(v, target, mask_weight, mae)?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma }
                .recover_x0(x_t_f32, &v.to_dtype(DType::F32)?)?;
            a.aux_loss(&x0_hat)?
        }
        _ => None,
    };
    combine_terms(diffusion, aux_term)
}

/// [`compute_loss_grads`] with the step's planned perceptual terms (epic 2123 E8): an aux-only step
/// does not compute the diffusion term; both the dense and the checkpointed backward (the aux term
/// rides in the final checkpoint segment, after the velocity head) carry it. `aux = None` ⇒ the
/// legacy graph.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    dit: &KreaTrainDit,
    lora_vars: &[Var],
    x0: &Tensor,
    cap: &Tensor,
    sigma: f32,
    noise: &Tensor,
    mae: bool,
    mask_weight: Option<&Tensor>,
    compute_dtype: DType,
    use_checkpoint: bool,
    aux: Option<&AuxStep<'_>>,
) -> Result<(StepLosses, GradStore)> {
    let device = x0.device();
    let (x_t_f32, target, timestep) = build_batch(x0, noise, sigma)?;
    let x_t = x_t_f32.to_dtype(compute_dtype)?;
    let context = cap.unsqueeze(0)?; // (L, n, d) -> (1, L, n, d)
    let t = Tensor::from_vec(vec![timestep], (1,), device)?;

    if use_checkpoint {
        // The breakdown of the final segment's last evaluation, kept for reporting.
        let breakdown = std::cell::Cell::new(None);
        let (combined, ctx) = dit.forward_pre_main(&x_t, &t, &context)?;
        let mut segs = dit.main_layer_segments(&ctx);
        // Final segment: the post-main head + the (raw-velocity) flow-match regression (and the
        // step's aux term) -> [loss].
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        let x_t_ref = &x_t_f32;
        let breakdown_ref = &breakdown;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?;
            let (loss, losses) =
                step_loss(&v, &target_owned, x_t_ref, sigma, mask_weight, mae, aux)
                    .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
            breakdown_ref.set(Some(losses));
            Ok(vec![loss])
        }));
        // Seed the checkpointed chain with the detached joint-sequence boundary (no adapters upstream).
        let combined_d = combined.detach();
        let (loss_val, grads) =
            checkpointed_backward(&segs, std::slice::from_ref(&combined_d), lora_vars)?;
        drop(segs);
        let losses = breakdown.get().ok_or_else(|| {
            CandleError::Msg(format!("{LABEL}: the final checkpoint segment never ran"))
        })?;
        Ok((
            StepLosses {
                total: loss_val,
                ..losses
            },
            grads,
        ))
    } else {
        // Dense backward: one monolithic `loss.backward()`. Raw DiT velocity (no negation).
        let v = dit.forward(&x_t, &t, &context)?;
        let (loss, losses) = step_loss(&v, &target, &x_t_f32, sigma, mask_weight, mae, aux)?;
        let grads = loss.backward()?;
        Ok((losses, grads))
    }
}

/// Krea's latent family for the shared aux-loss builder (epic 2123 E8): the Qwen-Image VAE's
/// 16-channel `(μ − latents_mean)/latents_std` latent (`QwenVaeEncoder::encode`, the DiT's space),
/// decoded by TAEW2.1 — upstream's TAEHV checkpoint for Qwen-Image, which takes that normalized
/// latent with no scale/shift. A still image is one `T = 1` clip.
fn krea_decoder() -> candle_gen_perceptual::DecoderSpec {
    candle_gen_perceptual::DecoderSpec::Taehv {
        name: "TAEW2.1",
        config: TaehvConfig::taew2_1(),
    }
}

/// The epic-2123 perceptual path through the shared builder: `None` when no aux loss is enabled.
fn load_perceptual_path(cfg: &TrainingConfig, device: &Device) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(
        cfg,
        &candle_gen_perceptual::AuxLossContext {
            label: LABEL,
            decoder: krea_decoder(),
            device,
            latent_lpips: None,
        },
    )
}

/// Epic 2123 E7: refuse a depth job whose resident DiT (`base_bytes`, the transformer's on-disk
/// weights — the lower bound the aux models stack on) plus TAEW2.1 + the losses' frozen models at the
/// largest bucket, with one cached reference per (item, bucket) entry over `items` dataset items,
/// exceeds `budget_bytes`. No-op when no aux loss is enabled (the dense/checkpointed choice does
/// not change the aux models' resident cost, so this runs on both paths).
fn check_perceptual_memory(
    cfg: &TrainingConfig,
    items: usize,
    base_bytes: u64,
    budget_bytes: u64,
) -> Result<()> {
    let edges = bucket_edges(cfg);
    let aux = candle_gen_perceptual::perceptual_footprint(
        cfg,
        &krea_decoder(),
        candle_gen_perceptual::AuxGeometry::image(
            edges.iter().copied().max().unwrap_or(0),
            items * edges.len(),
        ),
    );
    if aux == 0 {
        return Ok(());
    }
    flow_match::check_aux_memory(LABEL, base_bytes, aux, budget_bytes)
}

/// Tokenize `caption` + encode it through the Qwen3-VL text encoder to the cached conditioning stack
/// `(L, num_text_layers, text_hidden)` at f32 — the exact tokenizer + select-layer stack the inference
/// [`crate::pipeline`] uses (parity), minus the device-dtype cast (caching keeps f32).
pub(crate) fn encode_caption(
    tok: &KreaTokenizer,
    te: &KreaTextEncoder,
    caption: &str,
) -> Result<Tensor> {
    let ids = tok.encode_prompt(caption, MAX_TEXT_TOKENS)?;
    let enc = te.forward(&ids)?; // (1, L, num_text_layers, text_hidden)
    Ok(enc.squeeze(0)?.to_dtype(DType::F32)?)
}

/// The Krea preview-sample render state (sc-8650) — everything [`KreaTrainer::render_sample`] needs to
/// run the family's **Raw** CFG denoise on the **in-progress** trainable DiT, built once in
/// [`KreaTrainer::cache`] while the text encoder is still resident. The trainer's loaded weights are the
/// undistilled Raw checkpoint, so the preview mirrors the MLX trainer's Raw recipe (dynamic-`mu`
/// schedule + classifier-free guidance), NOT the deployed few-step CFG-free `krea_2_turbo` render:
///
///  * `contexts` — the per-prompt pre-encoded Qwen3-VL conditioning, each `(L, num_text_layers,
///    text_hidden)` at f32 (exactly what [`encode_caption`] returns and [`KreaTrainDit::forward`]
///    consumes once unsqueezed to a batch axis), 1:1 with [`SamplePlan::prompts`].
///  * `ctx_neg` — one shared **empty-prompt** unconditional context (same shape/dtype) for the CFG
///    branch; pre-encoded once here while the encoder is resident (mirrors the MLX trainer).
///  * `vae` — the resident Qwen-Image VAE **decoder** (`Arc` as inference holds it); the cache pass
///    loads only the encoder, so the decoder is loaded here for the preview path.
///  * `edge` — the square preview edge: the largest training-bucket edge ([`bucket_edges`], just
///    `bucket_resolution(cfg.resolution)` with buckets off — the edge the cached latents use) the
///    seeded preview noise is shaped at.
pub struct KreaSampleState {
    contexts: Vec<Tensor>,
    ctx_neg: Tensor,
    vae: Arc<QwenVae>,
    edge: u32,
}

/// Seeded initial Gaussian latent noise `[1, 16, edge/8, edge/8]` (f32) for a preview render — the
/// training-side twin of [`crate::pipeline`]'s `init_noise`, square at the bucketed training `edge`
/// (sc-8650). Deterministic launch-portable CPU RNG (sc-3673), mirroring the inference path.
fn sample_noise_latent(edge: u32, seed: u64, device: &Device) -> Result<Tensor> {
    let lat = (edge / SPATIAL_SCALE) as usize;
    let n = LATENT_CHANNELS * lat * lat;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat, lat), &Device::Cpu)?.to_device(device)?)
}

/// VAE-decode a final preview latent `[1, 16, H/8, W/8]` → RGB8 [`Image`] — the training-side twin of
/// [`crate::pipeline`]'s `decode` (`QwenVae::decode` de-normalizes internally and returns `[1, 3, H, W]`
/// in `[-1, 1]`; the `(x+1)·127.5` is the reference `clamp(-1,1)·0.5 + 0.5` denormalize) (sc-8650).
fn decode_preview(vae: &QwenVae, lat: &Tensor) -> Result<Image> {
    let decoded = LatentDecoder::decode(vae, lat)?.to_dtype(DType::F32)?; // [1, 3, H, W] in [-1, 1]
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let img = candle_gen::round_rgb8(&scaled)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "krea: preview decode expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Identity + capabilities of the candle Krea trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: KREA_2_RAW_ID,
        family: "krea_2",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // sc-10894 lockstep catch-up: gen-core gained `TrainerDescriptor.supports_control` on mlx-gen
        // main between this crate's prior gen-core pin and the footprint-seam re-pin; the base LoRA/LoKr
        // trainer does not train a control branch (mirrors mlx-gen-krea's `false`).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets — one cached latent per bucket edge, walked
        // by the shared driver's `BucketSchedule`.
        // sc-24828 (epic 2123): subject-masked loss — a per-bucket weight map cached next to each
        // latent.
        // sc-24830 (epic 2123): depth anchoring on the dense and checkpointed backward — the shared
        // decoded-x0 perceptual path through TAEW2.1.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring
            // (decoded-x0 pixel losses through this trainer's x0 decoder).
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// A loaded candle Krea trainer. Loading is **lazy** (no file I/O — mirrors the SDXL/Z-Image trainers):
/// the heavy VAE encoder / text encoder / DiT are built inside [`train`](Trainer::train).
pub struct KreaTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Krea trainer from a [`LoadSpec`] whose `weights` is the Krea-2-Raw
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "krea_2_raw trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(KreaTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// `register_trainer!` defines the explicit trainer constant and bridges the crate's rich `Result`
// into `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl Trainer for KreaTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
        // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA
        // adapter the caller did not ask for (F-006/F-055).
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        validate_flow_match_request(req, LABEL).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        validate_flow_match_request(req, LABEL)?;
        run_flow_match_training(self, req, on_progress).map_err(Into::into)
    }
}

impl FlowMatchTrainer for KreaTrainer {
    type Dit = KreaTrainDit;
    /// `(x0 latent [1,16,h,w], caption stack (L, num_text_layers, text_hidden), subject-mask loss
    /// weight)`, all f32; the weight (broadcast to the latent shape) is `None` unless subject-masked
    /// loss is on (sc-24828).
    type Cached = (Tensor, Tensor, Option<Tensor>);
    type Aux = ();
    /// Preview-sample render state: per-prompt pre-encoded conditioning + resident VAE decoder + the
    /// training-resolution edge (sc-8650).
    type SampleState = KreaSampleState;
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        &KREA_ATTN_TARGETS
    }

    /// Epic 2123 E7: the aux-model memory guard (no-op with nothing enabled) — on every path.
    fn preflight(&self, req: &TrainingRequest) -> Result<()> {
        if !candle_gen_perceptual::any_aux_loss(&req.config) {
            return Ok(());
        }
        let base = flow_match::component_bytes(&self.root, "transformer", LABEL)?;
        check_perceptual_memory(
            &req.config,
            req.items.len(),
            base,
            flow_match::device_training_budget_bytes(&self.device, LABEL),
        )
    }

    fn perceptual_path(
        &self,
        req: &TrainingRequest,
        device: &Device,
    ) -> Result<Option<PerceptualPath>> {
        load_perceptual_path(&req.config, device)
    }

    /// The cached clean latent is already the decoder's NCHW `[1, 16, h, w]` (f32).
    fn reference_latent(&self, cached: &Self::Cached, _aux: &()) -> Result<Tensor> {
        Ok(cached.0.clone())
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<Self::Cached>, (), SamplePlan<KreaSampleState>)> {
        // sc-2127: one training edge per resolution bucket (just `[resolution]` when buckets are off);
        // previews render at the largest (epic 2123 E7).
        let edges = bucket_edges(&req.config);
        let edge = edges.iter().copied().max().unwrap_or(0);
        let vae_encoder = QwenVaeEncoder::new(flow_match::component_vb(
            &self.root,
            "vae",
            device,
            DType::F32,
            LABEL,
        )?)?;
        let tokenizer = KreaTokenizer::from_snapshot(&self.root, device)?;
        let te_cfg = KreaTeConfig::from_snapshot(&self.root)?;
        let te_w = Weights::from_dir(&self.root.join("text_encoder"), device, DType::F32)?;
        let text_encoder =
            KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

        let total = req.items.len() as u32;
        // Item-major over the bucket edges: `cache[item * edges.len() + bucket]` — the layout the
        // driver's `BucketSchedule` indexes (sc-2127).
        let mut cache: Vec<Self::Cached> = Vec::with_capacity(req.items.len() * edges.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let cap = encode_caption(&tokenizer, &text_encoder, &item.caption)?;
            // sc-24828: the item's subject mask is read + checked once, then resampled per bucket
            // onto that bucket's latent grid (`None` when masked loss is off).
            let mask = PreparedSubjectMask::load_if_enabled(
                LABEL,
                item,
                req.config.subject_mask_loss.as_ref(),
            )?;
            let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
            for &edge in &edges {
                let img = square_image_tensor(&square, edge, device)?;
                let x0 = vae_encoder.encode(&img)?; // (1, 16, edge/8, edge/8), already normalized
                                                    // `decode_square` centre-crops to a square, so the mask takes the same crop.
                let mask_weight = prepared_subject_mask_weight(
                    LABEL,
                    mask.as_ref(),
                    CropBox::center_square,
                    x0.dims(),
                    device,
                )?;
                cache.push((x0, cap.clone(), mask_weight));
            }
        }

        // Preview samples (sc-8650) — while the text encoder is STILL resident, pre-encode up to
        // `SAMPLE_PROMPT_CAP` of the configured prompts with the same `encode_caption` the cache loop
        // uses (train/infer conditioning parity), and load a resident VAE *decoder* (the cache pass
        // loaded only the encoder). The driver renders these from the in-progress adapter each cadence.
        let sample_plan = if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
            let prompts: Vec<String> = req
                .config
                .sample_prompts
                .iter()
                .take(SAMPLE_PROMPT_CAP)
                .cloned()
                .collect();
            let contexts = prompts
                .iter()
                .map(|p| encode_caption(&tokenizer, &text_encoder, p))
                .collect::<Result<Vec<Tensor>>>()?;
            // Shared empty-prompt unconditional context for the CFG preview branch (mirrors the MLX
            // trainer) — pre-encoded here while the text encoder is still resident.
            let ctx_neg = encode_caption(&tokenizer, &text_encoder, "")?;
            let vae = Arc::new(crate::vae::load_vae(&self.root, device)?);
            SamplePlan {
                prompts,
                state: Some(KreaSampleState {
                    contexts,
                    ctx_neg,
                    vae,
                    edge,
                }),
            }
        } else {
            SamplePlan::disabled()
        };

        // Encoders are dead weight once cached + previews pre-encoded — drop them before the DiT
        // (working set) loads. The resident VAE *decoder* lives on in the sample plan's state.
        drop(text_encoder);
        drop(vae_encoder);
        Ok((cache, (), sample_plan))
    }

    fn build_dit(&self, req: &TrainingRequest, device: &Device) -> Result<KreaTrainDit> {
        let compute_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);
        let dit_cfg = Krea2Config::from_snapshot(&self.root)?;
        let dit_w = Weights::from_dir(&self.root.join("transformer"), device, compute_dtype)?;
        Ok(KreaTrainDit::load(&dit_w, &dit_cfg)?)
    }

    fn micro_step(
        &self,
        dit: &KreaTrainDit,
        vars: &[Var],
        cached: &Self::Cached,
        _aux: &(),
        cfg: &TrainingConfig,
        step: u32,
        sample: flow_match::StepSample<'_>,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let (x0, cap, mask_weight) = cached;
        let mut sigma = flow_match::sample_unit_timestep(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            flow_match::timestep_seed(cfg.seed, step),
        );
        // Epic 2123 E8: an aux-only step trains at σ remapped into the loss window.
        let aux = sample.plan(sigma)?;
        if let Some(a) = &aux {
            sigma = a.noise_level();
        }
        let noise =
            flow_match::sample_noise(x0.dims(), flow_match::noise_seed(cfg.seed, step), device)?;
        let (losses, grads) = compute_step_loss_grads(
            dit,
            vars,
            x0,
            cap,
            sigma,
            &noise,
            flow_match::is_mae(cfg),
            mask_weight.as_ref(),
            flow_match::parse_compute_dtype(&cfg.train_dtype),
            cfg.gradient_checkpointing,
            aux.as_ref(),
        )?;
        Ok((losses.total, grads))
    }

    /// Render preview prompt `index` from the **in-progress** trainable DiT (sc-8650) — the training-side
    /// mirror of the MLX trainer's **Raw** preview render (`mlx-gen-krea`). The trainer's loaded weights
    /// are the undistilled **Raw** DiT, so the preview uses the Raw sampler recipe — the
    /// resolution-dynamic `mu` schedule ([`dynamic_mu`] over the image-sequence length `(edge/16)²`) and
    /// classifier-free guidance — NOT the deployed few-step CFG-free `krea_2_turbo` render
    /// ([`crate::pipeline`]). A CFG-free Turbo-schedule render of the Raw checkpoint collapses to
    /// washed-out, unrepresentative output. Krea consumes the **raw** velocity at timestep `σ`
    /// ([`TimestepConvention::Sigma`], no negation); the CFG mix is the shared reference
    /// [`krea_cfg_combine`] `v = v_cond + guidance·(v_cond − v_uncond)` (`guidance ≤ 0` ⇒
    /// conditional-only, a single forward). Best-effort: any error here is logged + skipped by the
    /// driver, never aborting the run.
    fn render_sample(
        &self,
        dit: &KreaTrainDit,
        state: &KreaSampleState,
        index: usize,
        cfg: &TrainingConfig,
        seed: u64,
    ) -> Result<Image> {
        let device = &self.device;
        let steps = (cfg.sample_steps.max(1)) as usize;
        // Raw resolution-dynamic `mu` schedule (image-sequence length = `(edge/16)²`), mirroring the MLX
        // trainer — the Raw checkpoint is not TDM-distilled, so the fixed Turbo `mu` does not apply.
        let img_seq = (state.edge as f64 / 16.0).powi(2);
        let sigmas = krea_sigmas(steps, dynamic_mu(img_seq));

        let noise = sample_noise_latent(state.edge, seed, device)?;
        // The DiT's `text_fusion` consumes a batched `(1, L, n, d)` context — the cached
        // `encode_caption` output is `(L, n, d)`, so unsqueeze the batch axis exactly as
        // `compute_loss_grads` does before its own forward.
        let ctx_pos = state.contexts[index].unsqueeze(0)?;
        let ctx_neg = state.ctx_neg.unsqueeze(0)?;
        let guidance = cfg.sample_guidance_scale;

        // A preview need not honor cancel mid-denoise — a fresh never-cancel flag (the trainer's
        // `req.cancel` is only available in `cache`, not here).
        let cancel = CancelFlag::new();
        let mut on_progress = |_: Progress| {};
        let lat = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &cancel,
            &mut on_progress,
            None,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let v_cond = dit.forward(x, &t, &ctx_pos)?;
                // CFG via the shared [`krea_cfg_combine`] (reference `sampling.py:129`
                // `v_cond + g·(v_cond − v_uncond)`, one source of truth with the Raw inference path).
                // Prior to sc-9994 this used the standard `v_uncond + g·Δ` gated on `guidance > 1.0`,
                // which at the shared default `sample_guidance_scale = 1.0` collapsed to exactly `v_cond`
                // — zero effective CFG, the washed-out previews (the candle twin of mlx-gen sc-10009).
                // `guidance ≤ 0` still collapses to the bare conditional velocity (one forward).
                let v = if guidance > 0.0 {
                    let v_uncond = dit.forward(x, &t, &ctx_neg)?;
                    krea_cfg_combine(&v_cond, &v_uncond, guidance)?
                } else {
                    v_cond
                };
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        decode_preview(&state.vae, &lat)
    }

    /// Stamp `baseModel`/`family` provenance into the adapter header so the Turbo cross-apply policy
    /// (family-match) can validate it.
    fn save(&self, set: &LoraSet, path: &Path) -> Result<()> {
        let mut meta = HashMap::new();
        meta.insert("baseModel".to_string(), KREA_2_RAW_ID.to_string());
        meta.insert("family".to_string(), "krea_2".to_string());
        flow_match::save_adapter(set, &meta, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfix::{randn_seeded, tiny_batch, tiny_batch_seeded, tiny_dit, tiny_dit_seeded};
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};

    /// `build_batch`: `x_t = (1−σ)x0 + σ·noise`, `target = noise − x0`, `timestep = σ` (the Krea raw-σ
    /// convention — NOT the Z-Image `1−σ` — layered over the shared `flow_match::build_batch`).
    #[test]
    fn build_batch_math() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target, timestep) = build_batch(&x0, &noise, 0.25).unwrap();
        // x_t = 0.75·[2,4] + 0.25·[1,0] = [1.75, 3.0]; target = [1-2, 0-4] = [-1, -4]; t = σ = 0.25.
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        assert!((timestep - 0.25).abs() < 1e-6);
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (backprop reaches the
    /// LoRA seam through the composable softmax/RMSNorm of the single-stream blocks + final layer).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let (mut dit, c, path) = tiny_dit(&tmp);
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        for (i, v) in set.vars.iter().enumerate() {
            let g = grads
                .get(v.as_tensor())
                .unwrap_or_else(|| panic!("adapter var {i} has no gradient"));
            assert!(
                g.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite()),
                "var {i} gradient has non-finite entries"
            );
        }
        // The LoRA set covers every attention projection (4) in the single block: 4 targets × 2 factors.
        assert_eq!(
            set.vars.len(),
            4 * c.num_layers * 2,
            "two factors per attention target"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The correctness gate for the `gradient_checkpointing` lever: the checkpointed backward over the
    /// single-stream `blocks` must reproduce the dense `loss.backward()` grads (mod float reassociation).
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let (mut dit, c, path) = tiny_dit(&tmp);
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);

        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            true,
        )
        .unwrap();

        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        let grad_vec = |g: &GradStore, v: &Var| {
            g.get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let mut saw_nonzero = false;
        for (idx, v) in set.vars.iter().enumerate() {
            assert!(
                g_d.get(v.as_tensor()).is_some() && g_c.get(v.as_tensor()).is_some(),
                "var {idx} missing a gradient (dense or checkpoint)"
            );
            let a = grad_vec(&g_d, v);
            let b = grad_vec(&g_c, v);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {idx} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
        let _ = std::fs::remove_file(path);
    }

    /// sc-24828: subject-masked loss on both backward paths. An all-ones map is the unweighted loss;
    /// an all-zero map zeroes the loss AND every adapter gradient (dense and checkpointed — a path
    /// that dropped the weight would train on the background); a half map matches across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let (mut dit, c, path) = tiny_dit(&tmp);
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);
        let shape = x0.dims().to_vec();
        let map = |w: &[f32]| flow_match::subject_mask_weight(w, 4, 4, &shape, &dev).unwrap();
        let run = |weight: Option<&Tensor>, ckpt: bool| {
            compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                weight,
                DType::F32,
                ckpt,
            )
            .unwrap()
        };
        let (plain, _) = run(None, false);
        let ones = map(&[1.0; 16]);
        assert!((run(Some(&ones), false).0 - plain).abs() < 1e-6);
        let zeros = map(&[0.0; 16]);
        for ckpt in [false, true] {
            let (loss, grads) = run(Some(&zeros), ckpt);
            assert_eq!(
                loss, 0.0,
                "ckpt={ckpt}: an all-background map must zero the loss"
            );
            for v in &set.vars {
                if let Some(g) = grads.get(v.as_tensor()) {
                    let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert!(
                        g.iter().all(|x| *x == 0.0),
                        "ckpt={ckpt}: nonzero adapter grad"
                    );
                }
            }
        }
        let half: Vec<f32> = (0..16).map(|i| if i % 4 < 2 { 1.0 } else { 0.0 }).collect();
        let half = map(&half);
        let (dense, _) = run(Some(&half), false);
        let (ckpt, _) = run(Some(&half), true);
        assert!(dense > 0.0 && dense < plain, "{dense} vs {plain}");
        assert!(
            (dense - ckpt).abs() < 1e-4,
            "dense {dense} vs checkpoint {ckpt}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// The candle Krea LoRA trainer declares subject-masked loss (sc-24828).
    #[test]
    fn descriptor_declares_subject_mask_loss() {
        assert!(trainer_descriptor().techniques.subject_mask_loss);
    }

    /// A few optimizer steps over the tiny DiT lower the loss on the same fixed batch — the step
    /// actually descends the flow-match objective, end to end through the harness.
    #[test]
    fn optimizer_steps_descend() {
        let dev = Device::Cpu;
        // Draw the ENTIRE fixture — base DiT weights, LoRA adapter nudge, and probe batch — from one
        // seeded `StdRng`, for the same reason as `control_train::control_trainer_descends` (sc-10794):
        // candle's CPU `randn` is unseedable (it pulls the process-global `rand::rng()`), so before this
        // every draw was nondeterministic and a marginal 6-step descent could flip sign on an unlucky
        // init (or on ubuntu's float reassociation vs macos), red-failing CI. A distinct fixed seed
        // makes the loss trajectory reproducible run-to-run and platform-to-platform; the larger step
        // budget then buys an unambiguous drop (see the relative-floor assert below). Seed is
        // 10794-adjacent but distinct from the sibling tests so they don't share a trajectory.
        let mut rng = StdRng::seed_from_u64(10796);
        let tmp = tempfile::tempdir().unwrap();
        let (mut dit, c, path) = tiny_dit_seeded(&tmp, &mut rng);
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&randn_seeded(&mut rng, 0.0, 0.02, v.as_tensor().dims()))
                .unwrap();
        }
        let (x0, cap, noise) = tiny_batch_seeded(&c, &mut rng);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let (loss0, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
        )
        .unwrap();
        let mut grads = grads;
        for _ in 0..60 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            let (_l, g) = compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                DType::F32,
                false,
            )
            .unwrap();
            grads = g;
        }
        let (loss1, _) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
        )
        .unwrap();
        // A correctly working LoRA adapter descends the fixed-batch loss by a wide margin over these
        // 60 AdamW steps. Assert a >=10% relative drop rather than a bare `loss1 < loss0`: the 0.90
        // bar sits far clear of the real ratio, so cross-platform float reassociation (the ubuntu-vs-
        // macos delta that flaked the sibling `control_trainer_descends`) cannot lift it over the bar,
        // while an adapter that stopped learning (ratio ~1.0) still fails hard. This stays a genuine
        // descent gate, not a `<= before + epsilon` no-op.
        assert!(
            loss1 < loss0 * 0.9,
            "steps on a fixed batch should lower the loss by >=10%: {loss0} -> {loss1} (ratio {})",
            loss1 / loss0
        );
        let _ = std::fs::remove_file(path);
    }

    /// The trainer resolves through the explicit family registry as the candle Krea
    /// trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer(KREA_2_RAW_ID, &spec)
            .expect("candle krea trainer is registered");
        assert_eq!(t.descriptor().id, KREA_2_RAW_ID);
        assert_eq!(t.descriptor().family, "krea_2");
        assert_eq!(t.descriptor().backend, "candle");
        assert!(t.descriptor().supports_lora);
        assert!(t.descriptor().supports_lokr);
        assert!(t.descriptor().techniques.resolution_buckets);
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and an
    /// unrecognized timestep/loss knob — before any load (now via the shared
    /// `flow_match::validate_flow_match_request`).
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer(KREA_2_RAW_ID, &spec)
            .unwrap();

        let item = TrainingItem {
            image_path: "/img.png".into(),
            caption: "x".into(),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: None,
        };
        let base = TrainingRequest {
            items: vec![item],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        assert!(t.validate(&base).is_ok());

        let bad = |mutate: &dyn Fn(&mut TrainingRequest)| {
            let mut r = base.clone();
            mutate(&mut r);
            assert!(t.validate(&r).is_err());
        };
        bad(&|r| r.items.clear());
        bad(&|r| r.config.rank = 0);
        bad(&|r| r.config.steps = 0);
        bad(&|r| r.config.optimizer = "lion".into());
        bad(&|r| r.config.timestep_type = "bogus".into());
        bad(&|r| r.config.timestep_bias = "bogus".into());
        bad(&|r| r.config.loss_type = "huber".into());
        // A `_`/case-normalized spelling of a recognized value is accepted.
        let mut ok = base.clone();
        ok.config.timestep_type = "Weighted".into();
        ok.config.timestep_bias = "high-noise".into();
        assert!(t.validate(&ok).is_ok());
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the candle Krea step seam on the tiny DiT
/// (`testfix::tiny_dit`, 4-channel latent `[1, 4, 4, 4]`) with a random-init tiny-width TAEHV carrying
/// TAEW2.1's hyperparameters at 4 latent channels and a random-init tiny Depth-Anything-V2, planned
/// through the shared flow-match `AuxDriver` exactly as the driver plans `micro_step`. CPU.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use crate::testfix::{tiny_batch, tiny_dit};
    use candle_gen::gen_core::train::{AuxLossSchedule, ResolutionBucket};
    use candle_gen::gen_core::BucketSchedule;
    use candle_gen::train::flow_match::AuxDriver;
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::perceptual::AuxLoss;
    use candle_gen::train::taehv::{synthetic_taehv_weights, TaehvDecoder};

    fn schedule() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    fn path() -> PerceptualPath {
        let tae = TaehvConfig {
            latent_channels: 4,
            channels: [8, 6, 4, 4],
            ..TaehvConfig::taew2_1()
        };
        let dec = TaehvDecoder::from_weights(
            &synthetic_taehv_weights(&tae, 11, &Device::Cpu).unwrap(),
            tae,
        )
        .unwrap();
        let loss = candle_gen_depth::anchor::tiny_depth_anchor_loss(12, &Device::Cpu).unwrap();
        PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule: schedule(),
                loss: Box::new(loss),
            }],
        )
        .unwrap()
    }

    fn one_item_schedule() -> BucketSchedule {
        BucketSchedule::new(
            1,
            &[ResolutionBucket {
                resolution: 32,
                repeats: 1,
            }],
            7,
        )
    }

    fn grad_bits(g: &GradStore, vars: &[Var]) -> Vec<Vec<u32>> {
        vars.iter()
            .map(|v| {
                g.get(v.as_tensor())
                    .map(|t| {
                        t.flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    /// AC (a)+(b), dense and checkpointed: the driver's step 2 (the item's 2nd window) is a depth
    /// step — no diffusion term, total == the weighted depth term, a nonzero finite gradient on the
    /// zero-init LoRA-B factors — and step 1 carries no depth term. Mutation: compute the diffusion
    /// term unconditionally in `step_loss` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only_on_both_paths() {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let (mut dit, c, _p) = tiny_dit(&tmp);
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        let (x0, cap, noise) = tiny_batch(&c);
        let sched = one_item_schedule();
        for ckpt in [false, true] {
            let mut driver =
                AuxDriver::prepare(path(), 1, |_| Ok(x0.clone()), &sched, 1, 0).unwrap();
            let s1 = driver.sample(1, &sched).plan(0.5).unwrap().unwrap();
            let (diff, _) = compute_step_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                DType::F32,
                ckpt,
                Some(&s1),
            )
            .unwrap();
            assert_eq!(diff.aux, None, "ckpt={ckpt}");
            assert_eq!(Some(diff.total), diff.diffusion);
            let s2 = driver.sample(2, &sched).plan(0.5).unwrap().unwrap();
            assert!(!s2.diffusion());
            let (depth, g) = compute_step_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &cap,
                s2.noise_level(),
                &noise,
                false,
                None,
                DType::F32,
                ckpt,
                Some(&s2),
            )
            .unwrap();
            assert_eq!(depth.diffusion, None, "ckpt={ckpt}");
            let a = depth.aux.expect("depth term");
            assert!(a > 0.0 && a.is_finite(), "ckpt={ckpt}: {a}");
            assert!((depth.total - a).abs() <= 1e-6 * a.abs(), "ckpt={ckpt}");
            // LoRA B is the second factor of each (A, B) pair.
            let gb: f32 = set
                .vars
                .iter()
                .skip(1)
                .step_by(2)
                .map(|v| {
                    g.get(v.as_tensor())
                        .map(|t| {
                            t.abs()
                                .unwrap()
                                .sum_all()
                                .unwrap()
                                .to_scalar::<f32>()
                                .unwrap()
                        })
                        .unwrap_or(0.0)
                })
                .sum();
            assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B grad {gb}");
        }
    }

    /// AC (c): depth off ⇒ bit-identical to the legacy step (its body reproduced here), and a
    /// diffusion-only planned step equals it too. Mutation: scale the diffusion loss (×1.0001) in
    /// `step_loss` ⇒ red.
    #[test]
    fn depth_off_is_bit_identical_to_the_legacy_step() {
        assert!(
            load_perceptual_path(&TrainingConfig::default(), &Device::Cpu)
                .unwrap()
                .is_none()
        );
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let (mut dit, c, _p) = tiny_dit(&tmp);
        let suffixes: Vec<String> = KREA_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for (i, v) in set.vars.iter().enumerate() {
            v.set(
                &candle_gen::train::taehv::splitmix_uniform(
                    v.as_tensor().dims(),
                    100 + i as u64,
                    0.02,
                    0.0,
                    &dev,
                )
                .unwrap(),
            )
            .unwrap();
        }
        let (x0, cap, noise) = tiny_batch(&c);
        let (off, g_off) = compute_step_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
            None,
        )
        .unwrap();
        assert_eq!(off.aux, None);
        // The pre-sc-24830 dense body.
        let (x_t, target, timestep) = build_batch(&x0, &noise, 0.5).unwrap();
        let t = Tensor::from_vec(vec![timestep], (1,), &dev).unwrap();
        let v = dit.forward(&x_t, &t, &cap.unsqueeze(0).unwrap()).unwrap();
        let loss = weighted_velocity_loss(&v, &target, None, false).unwrap();
        let legacy = loss.to_scalar::<f32>().unwrap();
        let g_legacy = loss.backward().unwrap();
        assert_eq!(off.total.to_bits(), legacy.to_bits());
        assert_eq!(
            grad_bits(&g_off, &set.vars),
            grad_bits(&g_legacy, &set.vars)
        );
        let sched = one_item_schedule();
        let mut driver = AuxDriver::prepare(path(), 1, |_| Ok(x0.clone()), &sched, 1, 0).unwrap();
        let s1 = driver.sample(1, &sched).plan(0.5).unwrap().unwrap();
        let (on, g_on) = compute_step_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &cap,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
            Some(&s1),
        )
        .unwrap();
        assert_eq!(on, off);
        assert_eq!(grad_bits(&g_on, &set.vars), grad_bits(&g_off, &set.vars));
    }

    /// AC (d), E7: depth grows the guarded footprint by TAEW2.1 + DA2 (more for Large), and the guard
    /// refuses at a synthetic budget between base and base+aux — the same guard on both backward
    /// paths (`preflight` runs it regardless of `gradient_checkpointing`). Mutation: compare `base`
    /// alone ⇒ red.
    #[test]
    fn memory_guard_counts_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule();
        let fp = |c: &TrainingConfig| {
            candle_gen_perceptual::perceptual_footprint(
                c,
                &krea_decoder(),
                candle_gen_perceptual::AuxGeometry::image(1024, 1),
            )
        };
        let small = fp(&on);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = fp(&on);
        assert!(small > 0 && large > small + (1u64 << 30), "{small} {large}");
        let base = 24u64 << 30;
        for ckpt in [false, true] {
            on.gradient_checkpointing = ckpt;
            assert!(check_perceptual_memory(&TrainingConfig::default(), 1, base, base).is_ok());
            assert!(check_perceptual_memory(&on, 1, base, base + large / 2).is_err());
            assert!(check_perceptual_memory(&on, 1, base, base + large + (1 << 30)).is_ok());
        }
    }

    /// Review fix: the guard prices one cached depth reference per (item, bucket) entry, so a budget
    /// that fits one item's references refuses a dataset whose references do not fit. Mutation:
    /// size the entries as `edges.len()` (buckets only) ⇒ the many-item case passes ⇒ red.
    #[test]
    fn memory_guard_scales_references_with_items() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule();
        let at = |items: usize| {
            candle_gen_perceptual::perceptual_footprint(
                &on,
                &krea_decoder(),
                candle_gen_perceptual::AuxGeometry::image(
                    bucket_edges(&on).iter().copied().max().unwrap_or(0),
                    items * bucket_edges(&on).len(),
                ),
            )
        };
        let (one, many) = (at(1), at(10_000));
        assert!(many > one, "{one} {many}");
        let base = 24u64 << 30;
        let budget = base + one + (many - one) / 2;
        assert!(check_perceptual_memory(&on, 1, base, budget).is_ok());
        assert!(check_perceptual_memory(&on, 10_000, base, budget).is_err());
    }

    /// AC (e): the descriptor declares depth anchoring; a missing TAEW2.1 checkpoint is a named
    /// error.
    #[test]
    fn descriptor_declares_depth_and_missing_decoder_is_named() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
        let tmp = tempfile::tempdir().unwrap();
        let mut c = TrainingConfig::default();
        c.depth_anchoring.schedule = schedule();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taehv"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c, &Device::Cpu)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("TAEW2.1"), "{err}");
    }
}
