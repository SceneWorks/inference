//! The candle **Z-Image LoRA/LoKr trainer** (sc-5166) — the candle twin of `mlx-gen-z-image`'s
//! trainer, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"`. It retires the worker's Python torch `ZImageLoraTrainer` — the **base
//! class** of the whole LoRA-trainer hierarchy (epic 5164) — and reuses the shared
//! [`candle_gen::train`] harness the SDXL story established.
//!
//! Since sc-7787 the cache → loop → save scaffolding lives in the shared single-model flow-match
//! driver ([`candle_gen::train::flow_match`]); this module supplies the Z-Image-specific hooks via
//! [`FlowMatchTrainer`] — caching, DiT construction, and the one parity-critical piece that genuinely
//! differs per family: [`compute_loss_grads`].
//!
//! ## The Z-Image realities that shape the hooks (flow-match, not DDPM ε-prediction)
//!
//!  1. **Cache** — for each captioned image: decode/crop/resize to a VAE-input tensor
//!     ([`decode_square`] + [`square_image_tensor`]), encode the **deterministic latent mean** (the stock z_image
//!     [`Encoder`](VaeEncoder) → `(mean − shift)·scale`; the `reg` sampling is skipped so caching is
//!     reproducible), and encode the caption through the Qwen3 text encoder with the *exact* gen-core
//!     [`TokenizerConfig`](candle_gen::gen_core::tokenizer::TokenizerConfig) inference uses → `(L, 2560)`. The VAE + text encoder are dropped after
//!     caching.
//!  2. **Loop** (driver-owned) — sample a flow-match `σ ∈ [1e-3, 1−1e-3]`
//!     ([`sample_unit_timestep`](candle_gen::train::flow_match::sample_unit_timestep)), form
//!     `x_t = (1−σ)·x0 + σ·noise`, predict the velocity through the vendored trainable DiT, regress it
//!     toward `noise − x0`, and step the adapter factors.
//!  3. **Save** — a PEFT `.safetensors` (`save_lora_peft` with the DiT's **bare** key prefix /
//!     `save_lokr`), the exact on-disk format [`crate::adapters::merge_adapters`] reads back.
//!
//! **Velocity sign.** The Z-Image inference pipeline negates the DiT's raw output before the
//! flow-match Euler step (`noise_pred.neg()` — the Z-Image sign convention). The vendored
//! [`ZImageTransformer2DModel`] returns the **raw** velocity, so the trainer negates it and regresses
//! the negated velocity toward `noise − x0` — exactly optimizing the adapter against the function
//! inference will run (the MLX twin folds the negation into its `forward`; same target either way).
//! The timestep fed to the DiT is `1 − σ` (the `current_timestep_normalized` convention) — distinct
//! from Lens (`t`), Krea (`σ`), and Wan (`t·1000`), which is why [`compute_loss_grads`] stays local.
//!
//! **The eager-`Var` simplification** (inherited from the SDXL harness): the adapter factors are
//! storage-sharing `Var`s installed once ([`build_lora_targets`]/[`build_lokr_targets`]); each forward
//! re-reads the current factor storage and `loss.backward()` attributes grads straight to the `Var`s,
//! so the per-step body is forward → backward → clip → step, with no re-install.
//!
//! **Gradient checkpointing** (`config.gradient_checkpointing`, sc-5246 — the candle twin of the MLX
//! trainer's sc-4874 split) routes the backward through [`checkpointed_backward_with_input_grad`]
//! over the DiT's main `layers` stack ([`ZImageTransformer2DModel::main_layer_segments`]), recomputing
//! each layer instead of retaining its activations — numerically the dense grads (the
//! `dense_and_checkpoint_grads_match` gate asserts this), at the cost of one extra forward over the
//! main stack. The pre-main refiner/embedder forward is run *retained*, so those adapters train via
//! ordinary autograd and their grads are stitched in through the recovered `unified`-boundary
//! cotangent (see [`compute_loss_grads`]). On CUDA an OOM is a catchable error (not the uncatchable
//! Mac-unified-memory SIGKILL that forced the MLX work), so this is the lever for higher training
//! resolutions / smaller cards rather than a correctness requirement.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, IndexOp, Tensor, Var};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, Image, LoadSpec, Modality, WeightsSource};
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor};
use candle_gen::train::flow_match::{
    self, combine_terms, prepared_subject_mask_weight, run_flow_match_training, step_terms,
    validate_flow_match_request, weighted_velocity_loss, AuxStep, FlowMatchTrainer, SamplePlan,
    StepLosses, StepSample,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward_with_input_grad;
use candle_gen::train::perceptual::{Parameterization, PerceptualPath};
use candle_gen::train::tae::TinyDecoderSpec;
use candle_gen::{CandleError, Result};
use candle_gen_perceptual::{AuxGeometry, AuxLossContext, DecoderSpec};

use candle_transformers::models::z_image::preprocess::prepare_inputs;
use candle_transformers::models::z_image::sampling::postprocess_image;
use candle_transformers::models::z_image::scheduler::{
    calculate_shift, FlowMatchEulerDiscreteScheduler, SchedulerConfig, BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT,
};
use candle_transformers::models::z_image::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use candle_transformers::models::z_image::transformer::Config as DitConfig;
use candle_transformers::models::z_image::vae::{AutoEncoderKL, Encoder as VaeEncoder, VaeConfig};
use rand::{rngs::StdRng, SeedableRng};

use crate::dit::{ZImageTransformer2DModel, Z_IMAGE_ATTN_TARGETS};
use crate::pipeline::{LATENT_CHANNELS, PATCH_SIZE, SPATIAL_SCALE};
use crate::MODEL_ID;

/// Cap on preview prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-8650) — the
/// candle twin of the MLX sc-5637 cap. Bounds per-cadence preview cost regardless of how many
/// `sample_prompts` the request carries.
const SAMPLE_PROMPT_CAP: usize = 4;

/// Error-message prefix shared by [`validate_flow_match_request`] and the driver's `no usable dataset
/// items` guard.
const LABEL: &str = "z_image trainer";

/// `(x_t, target, timestep)` for one sample at flow-match `σ`: delegates the latent mix
/// (`x_t = (1−σ)·x0 + σ·noise`, `target = noise − x0`) to the shared
/// [`flow_match::build_batch`](candle_gen::train::flow_match::build_batch) and appends Z-Image's
/// timestep convention `1 − σ` (the `current_timestep_normalized` the DiT's `t_embedder` consumes).
fn build_batch(x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<(Tensor, Tensor, f32)> {
    let (x_t, target) = flow_match::build_batch(x0, noise, sigma as f64)?;
    Ok((x_t, target, 1.0 - sigma))
}

/// The step's loss on the (already-negated) velocity `v` (`(1, 16, h, w)`): the flow-match
/// regression toward `target` and — epic 2123 E8, when `aux` plans one — the weighted perceptual
/// term on the model's x0 estimate `x_t − σ·v` (Z-Image's latent is already the decoder's NCHW
/// model-space layout; TAEF1 decodes the normalized latent directly). On an aux-only step the
/// diffusion term is not computed. `aux = None` is exactly the pre-epic-2123 loss.
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

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at
/// `sigma`, predict the velocity through the (raw-output) DiT, **negate** it (the Z-Image inference
/// sign convention), regress it toward `noise − x0`, and return the step's [`StepLosses`] + grads
/// keyed by the adapter `Var`s. A free function (not a method) so the parity test can drive it
/// against a tiny DiT.
///
/// `x0`/`noise` are the cached `[1, 16, h, w]` clean latent + the per-step noise (f32); `cap` is the
/// cached `(L, cap_feat_dim)` caption embedding. `prepare_inputs` adds the DiT's singleton frame axis
/// and pads the caption to `SEQ_MULTI_OF` (the same call inference makes), so train and infer feed the
/// DiT the identical tensor surface. `aux` (epic 2123 E8) is the step's planned perceptual step
/// ([`step_loss`]); `sigma` is then the plan's noise level.
///
/// `use_checkpoint` selects the gradient-checkpointed backward over the dense `loss.backward()`. The
/// Z-Image split (mirroring the MLX trainer): the **pre-main** forward
/// ([`forward_pre_main`](ZImageTransformer2DModel::forward_pre_main) — embed + refiners) is run
/// retained, so the refiner/embedder adapters train via ordinary autograd; only the main `layers`
/// stack (the activation-memory bulk) is checkpointed via
/// [`main_layer_segments`](ZImageTransformer2DModel::main_layer_segments) +
/// [`checkpointed_backward_with_input_grad`], recomputing each layer in the backward. The returned
/// boundary cotangent `dL/d unified` is then stitched back through the retained pre-main forward to
/// fold in those adapters' grads. Both paths yield the same grads — the `tests` parity gate
/// (`dense_and_checkpoint_grads_match`) pins this, exactly as the SDXL trainer does.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &ZImageTransformer2DModel,
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
    let prepared = prepare_inputs(&x_t, std::slice::from_ref(cap), device)?;
    let cap_feats = prepared.cap_feats.to_dtype(compute_dtype)?;
    let t = Tensor::from_vec(vec![timestep], (1,), device)?;

    if use_checkpoint {
        // Retain the pre-main forward (refiner/embedder adapters train via ordinary autograd); only
        // the main `layers` stack is checkpointed (each layer recomputed in the backward).
        let (unified, ctx) =
            dit.forward_pre_main(&prepared.latents, &t, &cap_feats, &prepared.cap_mask)?;
        // The final segment's loss breakdown is read back through `parts` (declared before the
        // segments, which borrow it).
        let parts = std::cell::Cell::new(None);
        let mut segs = dit.main_layer_segments(&ctx);
        // Final segment: the post-main head + the negated-velocity flow-match regression (+ the
        // planned perceptual term) -> [loss]. (`squeeze(2)` drops the singleton frame axis:
        // (1, 16, 1, h, w) -> (1, 16, h, w).)
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        let (x_t_ref, parts_ref) = (&x_t_f32, &parts);
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?.squeeze(2)?.neg()?;
            let (loss, losses) =
                step_loss(&v, &target_owned, x_t_ref, sigma, mask_weight, mae, aux)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            parts_ref.set(Some(losses));
            Ok(vec![loss])
        }));
        // Seed the checkpointed chain with the detached `unified` boundary, recovering its cotangent.
        let unified_d = unified.detach();
        let (_, mut grads, input_cot) = checkpointed_backward_with_input_grad(
            &segs,
            std::slice::from_ref(&unified_d),
            lora_vars,
        )?;
        drop(segs); // release the closures' borrows of `dit`/`ctx` before the stitch backward
        let losses = parts.take().ok_or_else(|| {
            CandleError::Msg(format!("{LABEL}: the checkpointed loss segment never ran"))
        })?;

        // Continue the chain rule into the retained pre-main: `s = ⟨unified, dL/d unified⟩` then
        // `s.backward()` delivers the refiner/embedder adapter grads (cotangent is a detached
        // constant). The main-layer grads are already accumulated in `grads`; the two `Var` sets are
        // disjoint (refiner blocks ≠ main blocks), so this is a union, not a sum — but accumulate
        // defensively in case a future layout shares a factor across the boundary.
        let cot = input_cot[0].to_dtype(DType::F32)?;
        let surrogate = (unified.to_dtype(DType::F32)? * cot)?.sum_all()?;
        let pre_grads = surrogate.backward()?;
        for v in lora_vars {
            if let Some(g) = pre_grads.get(v.as_tensor()) {
                let merged = match grads.get(v.as_tensor()) {
                    Some(prev) => (prev + g)?,
                    None => g.clone(),
                };
                grads.insert(v.as_tensor(), merged);
            }
        }
        Ok((losses, grads))
    } else {
        // Dense backward: one monolithic `loss.backward()` retains every layer's activations.
        // Raw DiT velocity, then negated to match the inference pipeline's `noise_pred.neg()`.
        let v = dit
            .forward(&prepared.latents, &t, &cap_feats, &prepared.cap_mask)?
            .squeeze(2)? // drop the singleton frame axis: (1, 16, 1, h, w) -> (1, 16, h, w)
            .neg()?;
        let (loss, losses) = step_loss(&v, &target, &x_t_f32, sigma, mask_weight, mae, aux)?;
        let grads = loss.backward()?;
        Ok((losses, grads))
    }
}

/// Encode the **deterministic latent mean** of a VAE-input image `[1, 3, edge, edge]` (range
/// `[-1, 1]`) to a clean latent `[1, 16, edge/8, edge/8]`: run the encoder, take the distribution
/// **mean** (the first channel half, skipping `DiagonalGaussian`'s sampling), then `(mean − shift)·
/// scale` — the same affine `AutoEncoderKL::encode` applies, minus the stochastic draw.
fn vae_encode_mean(encoder: &VaeEncoder, img: &Tensor, shift: f64, scale: f64) -> Result<Tensor> {
    let moments = img.to_dtype(DType::F32)?.apply(encoder)?; // (1, 32, h, w) = [mean; logvar]
    let mean = moments.chunk(2, 1)?[0].clone(); // (1, 16, h, w)
    Ok(((mean - shift)? * scale)?)
}

/// Tokenize `caption` with the Qwen chat template + encode it through the Qwen3 text encoder to the
/// cached conditioning `(L, cap_feat_dim)` at f32 — the exact [`TokenizerConfig`](candle_gen::gen_core::tokenizer::TokenizerConfig) / encoder the
/// inference [`crate::pipeline`] uses (parity), minus the device-dtype cast (caching keeps f32).
fn encode_caption(
    tok: &TextTokenizer,
    te: &ZImageTextEncoder,
    caption: &str,
    device: &Device,
) -> Result<Tensor> {
    let out = tok
        .tokenize(caption)
        .map_err(|e| CandleError::Msg(format!("z_image trainer: tokenize: {e}")))?;
    if out.ids.is_empty() {
        return Err(CandleError::Msg(
            "z_image trainer: empty caption tokenization".into(),
        ));
    }
    let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), device)?;
    let enc = te.forward(&input_ids)?; // (1, L, 2560)
    Ok(enc.squeeze(0)?.to_dtype(DType::F32)?) // (L, 2560)
}

/// The Z-Image preview-sample render state (sc-8650) — everything [`ZImageTrainer::render_sample`]
/// needs to run the family's CFG-free, guidance-distilled denoise on the **in-progress** trainable
/// DiT, built once in [`ZImageTrainer::cache`] while the text encoder is still resident:
///
///  * `caps` — the per-prompt pre-encoded Qwen3 conditioning, each `(L, 2560)` at f32 (exactly what
///    [`encode_caption`] returns and `prepare_inputs` consumes), 1:1 with [`SamplePlan::prompts`].
///  * `vae` — the resident `AutoEncoderKL` **decoder** (`Arc` as inference holds it); the cache pass
///    loads only the VAE `Encoder`, so the full VAE is loaded here for the preview decode path.
///  * `edge` — the square preview edge: the largest training-bucket edge
///    ([`bucket_edges`], just `bucket_resolution(cfg.resolution)` with buckets off — the edge the
///    cached latents use) the seeded preview noise + the `mu` shift are shaped at.
pub struct ZImageSampleState {
    caps: Vec<Tensor>,
    vae: Arc<AutoEncoderKL>,
    edge: u32,
}

/// Seeded initial Gaussian latent noise `[1, 16, edge/8, edge/8]` (f32) for a preview render — the
/// training-side twin of [`crate::pipeline::Pipeline::render`]'s noise init (sc-8650): a
/// fixed-algorithm CPU `StdRng` (sc-3673 launch-portable discipline) seeded by `seed`, built on CPU
/// then moved to `device`, exactly as inference draws its prior.
fn sample_noise_latent(edge: u32, seed: u64, device: &Device) -> Result<Tensor> {
    let lat = (edge / SPATIAL_SCALE) as usize;
    let n = LATENT_CHANNELS * lat * lat;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat, lat), &Device::Cpu)?.to_device(device)?)
}

/// VAE-decode a final preview latent `[1, 16, 1, h, w]` → RGB8 [`Image`] — the training-side twin of
/// [`crate::pipeline::Pipeline::decode`] (sc-8650): drop the singleton frame axis, `AutoEncoderKL::
/// decode` applies its own `/scaling + shift` un-scale internally and returns `[1, 3, H, W]` in
/// `[-1, 1]`, and `postprocess_image` maps that to `[0, 255]` u8.
fn decode_preview(vae: &AutoEncoderKL, latents: &Tensor) -> Result<Image> {
    // Drop the singleton frame axis: (1, 16, 1, h, w) -> (1, 16, h, w).
    let latents = latents.squeeze(2)?;
    let decoded =
        candle_gen::bounded_kl_decode(&crate::common::VAE_DECODER, &latents, |l| vae.decode(l))?
            .to_dtype(DType::F32)?; // (1, 3, H, W) in [-1, 1]
    let img = postprocess_image(&decoded)? // u8 (1, 3, H, W)
        .i(0)?
        .to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "z_image trainer: preview decode expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Build the vendored, trainable DiT from the snapshot `transformer/` safetensors at `dtype`.
fn build_trainable_dit(
    root: &Path,
    device: &Device,
    dtype: DType,
) -> Result<ZImageTransformer2DModel> {
    let vb = flow_match::component_vb(root, "transformer", device, dtype, LABEL)?;
    // The vendored DiT always runs the materialized math attention (the flash/SDPA paths have no
    // backward); the stock `use_accelerated_attn` knob is irrelevant to it. `z_image_turbo` config.
    Ok(ZImageTransformer2DModel::new(
        &DitConfig::z_image_turbo(),
        vb,
    )?)
}

/// Z-Image's latent family for the shared aux-loss builder (epic 2123 E8): the FLUX.1 16-channel
/// VAE latent (`(mean − shift)·scale`), decoded by TAEF1 (its diffusers config has
/// `scaling_factor 1.0 / shift_factor 0.0`, i.e. it decodes that normalized latent directly).
fn aux_loss_context(device: &Device) -> AuxLossContext<'_> {
    AuxLossContext {
        label: LABEL,
        decoder: taef1_decoder(),
        latent_lpips: Some(gen_core::train::LatentLpipsFamily::Flux),
        device,
    }
}

fn taef1_decoder() -> DecoderSpec {
    DecoderSpec::Tiny {
        name: "TAEF1",
        config: TinyDecoderSpec::taef1(),
    }
}

/// The epic-2123 perceptual path for `cfg`: `None` when no aux loss is enabled (nothing loads).
fn load_perceptual_path(cfg: &TrainingConfig, device: &Device) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(cfg, &aux_loss_context(device))
}

/// Extra training memory (bytes) the enabled perceptual losses add for `items` dataset items: the
/// TAEF1 decoder + each loss at the largest bucket edge, plus one cached reference per (item,
/// bucket) entry (epic 2123 E7). `0` when no aux loss is on.
fn perceptual_footprint_bytes(cfg: &TrainingConfig, items: usize) -> u64 {
    let edges = bucket_edges(cfg);
    let edge = edges.iter().copied().max().unwrap_or(0);
    candle_gen_perceptual::perceptual_footprint(
        cfg,
        &taef1_decoder(),
        AuxGeometry::image(edge, items * edges.len()),
    )
}

/// Epic 2123 E7 preflight: when a perceptual loss is on, the trained DiT's resident weights (its
/// `transformer/` safetensors — the lower bound; no fitted activation model exists for this
/// trainer, so the same base applies checkpointed or dense) plus the aux footprint must fit
/// `budget_bytes`. Depth off ⇒ no check (unchanged behavior).
fn aux_memory_preflight(
    root: &Path,
    cfg: &TrainingConfig,
    items: usize,
    budget_bytes: u64,
) -> Result<()> {
    let aux = perceptual_footprint_bytes(cfg, items);
    if aux == 0 {
        return Ok(());
    }
    let base = flow_match::component_bytes(root, "transformer", LABEL)?;
    flow_match::check_aux_memory(LABEL, base, aux, budget_bytes)
}

/// Identity + capabilities of the candle Z-Image trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "z-image",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // sc-10894 lockstep catch-up: gen-core gained `TrainerDescriptor.supports_control` (mirrors
        // mlx-gen-z-image's `false` — the base LoRA/LoKr trainer trains no control branch).
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
        // sc-24830 (epic 2123): depth anchoring through the shared perceptual path (TAEF1 +
        // Depth-Anything-V2), dense and gradient-checkpointed.
        // sc-24833 (epic 2123): the VAE anchor (same family decoder → FLUX.2 encoder taps) and
        // E-LatentLPIPS (this latent family's published weights) through the shared aux-loss
        // builder this trainer already drives.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
            identity_loss: true,
            face_landmark_loss: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring
            // (decoded-x0 pixel losses through this trainer's x0 decoder).
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            vae_anchor_loss: true,
            latent_lpips_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// A loaded candle Z-Image trainer. Loading is **lazy** (no file I/O — mirrors the candle SDXL
/// trainer): the heavy VAE / text-encoder / DiT are built inside [`train`](Trainer::train) at the
/// request's compute dtype.
pub struct ZImageTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Z-Image trainer from a [`LoadSpec`] whose `weights` is the Z-Image
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "z_image_turbo trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(ZImageTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// `register_trainer!` defines the explicit trainer constant and bridges the crate's rich `Result`
// into `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! {
    pub(crate) const REGISTRATION = trainer_descriptor => load_trainer
}

impl Trainer for ZImageTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
        // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA
        // adapter the caller did not ask for (F-006/F-055).
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared control-training floor (F-006/F-055): no control branch, so a `control_type`
        // request is a typed reject, never silently trained as a plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
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
        // Every other `validate` floor too (control / full fine-tune / edit / flow-match knobs), so
        // a caller that skips `validate` cannot train a request it would refuse.
        self.validate(req)?;
        run_flow_match_training(self, req, on_progress).map_err(Into::into)
    }
}

impl FlowMatchTrainer for ZImageTrainer {
    type Dit = ZImageTransformer2DModel;
    /// `(x0 latent [1,16,h,w], caption embed (L, 2560), subject-mask loss weight)`, all f32; the
    /// weight (broadcast to the latent shape) is `None` unless subject-masked loss is on (sc-24828).
    type Cached = (Tensor, Tensor, Option<Tensor>);
    type Aux = ();
    /// Preview-sample render state (sc-8650): per-prompt `(L, 2560)` conditioning + the resident VAE
    /// decoder + the training edge. See [`ZImageSampleState`].
    type SampleState = ZImageSampleState;
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        &Z_IMAGE_ATTN_TARGETS
    }

    /// Epic 2123 E7: with a perceptual loss on, refuse a run whose DiT weights + TAEF1/DA2
    /// footprint exceed the device budget (checkpointed or dense alike).
    fn preflight(&self, req: &TrainingRequest) -> Result<()> {
        aux_memory_preflight(
            &self.root,
            &req.config,
            req.items.len(),
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

    /// The cached `[1, 16, h, w]` latent is already TAEF1's NCHW model-space input.
    fn reference_latent(&self, cached: &Self::Cached, _aux: &()) -> Result<Tensor> {
        Ok(cached.0.clone())
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<Self::Cached>, (), SamplePlan<ZImageSampleState>)> {
        // sc-2127: one training edge per resolution bucket (just `[resolution]` when buckets are off);
        // previews render at the largest (epic 2123 E7).
        let edges = bucket_edges(&req.config);
        let edge = edges.iter().copied().max().unwrap_or(0);
        let vae_cfg = VaeConfig::z_image();
        let vae_encoder = VaeEncoder::new(
            &vae_cfg,
            flow_match::component_vb(&self.root, "vae", device, DType::F32, LABEL)?.pp("encoder"),
        )?;
        // F-133 (sc-11190): the trainer's caption encode is parity-critical with inference, so the
        // tokenizer policy lives in one home (`common::tokenizer_config`) — a policy change (sc-8646
        // class) now can't land in inference and silently miss the trainer.
        let tokenizer = crate::common::build_tokenizer_from_base(&self.root, "z_image trainer")?;
        let text_encoder = ZImageTextEncoder::new(
            &TextEncoderConfig::z_image(),
            flow_match::component_vb(&self.root, "text_encoder", device, DType::F32, LABEL)?,
        )?;

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
            let cap = encode_caption(&tokenizer, &text_encoder, &item.caption, device)?;
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
                let x0 = vae_encode_mean(
                    &vae_encoder,
                    &img,
                    vae_cfg.shift_factor,
                    vae_cfg.scaling_factor,
                )?;
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

        // Preview samples (sc-8650): while the text encoder is still resident, pre-encode up to
        // `SAMPLE_PROMPT_CAP` prompts with the SAME `encode_caption` the cache loop uses, and load a
        // resident VAE **decoder** (the cache pass only loaded the `Encoder`). The driver renders these
        // from the in-progress adapter at the `sample_every` cadence. Disabled (no state) when the
        // request opts out, so a non-sampling run loads no decoder and trains exactly as before.
        let cfg = &req.config;
        let sample_plan = if cfg.sample_every == 0 || cfg.sample_prompts.is_empty() {
            SamplePlan::disabled()
        } else {
            let mut prompts = Vec::new();
            let mut caps = Vec::new();
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                caps.push(encode_caption(&tokenizer, &text_encoder, prompt, device)?);
                prompts.push(prompt.clone());
            }
            // The full `AutoEncoderKL` (encoder + decoder); inference holds it `Arc`-shared.
            let vae = AutoEncoderKL::new(
                &vae_cfg,
                flow_match::component_vb(&self.root, "vae", device, DType::F32, LABEL)?,
            )?;
            SamplePlan {
                prompts,
                state: Some(ZImageSampleState {
                    caps,
                    vae: Arc::new(vae),
                    edge,
                }),
            }
        };

        // Encoders are dead weight once cached — drop them before the DiT (working set) loads.
        drop(text_encoder);
        drop(vae_encoder);
        Ok((cache, (), sample_plan))
    }

    fn build_dit(
        &self,
        req: &TrainingRequest,
        device: &Device,
    ) -> Result<ZImageTransformer2DModel> {
        build_trainable_dit(
            &self.root,
            device,
            flow_match::parse_compute_dtype(&req.config.train_dtype),
        )
    }

    fn micro_step(
        &self,
        dit: &ZImageTransformer2DModel,
        vars: &[Var],
        cached: &Self::Cached,
        _aux: &(),
        cfg: &TrainingConfig,
        step: u32,
        sample: StepSample<'_>,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let (x0, cap, mask_weight) = cached;
        let mut sigma = flow_match::sample_unit_timestep(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            flow_match::timestep_seed(cfg.seed, step),
        );
        let noise =
            flow_match::sample_noise(x0.dims(), flow_match::noise_seed(cfg.seed, step), device)?;
        // Epic 2123 E8: plan the step on the sampled σ (Z-Image's σ is the `[0, 1]` noise level,
        // 1 = pure noise); an aux-only step trains at σ remapped into the loss window.
        let aux = sample.plan(sigma)?;
        if let Some(a) = &aux {
            sigma = a.noise_level();
        }
        let (losses, grads) = compute_loss_grads(
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

    /// Render preview prompt `index` from the **in-progress** trainable DiT (sc-8650). Mirrors the
    /// Z-Image inference denoise ([`crate::pipeline::Pipeline::render`]) on the trainable
    /// [`ZImageTransformer2DModel`] (adapters live as eager `Var`s, so its plain `forward` runs the
    /// partially-trained LoRA): seed the latent prior, drive the distilled flow-match Euler schedule
    /// with `set_timesteps(steps, Some(mu))` (the speckle-free arm), and step with the **Z-Image sign
    /// convention** — the DiT is fed the `1 − σ` timestep ([`TimestepConvention::OneMinusSigma`]) and
    /// its predicted velocity is **negated** before the Euler step, exactly as inference does. Z-Image-
    /// Turbo is guidance-distilled (CFG-free), so `cfg.sample_guidance_scale` is ignored. Best-effort:
    /// any error here is logged + skipped by the driver, never aborting the run.
    fn render_sample(
        &self,
        dit: &ZImageTransformer2DModel,
        state: &ZImageSampleState,
        index: usize,
        cfg: &TrainingConfig,
        seed: u64,
    ) -> Result<Image> {
        let device = self.device();
        let cap = state.caps.get(index).ok_or_else(|| {
            CandleError::Msg(format!(
                "z_image trainer: preview prompt index {index} out of range"
            ))
        })?;
        let steps = (cfg.sample_steps as usize).max(1);
        let lat = (state.edge / SPATIAL_SCALE) as usize;
        // The DiT is built at the bf16 compute dtype (`build_dit`); the inference forward does NOT cast
        // its inputs (unlike Krea's), so feed bf16 latents + conditioning — exactly as `compute_loss_grads`
        // does (`x_t`/`cap_feats` → `compute_dtype`) — or the first matmul hits an F32×BF16 mismatch.
        let compute_dtype = flow_match::parse_compute_dtype(&cfg.train_dtype);

        // Seeded launch-portable prior at the training resolution (square `edge`). `prepare_inputs`
        // pads `cap` to SEQ_MULTI_OF (+ mask) and adds the singleton frame axis to the latents →
        // (1, 16, 1, lat, lat) — the exact tensor surface train + infer feed the DiT.
        let noise = sample_noise_latent(state.edge, seed, device)?;
        let prepared = prepare_inputs(&noise, std::slice::from_ref(cap), device)?;
        let cap_feats = prepared.cap_feats.to_dtype(compute_dtype)?;
        let cap_mask = prepared.cap_mask;

        // Distilled flow-match Euler schedule — pass `Some(mu)` (the resolution-dependent shift) so the
        // σ table stays consistent with the `1 − σ` conditioning (the `None` arm desyncs them and
        // speckles; see `pipeline::Pipeline::render`). `mu` is derived from the post-patchify seq len.
        let image_seq_len = ((lat as u32 / PATCH_SIZE) * (lat as u32 / PATCH_SIZE)) as usize;
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        scheduler.set_timesteps(steps, Some(mu));
        // `TrainingConfig` carries no sampler/scheduler knob — preview uses the native distilled σ table
        // verbatim (`None` ⇒ the scheduler's own schedule) and the default `euler` sampler (the N1 no-op
        // = the legacy Euler step), exactly the Z-Image inference default.
        let native: Vec<f32> = scheduler.sigmas.iter().map(|&s| s as f32).collect();
        let sigmas = candle_gen::resolve_flow_schedule(None, 0.0, steps, &native);

        // A preview never honors mid-denoise cancel; a fresh never-cancelled token suffices.
        let nocancel = CancelFlag::new();
        let latents = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::OneMinusSigma,
            &sigmas,
            prepared.latents.to_dtype(compute_dtype)?,
            seed,
            &nocancel,
            &mut |_| {},
            None,
            |latents, t| -> Result<Tensor> {
                // `t` is the `1 − σ` conditioning the DiT embeds; the raw velocity is NEGATED to match
                // inference's `noise_pred.neg()` (the Z-Image sign convention).
                let t_tensor = Tensor::from_vec(vec![t], (1,), device)?;
                let velocity = dit
                    .forward(latents, &t_tensor, &cap_feats, &cap_mask)?
                    .neg()?;
                Ok(velocity)
            },
        )?;

        // The denoise ran in `compute_dtype`; the resident VAE is F32 → cast back before decode.
        decode_preview(&state.vae, &latents.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::z_image::transformer::Config;

    /// The smallest valid Z-Image DiT (head_dim locked to 128 by `axes_dims`), 1 main layer + 1
    /// refiner each, tiny caption dim — exercises the real flow-match forward + backward on CPU.
    fn tiny_dit(vb: VarBuilder) -> (ZImageTransformer2DModel, Config) {
        let mut cfg = Config::z_image_turbo();
        cfg.dim = 128;
        cfg.n_heads = 1;
        cfg.n_kv_heads = 1;
        cfg.n_layers = 1;
        cfg.n_refiner_layers = 1;
        cfg.cap_feat_dim = 64;
        let model = ZImageTransformer2DModel::new(&cfg, vb).unwrap();
        (model, cfg)
    }

    /// The pre-epic-2123 call shape (no perceptual step) for the legacy tests.
    #[allow(clippy::too_many_arguments)]
    fn compute_loss_grads(
        dit: &ZImageTransformer2DModel,
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
        super::compute_loss_grads(
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
        )
        .map(|(l, g)| (l.total, g))
    }

    /// `build_batch`: `x_t = (1−σ)x0 + σ·noise`, `target = noise − x0`, `timestep = 1−σ` (the Z-Image
    /// convention layered over the shared `flow_match::build_batch`).
    #[test]
    fn build_batch_math() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target, timestep) = build_batch(&x0, &noise, 0.25).unwrap();
        // x_t = 0.75·[2,4] + 0.25·[1,0] = [1.75, 3.0]; target = [1-2, 0-4] = [-1, -4]; t = 0.75.
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        assert!((timestep - 0.75).abs() < 1e-6);
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (backprop reaches
    /// the LoRA seam through the negated-velocity loss).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }

        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();

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
        // Sanity: the LoRA set covers every attention projection in the 1 refiner ×2 + 1 layer.
        let expected_targets = 4 * (cfg.n_refiner_layers * 2 + cfg.n_layers);
        assert_eq!(
            set.vars.len(),
            expected_targets * 2,
            "two factors per target"
        );
    }

    /// The correctness gate for the `gradient_checkpointing` lever: the checkpointed backward must
    /// reproduce the dense `loss.backward()` grads (mod float reassociation) over the tiny DiT with
    /// nonzero LoRA factors. The tiny config adapts BOTH refiner and main-layer blocks, so this
    /// comparison spans the two distinct checkpoint paths — the **retained** pre-main refiner/embedder
    /// adapters (grads stitched in through the recovered `unified`-boundary cotangent) AND the
    /// **checkpointed** main `layers` (grads from the segmented VJP). A break in either path surfaces
    /// here as a missing or mismatched grad. Mirrors the SDXL trainer's `dense_and_checkpoint_grads_match`.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off zero so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        // Every projection in both refiner blocks AND the main layer is adapted, so the per-var
        // comparison below necessarily spans the retained (refiner) and checkpointed (main) paths.
        let expected_targets = 4 * (cfg.n_refiner_layers * 2 + cfg.n_layers);
        assert_eq!(
            set.vars.len(),
            expected_targets * 2,
            "two factors per target"
        );

        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();

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
    }

    /// sc-24828: subject-masked loss on both backward paths. An all-ones map is the unweighted loss;
    /// an all-zero map zeroes the loss AND every adapter gradient (dense and checkpointed — a path
    /// that dropped the weight would train on the background); a half map matches across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let shape = [1usize, cfg.in_channels, 4, 4];
        let x0 = Tensor::randn(0f32, 1f32, &shape, &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, &shape, &dev).unwrap();
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
    }

    /// One optimizer step over the tiny DiT lowers (or holds) the loss on the same fixed batch — the
    /// step actually descends the flow-match objective, end to end through the harness.
    #[test]
    fn one_optimizer_step_descends() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let (mut dit, cfg) = tiny_dit(vb);
        let suffixes: Vec<String> = Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();
        let cap = Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 4, 4), &dev).unwrap();

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
        // A few steps on the same batch should reduce the loss (over-fit a single sample).
        let mut grads = grads;
        for _ in 0..5 {
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
        assert!(
            loss1 < loss0,
            "5 steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
    }

    /// The trainer resolves through the explicit family registry as the candle
    /// Z-Image trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer("z_image_turbo", &spec)
            .expect("candle z-image trainer is registered");
        assert_eq!(t.descriptor().id, "z_image_turbo");
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
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer("z_image_turbo", &spec)
            .unwrap();

        let item = candle_gen::gen_core::train::TrainingItem {
            image_path: "/img.png".into(),
            caption: "x".into(),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: None,
        };
        let base = TrainingRequest {
            items: vec![item.clone()],
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

    /// Epic 2123 depth anchoring (sc-24830) on the candle Z-Image trainer: the tiny DiT + a
    /// random-init tiny TAEF1-shaped decoder + tiny DA2, driven through the real step.
    mod depth_anchoring {
        use super::*;
        use candle_gen::gen_core::train::{AuxLossSchedule, DepthModelSize};
        use candle_gen::train::flow_match::AuxDriver;

        /// Alternate every 2nd window, aux window [0.6, 0.9] so the remap is observable.
        fn schedule() -> AuxLossSchedule {
            AuxLossSchedule {
                weight: 0.5,
                t_min: 0.6,
                t_max: 0.9,
                every_n: 2,
            }
        }

        struct Fixture {
            dit: ZImageTransformer2DModel,
            vars: Vec<Var>,
            x0: Tensor,
            cap: Tensor,
            noise: Tensor,
        }

        fn fixture() -> Fixture {
            let dev = Device::Cpu;
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let (mut dit, cfg) = tiny_dit(vb);
            let suffixes: Vec<String> =
                Z_IMAGE_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
            let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
            for v in &set.vars {
                v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                    .unwrap();
            }
            let shape = (1, cfg.in_channels, 4, 4);
            Fixture {
                dit,
                vars: set.vars,
                x0: Tensor::randn(0f32, 1f32, shape, &dev).unwrap(),
                cap: Tensor::randn(0f32, 1f32, (3usize, cfg.cap_feat_dim), &dev).unwrap(),
                noise: Tensor::randn(0f32, 1f32, shape, &dev).unwrap(),
            }
        }

        /// A one-item, one-bucket aux driver over the fixture latent: step 1 is a diffusion step,
        /// step 2 (the item's 2nd window) is aux-only.
        fn driver(f: &Fixture) -> (AuxDriver, gen_core::BucketSchedule) {
            let path =
                candle_gen_perceptual::testing::tiny_depth_path(16, schedule(), &Device::Cpu)
                    .unwrap();
            let sched = gen_core::BucketSchedule::new(1, &[], 3);
            let x0 = f.x0.clone();
            let d = AuxDriver::prepare(
                path,
                1,
                |_| Ok(x0.clone()),
                &sched,
                1,
                0,
                &Default::default(),
            )
            .unwrap();
            (d, sched)
        }

        fn run(
            f: &Fixture,
            sigma: f32,
            ckpt: bool,
            aux: Option<&AuxStep<'_>>,
        ) -> (StepLosses, GradStore) {
            super::super::compute_loss_grads(
                &f.dit,
                &f.vars,
                &f.x0,
                &f.cap,
                sigma,
                &f.noise,
                false,
                None,
                DType::F32,
                ckpt,
                aux,
            )
            .unwrap()
        }

        fn grad_bits(g: &GradStore, v: &Var) -> Vec<u32> {
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
        }

        /// (a) An aux-only step on both backward paths: no diffusion term, total == aux, and the
        /// LoRA B factors get a nonzero finite gradient. (b) The diffusion step carries no aux
        /// term. Mutations: compute the diffusion term unconditionally in `step_loss` ⇒ (a) red;
        /// pass `None` as `aux` in the checkpointed segment ⇒ ckpt (a) red.
        #[test]
        fn aux_only_step_trains_the_adapter_through_depth_alone() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let s1 = d.sample(1, &sched).plan(0.5).unwrap().unwrap();
            assert!(s1.diffusion() && !s1.has_aux());
            for ckpt in [false, true] {
                let (l, _) = run(&f, 0.5, ckpt, Some(&s1));
                assert!(
                    l.diffusion.is_some() && l.aux.is_none(),
                    "ckpt={ckpt}: {l:?}"
                );
                assert_eq!(l.total, l.diffusion.unwrap());
            }
            let s2 = d.sample(2, &sched).plan(0.5).unwrap().unwrap();
            assert!(!s2.diffusion() && s2.has_aux());
            let t = s2.noise_level();
            assert!((0.6..=0.9).contains(&t), "remapped into the window: {t}");
            for ckpt in [false, true] {
                let (l, g) = run(&f, t, ckpt, Some(&s2));
                assert!(l.diffusion.is_none(), "ckpt={ckpt}: {l:?}");
                let aux = l.aux.expect("aux term");
                assert!(aux.is_finite() && aux > 0.0, "ckpt={ckpt}: {aux}");
                assert_eq!(l.total, aux);
                let mut nonzero_b = false;
                for (i, v) in f.vars.iter().enumerate() {
                    let gv: Vec<f32> = grad_bits(&g, v).into_iter().map(f32::from_bits).collect();
                    assert!(gv.iter().all(|x| x.is_finite()), "ckpt={ckpt} var {i}");
                    nonzero_b |= i % 2 == 1 && gv.iter().any(|x| *x != 0.0);
                }
                assert!(
                    nonzero_b,
                    "ckpt={ckpt}: no LoRA B gradient from the depth term"
                );
            }
        }

        /// The aux term is the depth loss of THE trainer's x0 estimate: recomputed independently from
        /// the model output with the trainer's parameterisation, it matches the step's aux bits.
        /// Mutation: swap the parameterisation in `step_loss` (e.g. `FlowX0MinusNoise`) ⇒ red.
        #[test]
        fn aux_term_is_the_depth_loss_of_the_recovered_x0() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let _ = d.sample(1, &sched);
            let s2 = d.sample(2, &sched).plan(0.5).unwrap().unwrap();
            let sigma = s2.noise_level();
            let (l, _) = run(&f, sigma, false, Some(&s2));
            let (x_t, _, timestep) = build_batch(&f.x0, &f.noise, sigma).unwrap();
            let prepared =
                prepare_inputs(&x_t, std::slice::from_ref(&f.cap), &Device::Cpu).unwrap();
            let t = Tensor::from_vec(vec![timestep], (1,), &Device::Cpu).unwrap();
            let v = f
                .dit
                .forward(
                    &prepared.latents,
                    &t,
                    &prepared.cap_feats,
                    &prepared.cap_mask,
                )
                .unwrap()
                .squeeze(2)
                .unwrap()
                .neg()
                .unwrap();
            let x0 = (&x_t - (v * sigma as f64).unwrap()).unwrap();
            let want = s2
                .aux_loss(&x0)
                .unwrap()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(l.aux.unwrap().to_bits(), want.to_bits());
        }

        /// (c) Depth off is bit-identical to the pre-epic-2123 dense step (reproduced here
        /// verbatim): loss and every gradient. Mutation: route the `None` case through a scaled
        /// combine (e.g. `diffusion * 1.0000001`) ⇒ red.
        #[test]
        fn depth_off_is_bit_identical_to_the_legacy_step() {
            let f = fixture();
            let (l, g) = run(&f, 0.5, false, None);
            assert_eq!((l.diffusion, l.aux), (Some(l.total), None));
            // The legacy body.
            let (x_t, target, timestep) = build_batch(&f.x0, &f.noise, 0.5).unwrap();
            let prepared =
                prepare_inputs(&x_t, std::slice::from_ref(&f.cap), &Device::Cpu).unwrap();
            let t = Tensor::from_vec(vec![timestep], (1,), &Device::Cpu).unwrap();
            let v = f
                .dit
                .forward(
                    &prepared.latents,
                    &t,
                    &prepared.cap_feats,
                    &prepared.cap_mask,
                )
                .unwrap()
                .squeeze(2)
                .unwrap()
                .neg()
                .unwrap();
            let loss = weighted_velocity_loss(&v, &target, None, false).unwrap();
            let legacy = loss.to_scalar::<f32>().unwrap();
            let lg = loss.backward().unwrap();
            assert_eq!(l.total.to_bits(), legacy.to_bits());
            for v in &f.vars {
                assert_eq!(grad_bits(&g, v), grad_bits(&lg, v));
            }
        }

        /// The trainer's `micro_step` trains an aux-only step at the plan's REMAPPED σ (not the
        /// sampled one). Mutation: drop `sigma = a.noise_level()` ⇒ red.
        #[test]
        fn micro_step_trains_at_the_planned_noise_level() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let trainer = ZImageTrainer {
                descriptor: trainer_descriptor(),
                root: PathBuf::from("/nonexistent"),
                device: Device::Cpu,
            };
            let cfg = TrainingConfig {
                train_dtype: "f32".into(),
                ..TrainingConfig::default()
            };
            let cached = (f.x0.clone(), f.cap.clone(), None);
            // Step 1 builds the alternation phase; step 2 is aux-only.
            let _ = d.sample(1, &sched);
            let sample = d.sample(2, &sched);
            let raw = flow_match::sample_unit_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                flow_match::timestep_seed(cfg.seed, 2),
            );
            let planned = sample.plan(raw).unwrap().unwrap();
            assert!(!planned.diffusion());
            let noise = flow_match::sample_noise(
                f.x0.dims(),
                flow_match::noise_seed(cfg.seed, 2),
                &Device::Cpu,
            )
            .unwrap();
            let expect = super::super::compute_loss_grads(
                &f.dit,
                &f.vars,
                &f.x0,
                &f.cap,
                planned.noise_level(),
                &noise,
                false,
                None,
                DType::F32,
                false,
                Some(&planned),
            )
            .unwrap()
            .0
            .total;
            let (got, _) = trainer
                .micro_step(&f.dit, &f.vars, &cached, &(), &cfg, 2, sample, &Device::Cpu)
                .unwrap();
            assert_eq!(got.to_bits(), expect.to_bits());
        }

        fn depth_on() -> TrainingConfig {
            let mut cfg = TrainingConfig::default();
            cfg.depth_anchoring.schedule = schedule();
            cfg
        }

        /// (d) E7: the aux footprint is 0 off, grows with Large DA2, and the preflight refuses at a
        /// budget between base and base + aux — checkpointed or dense (same base: the DiT weights).
        /// Mutation: skip the guard (`if aux == 0` ⇒ `if true`) ⇒ red.
        #[test]
        fn preflight_counts_the_perceptual_models() {
            assert_eq!(perceptual_footprint_bytes(&TrainingConfig::default(), 4), 0);
            let mut cfg = depth_on();
            let small = perceptual_footprint_bytes(&cfg, 4);
            assert!(small > 0);
            cfg.depth_anchoring.model_size = DepthModelSize::Large;
            assert!(perceptual_footprint_bytes(&cfg, 4) > small);
            cfg.depth_anchoring.model_size = DepthModelSize::Small;

            let tmp = tempfile::tempdir().unwrap();
            let tdir = tmp.path().join("transformer");
            std::fs::create_dir_all(&tdir).unwrap();
            std::fs::write(tdir.join("m.safetensors"), vec![0u8; 4096]).unwrap();
            let base = 4096u64;
            // Depth off never consults the budget.
            assert!(aux_memory_preflight(tmp.path(), &TrainingConfig::default(), 4, 1).is_ok());
            for ckpt in [false, true] {
                cfg.gradient_checkpointing = ckpt;
                assert!(aux_memory_preflight(tmp.path(), &cfg, 4, base + small).is_ok());
                let e = aux_memory_preflight(tmp.path(), &cfg, 4, base + small - 1)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("perceptual"), "ckpt={ckpt}: {e}");
                assert!(aux_memory_preflight(tmp.path(), &cfg, 4, base + small / 2).is_err());
            }
        }

        /// (e) The descriptor declares depth anchoring; depth off builds no path; a missing decoder
        /// dir is an error naming TAEF1. Mutation: `depth_anchoring: false` ⇒ red.
        #[test]
        fn descriptor_and_loader_errors() {
            assert!(trainer_descriptor().techniques.depth_anchoring);
            let dev = Device::Cpu;
            assert!(load_perceptual_path(&TrainingConfig::default(), &dev)
                .unwrap()
                .is_none());
            let tmp = tempfile::tempdir().unwrap();
            let mut cfg = depth_on();
            cfg.depth_anchoring.model_dir = Some(tmp.path().join("da2"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAEF1"), "{e}");
            cfg.perceptual_decoder_dir = Some(tmp.path().join("no-taef1"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAEF1"), "{e}");
        }
    }
}
