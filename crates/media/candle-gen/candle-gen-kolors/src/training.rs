//! Candle LoRA/LoKr training for Kolors. Kolors shares the SDXL U-Net adapter surface, but its
//! conditioning is deliberately not the SDXL dual-CLIP path: captions are encoded by the snapshot's
//! ChatGLM3 tokenizer/model, projected from 4096 to 2048, and paired with Kolors' 5632-wide pooled
//! plus size embedding.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{linear, Module};
use candle_gen::diffusion_schedule::{
    KOLORS_BETA_END as BETA_END, KOLORS_BETA_START as BETA_START,
    KOLORS_TRAIN_STEPS as NUM_TRAIN_TIMESTEPS,
};
use candle_gen::gen_core::sampling::AlphaSchedule;
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    self, BucketSchedule, LoadSpec, Modality, NetworkType, Precision, WeightsSource,
};
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor, SquareImage};
use candle_gen::train::flow_match::{
    self, combine_terms, effective_weight_decay, noise_seed, prepared_subject_mask_weight,
    sample_noise, step_sample, step_terms, weighted_velocity_loss, AuxDriver, AuxStep, StepLosses,
    StepSample,
};
use candle_gen::train::lora::{
    adapter_optimizer_step, build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft,
    AdapterKind, LoraSet, SDXL_ATTN_TARGETS, SDXL_PEFT_PREFIX,
};
use candle_gen::train::optim::{
    accumulate_grads, accumulation_divisor, scale_grads, TrainOptimizer,
};
use candle_gen::train::perceptual::{Parameterization, PerceptualPath};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::train::tae::TinyDecoderSpec;
use candle_gen::{CandleError, Result};
use candle_gen_perceptual::{AuxGeometry, AuxLossContext, DecoderSpec};
use candle_gen_sdxl::{sdxl_unet_config, UNet2DConditionModel, VaeMomentsEncoder};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::chatglm3::ChatGlmModel;
use crate::common::build_time_ids;
use crate::config::ChatGlmConfig;
use crate::tokenizer::KolorsTokenizer;
use crate::MODEL_ID;

const LABEL: &str = "kolors trainer";
const VAE_SCALE: f64 = 0.13025;
const ADDITION_TIME_EMBED_DIM: usize = 256;
const PROJECTION_INPUT_DIM: usize = 5632;
const CONTEXT_DIM: usize = 4096;
const CROSS_ATTENTION_DIM: usize = 2048;

pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "kolors",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets — one cached latent per (item, bucket).
        // sc-24828 (epic 2123): subject-masked loss.
        // sc-24830 (epic 2123): depth anchoring through the shared perceptual path (TAESDXL +
        // Depth-Anything-V2) on the trainer's one (dense) loss path.
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

/// One item's item-major cache entries (sc-2127 × sc-24828): the decoded `square` encoded at each
/// bucket edge by `encode` (`[1, 3, edge, edge]` → clean latent), each paired with its subject-mask
/// loss weight on THAT bucket's latent grid (`None` when masked loss is off). `decode_square`
/// center-crops, so the mask is cropped with [`CropBox::center_square`].
fn encode_item_buckets(
    square: &SquareImage,
    edges: &[u32],
    mask: Option<&PreparedSubjectMask>,
    device: &Device,
    mut encode: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Vec<(Tensor, Option<Tensor>)>> {
    edges
        .iter()
        .map(|&edge| {
            let x0 = encode(&square_image_tensor(square, edge, device)?)?;
            let mask_weight = prepared_subject_mask_weight(
                LABEL,
                mask,
                CropBox::center_square,
                x0.dims(),
                device,
            )?;
            Ok((x0, mask_weight))
        })
        .collect()
}

/// One SDXL-style `time_ids` row (`[h, w, 0, 0, h, w]`) per bucket edge, in bucket order, at
/// `dtype` (sc-2127). A step selects the row of the bucket its cached latent was encoded at, so the
/// micro-conditioning always names the size the latent really has.
fn bucket_time_ids(device: &Device, edges: &[u32], dtype: DType) -> Result<Vec<Tensor>> {
    edges
        .iter()
        .map(|&edge| Ok(build_time_ids(device, 1, edge, edge)?.to_dtype(dtype)?))
        .collect()
}

pub struct KolorsTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
}

pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "kolors trainer expects a snapshot directory".into(),
            ))
        }
    };
    if spec.precision != Precision::Bf16
        || spec.quantize.is_some()
        || packed_component(&root, "unet")?
    {
        return Err(CandleError::Msg(
            "kolors trainer requires the dense bf16 base tier; packed/quantized weights are not trainable"
                .into(),
        ));
    }
    Ok(Box::new(KolorsTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl Trainer for KolorsTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        validate_request(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        self.validate(req)?;
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err(CandleError::Msg(format!("{LABEL}: dataset is empty")));
    }
    if req.config.rank == 0 || req.config.steps == 0 {
        return Err(CandleError::Msg(format!(
            "{LABEL}: rank and steps must be > 0"
        )));
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(CandleError::Msg(format!(
            "{LABEL}: optimizer '{}' is not supported",
            req.config.optimizer
        )));
    }
    if req.config.resume {
        return Err(CandleError::Msg(format!(
            "{LABEL}: resume is not yet supported"
        )));
    }
    if req.config.gradient_checkpointing {
        return Err(CandleError::Msg(format!(
            "{LABEL}: gradient checkpointing is not yet supported"
        )));
    }
    if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
        return Err(CandleError::Msg(format!(
            "{LABEL}: in-training previews are not yet supported"
        )));
    }
    if req
        .items
        .iter()
        .any(|item| item.control_image_path.is_some())
    {
        return Err(CandleError::Msg(format!(
            "{LABEL}: per-item control/source images are not consumed"
        )));
    }
    Ok(())
}

fn packed_component(root: &Path, component: &str) -> Result<bool> {
    let path = root.join(component).join("config.json");
    if !path.is_file() {
        return Ok(false);
    }
    let bytes = std::fs::read(&path)
        .map_err(|e| CandleError::Msg(format!("read {}: {e}", path.display())))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| CandleError::Msg(format!("parse {}: {e}", path.display())))?;
    Ok(value.get("quantization").is_some())
}

/// A narrow seam makes the conditioning provenance testable: production constructs only this
/// ChatGLM-backed implementation, so the U-Net trainer cannot accidentally substitute SDXL CLIP.
trait CaptionEncoder {
    fn encode(&self, caption: &str) -> Result<(Tensor, Tensor)>;
}

struct ChatGlmCaptionEncoder {
    tokenizer: KolorsTokenizer,
    model: ChatGlmModel,
}

impl CaptionEncoder for ChatGlmCaptionEncoder {
    fn encode(&self, caption: &str) -> Result<(Tensor, Tensor)> {
        Ok(self.model.encode_prompt(&self.tokenizer.encode(caption)?)?)
    }
}

fn cache_caption(encoder: &dyn CaptionEncoder, caption: &str) -> Result<(Tensor, Tensor)> {
    let (context, pooled) = encoder.encode(caption)?;
    Ok((context.detach(), pooled.detach()))
}

fn compute_dtype(name: &str) -> DType {
    if name.eq_ignore_ascii_case("bf16") || name.eq_ignore_ascii_case("bfloat16") {
        DType::BF16
    } else {
        DType::F32
    }
}

fn target_paths(unet: &mut UNet2DConditionModel, req: &TrainingRequest) -> Result<Vec<String>> {
    let suffixes = if req.config.lora_target_modules.is_empty() {
        SDXL_ATTN_TARGETS
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    } else {
        req.config.lora_target_modules.clone()
    };
    let paths = unet
        .lora_target_paths()?
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|suffix| path == suffix || path.ends_with(&format!(".{suffix}")))
        })
        .filter(|path| req.config.network_type != NetworkType::Lokr || !path.contains("mid_block"))
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Err(CandleError::Msg(format!(
            "{LABEL}: no adapter targets matched {suffixes:?}"
        )));
    }
    Ok(paths)
}

fn save_adapter(set: &LoraSet, path: &Path) -> Result<()> {
    match set.kind {
        AdapterKind::Lora => save_lora_peft(set, SDXL_PEFT_PREFIX, &HashMap::new(), path),
        AdapterKind::Lokr => save_lokr(set, &HashMap::new(), path),
    }
}

fn ddpm_noise(schedule: &AlphaSchedule, x0: &Tensor, noise: &Tensor, t: usize) -> Result<Tensor> {
    let alpha = schedule.alphas_cumprod[t] as f64;
    Ok(((x0 * alpha.sqrt())? + (noise * (1.0 - alpha).sqrt())?)?)
}

/// ε-prediction loss in f32. `weight` is the item's subject-mask loss weight (sc-24828), broadcast
/// to the latent shape: `None` is exactly `mean(ℓ)`, `Some(w)` is `mean(w ⊙ ℓ)`.
fn epsilon_loss(
    prediction: &Tensor,
    noise: &Tensor,
    weight: Option<&Tensor>,
    mae: bool,
) -> Result<Tensor> {
    Ok(weighted_velocity_loss(
        prediction,
        &noise.to_dtype(DType::F32)?,
        weight,
        mae,
    )?)
}

/// Kolors' latent family for the shared aux-loss builder (epic 2123 E8): the SDXL 4-channel VAE
/// latent (`mean × 0.13025`), decoded by TAESDXL (it decodes the scaled latent directly);
/// latent-LPIPS family SDXL.
fn aux_loss_context(device: &Device) -> AuxLossContext<'_> {
    AuxLossContext {
        label: LABEL,
        decoder: taesdxl_decoder(),
        latent_lpips: Some(gen_core::train::LatentLpipsFamily::Sdxl),
        device,
    }
}

fn taesdxl_decoder() -> DecoderSpec {
    DecoderSpec::Tiny {
        name: "TAESDXL",
        config: TinyDecoderSpec::taesdxl(),
    }
}

/// The epic-2123 perceptual path for `cfg`: `None` when no aux loss is enabled (nothing loads).
fn load_perceptual_path(
    cfg: &gen_core::train::TrainingConfig,
    device: &Device,
) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(cfg, &aux_loss_context(device))
}

/// Extra training memory (bytes) of the enabled perceptual losses for `items` items: TAESDXL + each
/// loss at the largest bucket edge, plus one reference per (item, bucket) entry (E7).
fn perceptual_footprint_bytes(cfg: &gen_core::train::TrainingConfig, items: usize) -> u64 {
    let edges = bucket_edges(cfg);
    let edge = edges.iter().copied().max().unwrap_or(0);
    candle_gen_perceptual::perceptual_footprint(
        cfg,
        &taesdxl_decoder(),
        AuxGeometry::image(edge, items * edges.len()),
    )
}

/// Epic 2123 E7 preflight: with a perceptual loss on, the UNet's resident weights (`unet/`
/// safetensors — the lower bound; this trainer has no fitted activation model and only a dense
/// path) plus the aux footprint must fit `budget_bytes`. Depth off ⇒ no check.
fn aux_memory_preflight(
    root: &Path,
    cfg: &gen_core::train::TrainingConfig,
    items: usize,
    budget_bytes: u64,
) -> Result<()> {
    let aux = perceptual_footprint_bytes(cfg, items);
    if aux == 0 {
        return Ok(());
    }
    let base = flow_match::component_bytes(root, "unet", LABEL)?;
    flow_match::check_aux_memory(LABEL, base, aux, budget_bytes)
}

/// The `[0, 1]` noise level of DDPM timestep `t` (`t / (T − 1)`, 1 = pure noise).
fn noise_level_of(t: usize) -> f32 {
    t as f32 / (NUM_TRAIN_TIMESTEPS - 1) as f32
}

/// The DDPM timestep training a `[0, 1]` noise level (`round(level · (T − 1))`).
fn timestep_at(level: f32) -> usize {
    ((level.clamp(0.0, 1.0) * (NUM_TRAIN_TIMESTEPS - 1) as f32).round() as usize)
        .min(NUM_TRAIN_TIMESTEPS - 1)
}

/// Epic 2123 E8: plan the step on timestep `t`'s noise level; an aux-only step trains at the
/// plan's remapped level (a diffusion step keeps `t` exactly). `(t, None)` without a perceptual path.
fn plan_timestep<'a>(sample: &StepSample<'a>, t: usize) -> Result<(usize, Option<AuxStep<'a>>)> {
    let aux = sample.plan(noise_level_of(t))?;
    let t = match aux.as_ref() {
        Some(a) if !a.diffusion() => timestep_at(a.noise_level()),
        _ => t,
    };
    Ok((t, aux))
}

/// One micro-step's forward+backward (Kolors' only loss path, dense): DDPM-noise `x0` at
/// `timestep` with the trainer's [`AlphaSchedule`], predict ε through the add-embedding UNet,
/// regress ε→noise and — epic 2123 E8, when `aux` plans one — add the weighted perceptual term on
/// the ε-prediction's x0 `(x_t − √(1−ᾱ)·ε)/√ᾱ` (ᾱ = the same `alphas_cumprod[t]` the noising used).
/// On an aux-only step the diffusion term is not computed; `aux = None` is exactly the
/// pre-epic-2123 step. `projected` is the projected ChatGLM context at the compute dtype.
#[allow(clippy::too_many_arguments)]
fn kolors_step(
    unet: &UNet2DConditionModel,
    schedule: &AlphaSchedule,
    x0: &Tensor,
    projected: &Tensor,
    pooled: &Tensor,
    time_ids: &Tensor,
    timestep: usize,
    noise: &Tensor,
    mask_weight: Option<&Tensor>,
    mae: bool,
    dtype: DType,
    aux: Option<&AuxStep<'_>>,
) -> Result<(StepLosses, GradStore)> {
    let noisy_f32 = ddpm_noise(schedule, x0, noise, timestep)?;
    let noisy = noisy_f32.to_dtype(dtype)?;
    let prediction = unet.forward_instantid(
        &noisy,
        timestep as f64,
        projected,
        &pooled.to_dtype(dtype)?,
        time_ids,
        None,
        None,
    )?;
    let (diffusion_on, aux_on) = step_terms(aux);
    let diffusion = if diffusion_on {
        Some(epsilon_loss(&prediction, noise, mask_weight, mae)?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let alpha_bar = schedule.alphas_cumprod[timestep];
            let x0_hat = Parameterization::Epsilon { alpha_bar }
                .recover_x0(&noisy_f32, &prediction.to_dtype(DType::F32)?)?;
            a.aux_loss(&x0_hat)?
        }
        _ => None,
    };
    let (loss, losses) = combine_terms(diffusion, aux_term)?;
    let grads = loss.backward()?;
    Ok((losses, grads))
}

impl KolorsTrainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let cfg = &req.config;
        let device = &self.device;
        let dtype = compute_dtype(&cfg.train_dtype);
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are off).
        let edges = bucket_edges(cfg);
        on_progress(TrainingProgress::Preparing);
        // Epic 2123 E7 + E8: admit, then load the perceptual models BEFORE caching (a missing
        // checkpoint fails fast; `None` — nothing loaded — when no aux loss is on).
        aux_memory_preflight(
            &self.root,
            cfg,
            req.items.len(),
            flow_match::device_training_budget_bytes(device, LABEL),
        )?;
        let perceptual = load_perceptual_path(cfg, device)?;
        on_progress(TrainingProgress::LoadingModel);

        let vae = VaeMomentsEncoder::new(
            candle_gen::load_sorted_mmap(&self.root.join("vae"), DType::F32, device, LABEL)?,
            VAE_SCALE,
        )?;
        let caption_encoder = ChatGlmCaptionEncoder {
            tokenizer: KolorsTokenizer::from_dir(self.root.join("tokenizer"))?,
            model: ChatGlmModel::new(
                ChatGlmConfig::chatglm3_6b(),
                candle_gen::load_sorted_mmap(
                    &self.root.join("text_encoder"),
                    DType::BF16,
                    device,
                    LABEL,
                )?,
            )?,
        };
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127).
        let mut cache = Vec::with_capacity(req.items.len() * edges.len());
        for (index, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total: req.items.len() as u32,
            });
            let (context, pooled) = cache_caption(&caption_encoder, &item.caption)?;
            let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
                                                           // The item's subject mask, read + checked once (None when masked loss is off); each
                                                           // bucket's weight (sc-24828) is broadcast to that bucket's latent shape.
            let mask =
                PreparedSubjectMask::load_if_enabled(LABEL, item, cfg.subject_mask_loss.as_ref())?;
            let buckets = encode_item_buckets(&square, &edges, mask.as_ref(), device, |image| {
                Ok(vae.encode_mean(image)?.detach())
            })?;
            for (x0, mask_weight) in buckets {
                cache.push((x0, context.clone(), pooled.clone(), mask_weight));
            }
        }
        drop(caption_encoder);
        drop(vae);
        if cache.is_empty() {
            return Err(if req.cancel.is_cancelled() {
                CandleError::Canceled
            } else {
                CandleError::Msg(format!("{LABEL}: no usable dataset items"))
            });
        }

        let vb = candle_gen::load_sorted_mmap(&self.root.join("unet"), dtype, device, LABEL)?;
        let context_projection =
            linear(CONTEXT_DIM, CROSS_ATTENTION_DIM, vb.pp("encoder_hid_proj"))?;
        let mut unet = UNet2DConditionModel::new(vb.clone(), 4, 4, false, sdxl_unet_config())?
            .with_add_embedding(vb, ADDITION_TIME_EMBED_DIM, PROJECTION_INPUT_DIM)?;
        let targets = target_paths(&mut unet, req)?;
        let set = match cfg.network_type {
            NetworkType::Lora => {
                build_lora_targets(&mut unet, &targets, cfg.rank, cfg.alpha, cfg.seed, device)?
            }
            NetworkType::Lokr => build_lokr_targets(
                &mut unet,
                &targets,
                cfg.rank,
                cfg.alpha,
                cfg.decompose_factor,
                cfg.seed,
                device,
            )?,
        };
        let schedule = AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END);
        let accum = cfg.gradient_accumulation.max(1);
        let mut optimizer = TrainOptimizer::from_config(
            &cfg.optimizer,
            set.vars.clone(),
            cfg.learning_rate,
            effective_weight_decay(cfg),
        )?;
        let (updates, warmup) = schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let mut accumulated = None;
        let mut update = 0;
        let mut steps_run = 0;
        let mut last_loss = 0.0;
        let time_ids = bucket_time_ids(device, &edges, dtype)?;
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let sample_order =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        let mae = matches!(cfg.loss_type.to_ascii_lowercase().as_str(), "mae" | "l1");
        let stem = file_stem(&req.file_name).to_string();
        // Epic 2123 E8: references per (item, bucket) entry once (the cached latent is already
        // TAESDXL's NCHW input), alternation keyed on the real item.
        // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
        // cropped like the image and resampled onto its decoded size.
        let mut perceptual = perceptual;
        if let Some(path) = perceptual.as_mut() {
            path.attach_subject_masks(
                candle_gen::gen_core::train::subject_mask::PerceptualSubjectMasks::load(
                    "kolors trainer",
                    &req.items,
                    cfg,
                    edges.len(),
                    CropBox::center_square,
                )?,
            );
        }
        let mut aux_driver = match perceptual {
            Some(path) => Some(AuxDriver::prepare(
                path,
                cache.len(),
                |i| Ok(cache[i].0.clone()),
                &sample_order,
                accum,
                0,
            )?),
            None => None,
        };

        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let sample = step_sample(aux_driver.as_mut(), step, &sample_order);
            let index = sample.entry;
            let (x0, context, pooled, mask_weight) = &cache[index];
            let step_time_ids = &time_ids[index % edges.len()];
            let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(step as u64));
            let timestep = rng.random_range(0..NUM_TRAIN_TIMESTEPS);
            let noise = sample_noise(x0.dims(), noise_seed(cfg.seed, step), device)?;
            let (timestep, aux) = plan_timestep(&sample, timestep)?;
            let projected = context_projection.forward(&context.to_dtype(dtype)?)?;
            let (losses, grads) = kolors_step(
                &unet,
                &schedule,
                x0,
                &projected,
                pooled,
                step_time_ids,
                timestep,
                &noise,
                mask_weight.as_ref(),
                mae,
                dtype,
                aux.as_ref(),
            )?;
            last_loss = losses.total;
            accumulate_grads(&mut accumulated, grads, &set.vars)?;
            steps_run = step;

            if step % accum == 0 || step == cfg.steps {
                optimizer.set_lr_scaled(lr_multiplier(cfg.lr_scheduler, update, updates, warmup));
                let mut grads = accumulated
                    .take()
                    .expect("an update has accumulated gradients");
                let divisor = accumulation_divisor(step, accum);
                scale_grads(&mut grads, &set.vars, 1.0 / divisor as f64)?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_step(&mut optimizer, &mut grads, &set, cfg, update, cfg.seed)?;
                update += 1;
            }
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });
            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir).map_err(|e| {
                    CandleError::Msg(format!("create {}: {e}", req.output_dir.display()))
                })?;
                save_adapter(&set, &req.output_dir.join(checkpoint_filename(&stem, step)))?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }
        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)
            .map_err(|e| CandleError::Msg(format!("create {}: {e}", req.output_dir.display())))?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(&set, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem};

    use super::*;

    struct SentinelChatGlm<'a> {
        called: &'a Cell<bool>,
    }

    impl CaptionEncoder for SentinelChatGlm<'_> {
        fn encode(&self, caption: &str) -> Result<(Tensor, Tensor)> {
            assert_eq!(caption, "ChatGLM conditioning");
            self.called.set(true);
            Ok((
                Tensor::new(&[[[7.0f32]]], &Device::Cpu)?,
                Tensor::new(&[[11.0f32]], &Device::Cpu)?,
            ))
        }
    }

    #[test]
    fn training_caption_route_is_chatglm_conditioning_not_sdxl_clip() {
        let called = Cell::new(false);
        let (context, pooled) =
            cache_caption(&SentinelChatGlm { called: &called }, "ChatGLM conditioning").unwrap();
        assert!(called.get());
        assert_eq!(context.to_vec3::<f32>().unwrap(), vec![vec![vec![7.0]]]);
        assert_eq!(pooled.to_vec2::<f32>().unwrap(), vec![vec![11.0]]);
        let _: fn(&ChatGlmCaptionEncoder, &str) -> Result<(Tensor, Tensor)> =
            <ChatGlmCaptionEncoder as CaptionEncoder>::encode;
    }

    #[test]
    fn kolors_ddpm_uses_direct_1100_step_alpha_index() {
        let schedule = AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END);
        let x0 = Tensor::new(&[2.0f32], &Device::Cpu).unwrap();
        let noise = Tensor::new(&[3.0f32], &Device::Cpu).unwrap();
        let t = 1099;
        let got = ddpm_noise(&schedule, &x0, &noise, t)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let alpha = schedule.alphas_cumprod[t];
        let expected = alpha.sqrt() * 2.0 + (1.0 - alpha).sqrt() * 3.0;
        assert!((got - expected).abs() < 1e-6);
    }

    /// sc-24828: the ε loss honours the subject-mask weight — an all-ones map is the unweighted
    /// loss, an all-zero map zeroes the loss and the prediction's gradient — and the trainer
    /// declares the technique.
    #[test]
    fn epsilon_loss_applies_the_subject_mask_weight() {
        use candle_gen::candle_core::Var;
        let dev = Device::Cpu;
        let shape = [1usize, 4, 2, 2];
        let pred = Var::from_tensor(&Tensor::randn(0f32, 1f32, &shape, &dev).unwrap()).unwrap();
        let noise = Tensor::randn(0f32, 1f32, &shape, &dev).unwrap();
        let map = |w: &[f32]| {
            candle_gen::train::flow_match::subject_mask_weight(w, 2, 2, &shape, &dev).unwrap()
        };
        let loss = |w: Option<&Tensor>| epsilon_loss(pred.as_tensor(), &noise, w, false).unwrap();
        let plain = loss(None).to_scalar::<f32>().unwrap();
        let ones = loss(Some(&map(&[1.0; 4]))).to_scalar::<f32>().unwrap();
        assert!((ones - plain).abs() < 1e-6);
        let zero = loss(Some(&map(&[0.0; 4])));
        assert_eq!(zero.to_scalar::<f32>().unwrap(), 0.0);
        let g = zero.backward().unwrap();
        let g = g.get(pred.as_tensor()).unwrap().flatten_all().unwrap();
        assert!(g.to_vec1::<f32>().unwrap().iter().all(|x| *x == 0.0));
        let half = loss(Some(&map(&[1.0, 0.0, 1.0, 0.0])))
            .to_scalar::<f32>()
            .unwrap();
        assert!(half > 0.0 && half < plain);
        assert!(trainer_descriptor().techniques.subject_mask_loss);
    }

    /// sc-2127: the trainer declares buckets, every bucket gets the `time_ids` of its own edge, and
    /// the item-major cache index a step samples maps (`index % n_buckets`) to the bucket the
    /// schedule chose — so the micro-conditioning never names a size the latent does not have.
    #[test]
    fn each_bucket_conditions_on_its_own_edge() {
        use candle_gen::gen_core::ResolutionBucket;
        assert!(trainer_descriptor().techniques.resolution_buckets);
        let edges = [512u32, 1024];
        let rows = bucket_time_ids(&Device::Cpu, &edges, DType::F32).unwrap();
        assert_eq!(rows.len(), 2);
        for (row, edge) in rows.iter().zip(edges) {
            let e = edge as f32;
            assert_eq!(
                row.to_vec2::<f32>().unwrap(),
                vec![vec![e, e, 0.0, 0.0, e, e]]
            );
        }
        let buckets = [
            ResolutionBucket {
                resolution: 512,
                repeats: 4,
            },
            ResolutionBucket {
                resolution: 1024,
                repeats: 1,
            },
        ];
        let order = BucketSchedule::new(3, &buckets, 9);
        for k in 0..60 {
            assert_eq!(
                order.cache_index(k) % edges.len(),
                order.sample(k).1,
                "k {k}"
            );
        }
    }

    /// sc-24828 × sc-2127: with masked loss on and two buckets, each cached latent carries a weight
    /// of ITS OWN shape (built on that bucket's grid), and the masked-out region is zero.
    #[test]
    fn subject_mask_weight_follows_each_buckets_latent() {
        use candle_gen::candle_core::IndexOp;
        use candle_gen::gen_core::SubjectMaskLoss;
        let dir = tempfile::tempdir().unwrap();
        // 48×32 image → center square x ∈ [8, 40); the subject is that square's left half (x < 24).
        let image_path = dir.path().join("img.png");
        image::RgbImage::from_pixel(48, 32, image::Rgb([128, 64, 32]))
            .save(&image_path)
            .unwrap();
        let mask_path = dir.path().join("mask.png");
        image::GrayImage::from_fn(48, 32, |x, _| image::Luma([if x < 24 { 255 } else { 0 }]))
            .save(&mask_path)
            .unwrap();
        let item = TrainingItem {
            image_path,
            caption: String::new(),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: Some(mask_path),
        };
        let cfg = SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask = PreparedSubjectMask::load("t", &item, &cfg).unwrap();
        let square = decode_square(&item.image_path).unwrap();
        let dev = Device::Cpu;
        // A stand-in /8 encoder: `[1, 3, edge, edge]` → `[1, 3, edge/8, edge/8]`.
        let encode = |img: &Tensor| Ok(img.avg_pool2d(8)?);
        let entries = encode_item_buckets(&square, &[32, 64], Some(&mask), &dev, encode).unwrap();
        assert_eq!(entries.len(), 2);
        for ((x0, w), grid) in entries.iter().zip([4usize, 8]) {
            assert_eq!(x0.dims(), &[1, 3, grid, grid]);
            let w = w.as_ref().expect("masked loss is on");
            assert_eq!(w.dims(), x0.dims(), "bucket {grid}: weight shape");
            let rows = w.i((0, 0)).unwrap().to_vec2::<f32>().unwrap();
            for row in rows {
                for (x, v) in row.into_iter().enumerate() {
                    let want = if x < grid / 2 { 1.0 } else { 0.0 };
                    assert_eq!(v, want, "bucket {grid}: column {x}");
                }
            }
        }
        let off = encode_item_buckets(&square, &[32, 64], None, &dev, encode).unwrap();
        assert!(off.iter().all(|(_, w)| w.is_none()));
    }

    #[test]
    fn nondivisible_accumulation_tail_uses_its_actual_micro_count() {
        assert_eq!(accumulation_divisor(4, 4), 4);
        assert_eq!(accumulation_divisor(5, 4), 1);
        assert_eq!(accumulation_divisor(7, 4), 3);
    }

    #[test]
    fn validate_rejects_per_item_control_or_source_images() {
        let request = |control_image_path| TrainingRequest {
            items: vec![TrainingItem {
                image_path: "/image.png".into(),
                caption: "caption".into(),
                control_image_path,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            }],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        };
        assert!(validate_request(&request(None)).is_ok());
        assert!(validate_request(&request(Some("/control.png".into()))).is_err());
    }

    /// Epic 2123 depth anchoring (sc-24830) on the candle Kolors trainer: a tiny add-embedding
    /// SDXL-family UNet + a random-init tiny 4-channel decoder + tiny DA2, through [`kolors_step`].
    mod depth_anchoring {
        use super::*;
        use candle_gen::candle_core::Var;
        use candle_gen::candle_nn::{VarBuilder, VarMap};
        use candle_gen::gen_core::train::{AuxLossSchedule, DepthModelSize, TrainingConfig};
        use candle_gen_sdxl::{BlockConfig, UNet2DConditionModelConfig};

        fn schedule() -> AuxLossSchedule {
            AuxLossSchedule {
                weight: 0.5,
                t_min: 0.6,
                t_max: 0.9,
                every_n: 2,
            }
        }

        struct Fixture {
            unet: UNet2DConditionModel,
            vars: Vec<Var>,
            alphas: AlphaSchedule,
            x0: Tensor,
            projected: Tensor,
            pooled: Tensor,
            time_ids: Tensor,
            noise: Tensor,
        }

        /// A tiny Kolors-shaped UNet: the SDXL-family blocks + `add_embedding` over
        /// `cat[pooled(16), add_time_proj(time_ids)(6 × 8)]`.
        fn fixture() -> Fixture {
            let dev = Device::Cpu;
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let cfg = UNet2DConditionModelConfig {
                blocks: vec![
                    BlockConfig {
                        out_channels: 32,
                        use_cross_attn: Some(1),
                        attention_head_dim: 8,
                    },
                    BlockConfig {
                        out_channels: 64,
                        use_cross_attn: None,
                        attention_head_dim: 8,
                    },
                ],
                center_input_sample: false,
                cross_attention_dim: 64,
                downsample_padding: 1,
                flip_sin_to_cos: true,
                freq_shift: 0.,
                layers_per_block: 1,
                mid_block_scale_factor: 1.,
                norm_eps: 1e-5,
                norm_num_groups: 32,
                use_linear_projection: false,
            };
            let mut unet = UNet2DConditionModel::new(vb.clone(), 4, 4, false, cfg)
                .unwrap()
                .with_add_embedding(vb, 8, 16 + 6 * 8)
                .unwrap();
            let paths: Vec<String> = unet
                .lora_target_paths()
                .unwrap()
                .into_iter()
                .filter(|p| {
                    SDXL_ATTN_TARGETS
                        .iter()
                        .any(|s| p.ends_with(&format!(".{s}")))
                })
                .collect();
            assert!(!paths.is_empty());
            let set = build_lora_targets(&mut unet, &paths, 4, 8.0, 7, &dev).unwrap();
            for v in &set.vars {
                v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                    .unwrap();
            }
            Fixture {
                unet,
                vars: set.vars,
                alphas: AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END),
                x0: Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap(),
                projected: Tensor::randn(0f32, 1f32, (1, 7, 64), &dev).unwrap(),
                pooled: Tensor::randn(0f32, 1f32, (1, 16), &dev).unwrap(),
                time_ids: build_time_ids(&dev, 1, 64, 64).unwrap(),
                noise: Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &dev).unwrap(),
            }
        }

        /// Step 1 = diffusion step, step 2 = the item's 2nd window ⇒ aux-only.
        fn driver(f: &Fixture) -> (AuxDriver, BucketSchedule) {
            let path = candle_gen_perceptual::testing::tiny_depth_path(4, schedule(), &Device::Cpu)
                .unwrap();
            let sched = BucketSchedule::new(1, &[], 3);
            let x0 = f.x0.clone();
            let d = AuxDriver::prepare(path, 1, |_| Ok(x0.clone()), &sched, 1, 0).unwrap();
            (d, sched)
        }

        fn run(f: &Fixture, t: usize, aux: Option<&AuxStep<'_>>) -> (StepLosses, GradStore) {
            kolors_step(
                &f.unet,
                &f.alphas,
                &f.x0,
                &f.projected,
                &f.pooled,
                &f.time_ids,
                t,
                &f.noise,
                None,
                false,
                DType::F32,
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

        /// The ε recovery uses the noising ᾱ (`alphas_cumprod[t]` of the 1100-step schedule): with
        /// ε = the true noise it returns x0; the level ↔ timestep mapping round-trips. Mutation:
        /// drop the `.round()` in `timestep_at` ⇒ red.
        #[test]
        fn epsilon_recovery_uses_the_noising_alpha_bar() {
            let f = fixture();
            for t in [10usize, 550, 1090] {
                let noisy = ddpm_noise(&f.alphas, &f.x0, &f.noise, t).unwrap();
                let rec = Parameterization::Epsilon {
                    alpha_bar: f.alphas.alphas_cumprod[t],
                }
                .recover_x0(&noisy, &f.noise)
                .unwrap();
                let err = (rec - &f.x0)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(err < 1e-3, "t={t}: x0 recovery error {err}");
                assert_eq!(timestep_at(noise_level_of(t)), t);
            }
            // Half-way rounds to the nearer index (0.5 · (T − 1) ends in .5).
            assert_eq!(timestep_at(0.5), 550);
        }

        /// (a)/(b): the diffusion step has no aux term; the aux-only step has no diffusion term,
        /// total == aux and a nonzero finite LoRA B gradient; the loop's timestep is the remapped
        /// one. Mutations: compute the diffusion term unconditionally ⇒ red; return the sampled `t`
        /// from `plan_timestep` on an aux step ⇒ red.
        #[test]
        fn aux_only_step_trains_the_adapter_through_depth_alone() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let (t1, s1) = plan_timestep(&d.sample(1, &sched), 300).unwrap();
            let s1 = s1.unwrap();
            assert_eq!(t1, 300);
            assert!(s1.diffusion() && !s1.has_aux());
            let (l, _) = run(&f, t1, Some(&s1));
            assert!(l.diffusion.is_some() && l.aux.is_none(), "{l:?}");
            assert_eq!(l.total, l.diffusion.unwrap());
            let (t2, s2) = plan_timestep(&d.sample(2, &sched), 300).unwrap();
            let s2 = s2.unwrap();
            assert!(!s2.diffusion() && s2.has_aux());
            assert!((659..=990).contains(&t2), "remapped into the window: {t2}");
            let (l, g) = run(&f, t2, Some(&s2));
            assert!(l.diffusion.is_none(), "{l:?}");
            let aux = l.aux.expect("aux term");
            assert!(aux.is_finite() && aux > 0.0, "{aux}");
            assert_eq!(l.total, aux);
            let mut nonzero_b = false;
            for (i, v) in f.vars.iter().enumerate() {
                let gv: Vec<f32> = grad_bits(&g, v).into_iter().map(f32::from_bits).collect();
                assert!(gv.iter().all(|x| x.is_finite()), "var {i}");
                nonzero_b |= i % 2 == 1 && gv.iter().any(|x| *x != 0.0);
            }
            assert!(nonzero_b, "no LoRA B gradient from the depth term");
            assert_eq!(plan_timestep(&StepSample::plain(0, 0), 300).unwrap().0, 300);
        }

        /// The aux term is the depth loss of THE trainer's x0 estimate: recomputed independently from
        /// the model output with the trainer's parameterisation, it matches the step's aux bits.
        /// Mutation: recover with `alphas_cumprod[timestep - 1]` in `kolors_step` ⇒ red.
        #[test]
        fn aux_term_is_the_depth_loss_of_the_recovered_x0() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let _ = d.sample(1, &sched);
            let s2 = d.sample(2, &sched).plan(0.5).unwrap().unwrap();
            let t = timestep_at(s2.noise_level());
            let (l, _) = run(&f, t, Some(&s2));
            let noisy = ddpm_noise(&f.alphas, &f.x0, &f.noise, t).unwrap();
            let eps = f
                .unet
                .forward_instantid(
                    &noisy,
                    t as f64,
                    &f.projected,
                    &f.pooled,
                    &f.time_ids,
                    None,
                    None,
                )
                .unwrap();
            let ab = f.alphas.alphas_cumprod[t] as f64;
            let x0 = ((&noisy - (eps * (1.0 - ab).sqrt()).unwrap()).unwrap() / ab.sqrt()).unwrap();
            let want = s2
                .aux_loss(&x0)
                .unwrap()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let got = l.aux.unwrap();
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }

        /// (c) Depth off is bit-identical to the pre-epic-2123 step body (reproduced verbatim).
        /// Mutation: perturb the `None` combine ⇒ red.
        #[test]
        fn depth_off_is_bit_identical_to_the_legacy_step() {
            let f = fixture();
            let (l, g) = run(&f, 500, None);
            assert_eq!((l.diffusion, l.aux), (Some(l.total), None));
            let noisy = ddpm_noise(&f.alphas, &f.x0, &f.noise, 500).unwrap();
            let prediction = f
                .unet
                .forward_instantid(
                    &noisy,
                    500.0,
                    &f.projected,
                    &f.pooled,
                    &f.time_ids,
                    None,
                    None,
                )
                .unwrap();
            let loss = epsilon_loss(&prediction, &f.noise, None, false).unwrap();
            let legacy = loss.to_scalar::<f32>().unwrap();
            let lg = loss.backward().unwrap();
            assert_eq!(l.total.to_bits(), legacy.to_bits());
            for v in &f.vars {
                assert_eq!(grad_bits(&g, v), grad_bits(&lg, v));
            }
        }

        fn depth_on() -> TrainingConfig {
            let mut cfg = TrainingConfig::default();
            cfg.depth_anchoring.schedule = schedule();
            cfg
        }

        /// (d) E7: footprint 0 off, larger for Large DA2; the preflight refuses between base and
        /// base + aux (Kolors has only the dense path; checkpointing is refused by `validate`).
        /// Mutation: skip the guard ⇒ red.
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
            let udir = tmp.path().join("unet");
            std::fs::create_dir_all(&udir).unwrap();
            std::fs::write(udir.join("m.safetensors"), vec![0u8; 4096]).unwrap();
            let base = 4096u64;
            assert!(aux_memory_preflight(tmp.path(), &TrainingConfig::default(), 4, 1).is_ok());
            assert!(aux_memory_preflight(tmp.path(), &cfg, 4, base + small).is_ok());
            let e = aux_memory_preflight(tmp.path(), &cfg, 4, base + small - 1)
                .unwrap_err()
                .to_string();
            assert!(e.contains("perceptual"), "{e}");
        }

        /// (e) The descriptor declares depth anchoring; depth off builds no path; a missing decoder
        /// dir names TAESDXL. Mutation: `depth_anchoring: false` ⇒ red.
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
            assert!(e.contains("TAESDXL"), "{e}");
            cfg.perceptual_decoder_dir = Some(tmp.path().join("no-taesdxl"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAESDXL"), "{e}");
        }
    }
}
