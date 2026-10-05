//! Dense Candle training for the SD3.5 Large and Medium MMDiTs. Caption conditioning and VAE
//! latents are cached first; the frozen encoders are then dropped before the dense transformer and
//! its trainable forward-time LoRA/LoKr residuals are loaded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    self, BucketSchedule, LoadSpec, Modality, NetworkType, Precision, WeightsSource,
};
use candle_gen::quant::AdaptLinear;
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor};
use candle_gen::train::flow_match::{
    self, combine_terms, prepared_subject_mask_weight, step_sample, step_terms,
    weighted_velocity_loss, AuxDriver, AuxStep, StepLosses, StepSample,
};
use candle_gen::train::lora::{build_adapt_lokr_targets, build_adapt_lora_targets, AdaptLoraHost};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::perceptual::{Parameterization, PerceptualPath};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::train::tae::TinyDecoderSpec;
use candle_gen::{CandleError, Result};
use candle_gen_perceptual::{AuxGeometry, AuxLossContext, DecoderSpec};

use crate::conditioning::{aggregate, Sd3Conditioning};
use crate::pipeline::{Pipeline, Variant};
use crate::transformer::Sd3Transformer;
use crate::vae::encode_mean;
use crate::{MODEL_ID, MODEL_ID_MEDIUM};

const LABEL: &str = "sd3 trainer";
const DEFAULT_TARGETS: [&str; 8] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
];
const TIMESTEP_TYPES: [&str; 6] = [
    "logit_normal",
    "default",
    "sigmoid",
    "linear",
    "uniform",
    "weighted",
];

fn descriptor_for(variant: Variant) -> TrainerDescriptor {
    TrainerDescriptor {
        id: match variant {
            Variant::Large => MODEL_ID,
            Variant::Medium => MODEL_ID_MEDIUM,
            Variant::LargeTurbo => unreachable!("the distilled Large Turbo is not a training base"),
        },
        family: "sd3",
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
        // sc-24828 (epic 2123): subject-masked loss — a per-bucket weight map cached next to each
        // latent.
        // sc-24830 (epic 2123): depth anchoring through the shared perceptual path (TAESD3 +
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

fn large_descriptor() -> TrainerDescriptor {
    descriptor_for(Variant::Large)
}

fn medium_descriptor() -> TrainerDescriptor {
    descriptor_for(Variant::Medium)
}

pub struct Sd3Trainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    variant: Variant,
}

fn load_for(spec: &LoadSpec, variant: Variant) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "sd3 trainer expects a snapshot directory (transformer/ text_encoder{,_2,_3}/ \
                 tokenizer{,_2,_3}/ vae/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() || packed_component(&root, "transformer")? {
        return Err(CandleError::Msg(
            "sd3 trainer requires a dense transformer tier; quantized training is unsupported"
                .into(),
        ));
    }
    let dtype = match spec.precision {
        Precision::Bf16 => DType::BF16,
        Precision::Fp32 => DType::F32,
    };
    Ok(Box::new(Sd3Trainer {
        descriptor: descriptor_for(variant),
        root,
        device: candle_gen::default_device()?,
        dtype,
        variant,
    }))
}

fn packed_component(root: &Path, component: &str) -> Result<bool> {
    let path = root.join(component).join("config.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: read {}: {error}",
                path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| CandleError::Msg(format!("{LABEL}: parse {}: {error}", path.display())))?;
    Ok(candle_gen::quant::PackedConfig::from_config(&config).is_some())
}

fn load_large(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_for(spec, Variant::Large)
}

fn load_medium(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_for(spec, Variant::Medium)
}

candle_gen::register_trainer! {
    pub(crate) const LARGE_TRAINER_REGISTRATION = large_descriptor => load_large
}
candle_gen::register_trainer! {
    pub(crate) const MEDIUM_TRAINER_REGISTRATION = medium_descriptor => load_medium
}

impl Trainer for Sd3Trainer {
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
        validate_request(req)?;
        let want_bf16 = {
            let dtype = req.config.train_dtype.trim();
            dtype.eq_ignore_ascii_case("bf16") || dtype.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == DType::BF16;
        if want_bf16 != loaded_bf16 {
            return Err(gen_core::Error::Msg(format!(
                "{LABEL}: train_dtype '{}' does not match loaded {} precision",
                req.config.train_dtype,
                if loaded_bf16 { "bf16" } else { "f32" }
            )));
        }
        Ok(())
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

impl AdaptLoraHost for Sd3Transformer {
    fn visit_adapt_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        self.visit_adaptable_mut(f)
    }
}

fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err(CandleError::Msg(format!("{LABEL}: dataset is empty")));
    }
    if cfg.rank == 0 {
        return Err(CandleError::Msg(format!("{LABEL}: rank must be > 0")));
    }
    if cfg.steps == 0 {
        return Err(CandleError::Msg(format!("{LABEL}: steps must be > 0")));
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(CandleError::Msg(format!(
            "{LABEL}: optimizer '{}' is not available (supported: adamw, adam, rose, prodigy)",
            cfg.optimizer
        )));
    }
    let timestep_type = flow_match::normalize_cfg(&cfg.timestep_type);
    if !TIMESTEP_TYPES.contains(&timestep_type.as_str()) {
        return Err(CandleError::Msg(format!(
            "{LABEL}: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )));
    }
    if !flow_match::TIMESTEP_BIASES
        .contains(&flow_match::normalize_cfg(&cfg.timestep_bias).as_str())
    {
        return Err(CandleError::Msg(format!(
            "{LABEL}: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            flow_match::TIMESTEP_BIASES.join(", ")
        )));
    }
    if !flow_match::LOSS_TYPES.contains(&flow_match::normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(CandleError::Msg(format!(
            "{LABEL}: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            flow_match::LOSS_TYPES.join(", ")
        )));
    }
    if cfg.resume {
        return Err(CandleError::Msg(
            "sd3 candle trainer does not yet support resume".into(),
        ));
    }
    if cfg.gradient_checkpointing {
        return Err(CandleError::Msg(
            "sd3 candle trainer does not yet support gradient checkpointing".into(),
        ));
    }
    if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() {
        return Err(CandleError::Msg(
            "sd3 candle trainer does not yet support in-training previews".into(),
        ));
    }
    if req
        .items
        .iter()
        .any(|item| item.control_image_path.is_some())
    {
        return Err(CandleError::Msg(
            "sd3 candle trainer does not consume per-item control/source images".into(),
        ));
    }
    Ok(())
}

fn sample_sigma(req: &TrainingRequest, step: u32) -> f64 {
    let cfg = &req.config;
    let timestep_type = flow_match::normalize_cfg(&cfg.timestep_type);
    let sampler = if matches!(timestep_type.as_str(), "default" | "logit_normal") {
        "sigmoid"
    } else {
        timestep_type.as_str()
    };
    flow_match::sample_unit_timestep(
        sampler,
        &cfg.timestep_bias,
        flow_match::timestep_seed(cfg.seed, step),
    ) as f64
}

fn target_suffixes(req: &TrainingRequest) -> Vec<String> {
    if req.config.lora_target_modules.is_empty() {
        DEFAULT_TARGETS
            .iter()
            .map(|target| target.to_string())
            .collect()
    } else {
        req.config.lora_target_modules.clone()
    }
}

/// SD3's latent family for the shared aux-loss builder (epic 2123 E8): the SD3 16-channel VAE latent
/// (`(mean − shift)·scale`), decoded by TAESD3 (diffusers `scaling_factor 1.0`, no shift: it decodes
/// that normalized latent directly); latent-LPIPS family SD3.
fn aux_loss_context(device: &Device) -> AuxLossContext<'_> {
    AuxLossContext {
        label: LABEL,
        decoder: taesd3_decoder(),
        latent_lpips: Some(gen_core::train::LatentLpipsFamily::Sd3),
        device,
    }
}

fn taesd3_decoder() -> DecoderSpec {
    DecoderSpec::Tiny {
        name: "TAESD3",
        config: TinyDecoderSpec::taesd3(),
    }
}

/// The epic-2123 perceptual path for `cfg`: `None` when no aux loss is enabled (nothing loads).
fn load_perceptual_path(
    cfg: &gen_core::train::TrainingConfig,
    device: &Device,
) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(cfg, &aux_loss_context(device))
}

/// Extra training memory (bytes) of the enabled perceptual losses for `items` items: TAESD3 + each
/// loss at the largest bucket edge, plus one reference per (item, bucket) entry (E7).
fn perceptual_footprint_bytes(cfg: &gen_core::train::TrainingConfig, items: usize) -> u64 {
    let edges = bucket_edges(cfg);
    let edge = edges.iter().copied().max().unwrap_or(0);
    candle_gen_perceptual::perceptual_footprint(
        cfg,
        &taesd3_decoder(),
        AuxGeometry::image(edge, items * edges.len()),
    )
}

/// Epic 2123 E7 preflight: with a perceptual loss on, the MMDiT's resident weights (`transformer/`
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
    let base = flow_match::component_bytes(root, "transformer", LABEL)?;
    flow_match::check_aux_memory(LABEL, base, aux, budget_bytes)
}

/// Epic 2123 E8: plan the step on the sampled `σ` (SD3's flow-match noise level, 1 = pure noise); an
/// aux-only step trains at the plan's remapped `σ`. `(σ, None)` without a perceptual path.
fn plan_sigma<'a>(sample: &StepSample<'a>, sigma: f64) -> Result<(f64, Option<AuxStep<'a>>)> {
    let aux = sample.plan(sigma as f32)?;
    let sigma = match aux.as_ref() {
        Some(a) if !a.diffusion() => a.noise_level() as f64,
        _ => sigma,
    };
    Ok((sigma, aux))
}

/// One micro-step's flow-match loss: noise `x0` at `sigma`, predict the raw velocity through the
/// (adapter-carrying) MMDiT at timestep `σ·1000`, regress it toward `noise − x0`. `mask_weight` is the
/// optional subject-mask loss weight (sc-24828, broadcast to the latent shape; `None` ⇒ exactly the
/// unweighted `velocity_loss`). `aux` (epic 2123 E8) adds the planned perceptual term on the model's
/// x0 estimate `x_t − σ·v` (already TAESD3's NCHW layout); on an aux-only step the diffusion term
/// is not computed. `aux = None` is exactly the pre-epic-2123 loss.
#[allow(clippy::too_many_arguments)]
fn step_loss(
    transformer: &Sd3Transformer,
    x0: &Tensor,
    conditioning: &Sd3Conditioning,
    noise: &Tensor,
    sigma: f64,
    dtype: DType,
    mae: bool,
    mask_weight: Option<&Tensor>,
    aux: Option<&AuxStep<'_>>,
) -> Result<(Tensor, StepLosses)> {
    let device = x0.device();
    let (x_t, target) = flow_match::build_batch(x0, noise, sigma)?;
    let timestep = Tensor::new(&[(sigma * 1000.0) as f32], device)?.to_dtype(dtype)?;
    let prediction = transformer.forward(
        &x_t.to_dtype(dtype)?,
        &conditioning.context.to_dtype(dtype)?,
        &conditioning.pooled.to_dtype(dtype)?,
        &timestep,
    )?;
    let v = prediction.to_dtype(DType::F32)?;
    let (diffusion_on, aux_on) = step_terms(aux);
    let diffusion = if diffusion_on {
        Some(weighted_velocity_loss(&v, &target, mask_weight, mae)?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let x0_hat = Parameterization::FlowNoiseMinusX0 {
                sigma: sigma as f32,
            }
            .recover_x0(&x_t, &v)?;
            a.aux_loss(&x0_hat)?
        }
        _ => None,
    };
    combine_terms(diffusion, aux_term)
}

impl Sd3Trainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let cfg = &req.config;
        let device = &self.device;
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

        let pipe = Pipeline::load(&self.root, device, self.dtype, self.variant, None, &[]);
        let mut encoders = pipe.load_training_encoders()?;
        let vae_encoder = pipe.load_vae_encoder()?;
        let model_cfg = self.variant.config();
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are off).
        let edges = bucket_edges(cfg);
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127) of `(x0, conditioning,
        // subject-mask loss weight)`; the weight (broadcast to that bucket's latent shape) is `None`
        // unless subject-masked loss is on (sc-24828).
        let mut cache: Vec<(Tensor, Sd3Conditioning, Option<Tensor>)> =
            Vec::with_capacity(req.items.len() * edges.len());
        for (index, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total,
            });
            let conditioning = aggregate(&model_cfg, &encoders.encode(&item.caption)?)?;
            // sc-24828: the item's subject mask is read + checked once, then resampled per bucket
            // onto that bucket's latent grid (`None` when masked loss is off).
            let mask =
                PreparedSubjectMask::load_if_enabled(LABEL, item, cfg.subject_mask_loss.as_ref())?;
            let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
            for &edge in &edges {
                let image = square_image_tensor(&square, edge, device)?;
                let x0 = encode_mean(&vae_encoder, &image, DType::F32)?;
                // `decode_square` centre-crops to a square, so the mask takes the same crop.
                let mask_weight = prepared_subject_mask_weight(
                    LABEL,
                    mask.as_ref(),
                    CropBox::center_square,
                    x0.dims(),
                    device,
                )?;
                cache.push((x0, conditioning.clone(), mask_weight));
            }
        }
        drop(vae_encoder);
        drop(encoders);
        if cache.is_empty() {
            return Err(if req.cancel.is_cancelled() {
                CandleError::Canceled
            } else {
                CandleError::Msg("sd3 trainer: no usable dataset items".into())
            });
        }

        let mut transformer = pipe.load_training_transformer()?;
        let suffixes = target_suffixes(req);
        let set = match cfg.network_type {
            NetworkType::Lora => build_adapt_lora_targets(
                &mut transformer,
                &suffixes,
                cfg.rank,
                cfg.alpha,
                cfg.seed,
                device,
            )?,
            NetworkType::Lokr => build_adapt_lokr_targets(
                &mut transformer,
                &suffixes,
                cfg.rank,
                cfg.alpha,
                cfg.decompose_factor,
                cfg.seed,
                device,
            )?,
        };
        let accum = cfg.gradient_accumulation.max(1);
        let mut opt = TrainOptimizer::from_config(
            &cfg.optimizer,
            set.vars.clone(),
            cfg.learning_rate,
            flow_match::effective_weight_decay(cfg),
        )?;
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let mut accumulated = None;
        let mut update_idx = 0;
        let mut last_loss = 0.0;
        let mut steps_run = 0;
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: references per (item, bucket) entry once (the cached latent is already
        // TAESD3's NCHW input), alternation keyed on the real item.
        let mut aux_driver = match perceptual {
            Some(path) => Some(AuxDriver::prepare(
                path,
                cache.len(),
                |i| Ok(cache[i].0.clone()),
                &schedule,
                accum,
                0,
            )?),
            None => None,
        };
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let sample = step_sample(aux_driver.as_mut(), step, &schedule);
            let (x0, conditioning, mask_weight) = &cache[sample.entry];
            let noise = flow_match::sample_noise(
                x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                device,
            )?;
            let (sigma, aux) = plan_sigma(&sample, sample_sigma(req, step))?;
            let (loss, losses) = step_loss(
                &transformer,
                x0,
                conditioning,
                &noise,
                sigma,
                self.dtype,
                flow_match::is_mae(cfg),
                mask_weight.as_ref(),
                aux.as_ref(),
            )?;
            last_loss = losses.total;
            let grads = loss.backward()?;
            accumulate_grads(&mut accumulated, grads, &set.vars)?;
            if step.is_multiple_of(accum) {
                flow_match::apply_update(
                    &mut opt,
                    &mut accumulated,
                    &set,
                    accum,
                    cfg,
                    update_idx,
                    total_updates,
                    warmup_updates,
                    cfg.seed,
                )?;
                update_idx += 1;
            }
            steps_run = step;
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });
            if cfg.save_every > 0 && step.is_multiple_of(cfg.save_every) && step != cfg.steps {
                flow_match::create_output_dir(&req.output_dir)?;
                let name = format!(
                    "{}-step{step:06}.safetensors",
                    candle_gen::train::checkpoint::file_stem(&req.file_name)
                );
                flow_match::save_adapter(&set, &HashMap::new(), &req.output_dir.join(name))?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }
        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }
        if accumulated.is_some() {
            let window = steps_run % accum;
            flow_match::apply_update(
                &mut opt,
                &mut accumulated,
                &set,
                if window == 0 { accum } else { window },
                cfg,
                update_idx,
                total_updates,
                warmup_updates,
                cfg.seed,
            )?;
        }
        on_progress(TrainingProgress::Saving);
        flow_match::create_output_dir(&req.output_dir)?;
        let path = req.output_dir.join(&req.file_name);
        flow_match::save_adapter(&set, &HashMap::new(), &path)?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Sd3Config;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem};
    use candle_gen::gen_core::Quant;
    use candle_gen::train::optim::TrainOptimizer;

    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem::captioned(
                "/image.png".into(),
                "caption".into(),
            )],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        }
    }

    fn trainer(dtype: DType) -> Sd3Trainer {
        Sd3Trainer {
            descriptor: large_descriptor(),
            root: "/unused".into(),
            device: Device::Cpu,
            dtype,
            variant: Variant::Large,
        }
    }

    fn tiny_cfg() -> Sd3Config {
        Sd3Config {
            in_channels: 16,
            patch_size: 2,
            pos_embed_max_size: 8,
            inner_dim: 16,
            num_heads: 2,
            head_dim: 8,
            num_layers: 2,
            mlp_ratio: 2.0,
            qk_norm: true,
            context_pre_only_last: true,
            pooled_dim: 12,
            joint_attention_dim: 20,
            clip_l_dim: 4,
            clip_g_dim: 8,
            clip_concat_dim: 12,
            clip_seq_len: 3,
            t5_seq_len: 2,
            t5_dim: 20,
            timestep_channels: 16,
            dual_attention_layers: vec![0],
        }
    }

    #[test]
    fn descriptors_cover_large_and_medium_without_turbo() {
        for (descriptor, id) in [
            (large_descriptor(), MODEL_ID),
            (medium_descriptor(), MODEL_ID_MEDIUM),
        ] {
            assert_eq!(descriptor.id, id);
            assert_eq!(descriptor.backend, "candle");
            assert!(descriptor.supports_lora && descriptor.supports_lokr);
            assert!(!descriptor.supports_control && !descriptor.supports_full_finetune);
            assert!(descriptor.techniques.resolution_buckets, "sc-2127");
        }
    }

    /// sc-2127: one installed adapter set trains on latents of two bucket sizes through the real
    /// MMDiT (the positional table is cropped per forward, so no size is baked in at load).
    #[test]
    fn one_adapter_set_trains_on_two_bucket_sizes() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut transformer = Sd3Transformer::new(&cfg, vb).unwrap();
        let targets = vec!["attn2.to_out.0".to_string()];
        let set = build_adapt_lora_targets(&mut transformer, &targets, 2, 2.0, 7, &dev).unwrap();
        let context = Tensor::randn(
            0f32,
            1f32,
            (1, cfg.context_seq_len(), cfg.joint_attention_dim),
            &dev,
        )
        .unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap();
        let timestep = Tensor::full(500f32, 1, &dev).unwrap();
        for side in [8usize, 16] {
            let latent = Tensor::randn(0f32, 1f32, (1, 16, side, side), &dev).unwrap();
            let prediction = transformer
                .forward(&latent, &context, &pooled, &timestep)
                .unwrap();
            assert_eq!(prediction.dims(), latent.dims(), "side {side}");
            let grads = prediction
                .sqr()
                .unwrap()
                .mean_all()
                .unwrap()
                .backward()
                .unwrap();
            assert!(
                grads.get(set.vars[1].as_tensor()).is_some(),
                "side {side}: the bucket's step must reach the shared adapter"
            );
        }
    }

    #[test]
    fn validate_rejects_checkpointing_item_conditioning_and_dtype_mismatch() {
        let bf16 = trainer(DType::BF16);
        assert!(bf16.validate(&request()).is_ok());

        let mut checkpointed = request();
        checkpointed.config.gradient_checkpointing = true;
        assert!(bf16.validate(&checkpointed).is_err());

        let mut conditioned = request();
        conditioned.items[0].control_image_path = Some("/control.png".into());
        assert!(bf16.validate(&conditioned).is_err());

        let mut f32_request = request();
        f32_request.config.train_dtype = "f32".into();
        assert!(bf16.validate(&f32_request).is_err());
        assert!(trainer(DType::F32).validate(&request()).is_err());
        assert!(trainer(DType::F32).validate(&f32_request).is_ok());
    }

    #[test]
    fn load_rejects_explicit_and_physical_packed_transformer() {
        let root = tempfile::tempdir().unwrap();
        let transformer = root.path().join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            r#"{"quantization":{"group_size":64,"bits":8}}"#,
        )
        .unwrap();
        let physical = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        assert!(load_for(&physical, Variant::Large)
            .err()
            .expect("physical packed tier must be rejected")
            .to_string()
            .contains("dense"));

        let plain = tempfile::tempdir().unwrap();
        let mut explicit = LoadSpec::new(WeightsSource::Dir(plain.path().into()));
        explicit.quantize = Some(Quant::Q4);
        assert!(load_for(&explicit, Variant::Medium)
            .err()
            .expect("explicit packed tier must be rejected")
            .to_string()
            .contains("dense"));
    }

    #[test]
    fn scaled_timestep_and_raw_velocity_math_match_sd3_contract() {
        let dev = Device::Cpu;
        let x0 = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target) = flow_match::build_batch(&x0, &noise, 0.25).unwrap();
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        assert_eq!(0.25f32 * 1000.0, 250.0);
    }

    /// sc-24828: the subject-mask weight reaches the trainer's loss. An all-ones map is the unweighted
    /// loss; an all-zero map zeroes the loss AND every adapter gradient; a half map lands between.
    #[test]
    fn subject_mask_weight_reaches_the_step_loss() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut transformer = Sd3Transformer::new(&cfg, vb).unwrap();
        let targets = vec!["attn2.to_out.0".to_string()];
        let set = build_adapt_lora_targets(&mut transformer, &targets, 2, 2.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let shape = [1usize, 16, 4, 4];
        let x0 = Tensor::randn(0f32, 1f32, &shape, &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, &shape, &dev).unwrap();
        let conditioning = Sd3Conditioning {
            context: Tensor::randn(
                0f32,
                1f32,
                (1, cfg.context_seq_len(), cfg.joint_attention_dim),
                &dev,
            )
            .unwrap(),
            pooled: Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap(),
        };
        let map = |w: &[f32]| flow_match::subject_mask_weight(w, 4, 4, &shape, &dev).unwrap();
        let run = |weight: Option<&Tensor>| {
            step_loss(
                &transformer,
                &x0,
                &conditioning,
                &noise,
                0.5,
                DType::F32,
                false,
                weight,
                None,
            )
            .unwrap()
            .0
        };
        let plain = run(None).to_scalar::<f32>().unwrap();
        let ones = run(Some(&map(&[1.0; 16]))).to_scalar::<f32>().unwrap();
        assert!((ones - plain).abs() < 1e-6);
        let zero = run(Some(&map(&[0.0; 16])));
        assert_eq!(zero.to_scalar::<f32>().unwrap(), 0.0);
        let grads = zero.backward().unwrap();
        for v in &set.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(g.iter().all(|x| *x == 0.0), "nonzero adapter grad");
            }
        }
        let half: Vec<f32> = (0..16).map(|i| if i % 4 < 2 { 1.0 } else { 0.0 }).collect();
        let half = run(Some(&map(&half))).to_scalar::<f32>().unwrap();
        assert!(half > 0.0 && half < plain, "{half} vs {plain}");
    }

    #[test]
    fn descriptors_declare_subject_mask_loss() {
        assert!(large_descriptor().techniques.subject_mask_loss);
        assert!(medium_descriptor().techniques.subject_mask_loss);
    }

    #[test]
    fn backward_reaches_real_mmdit_lora_residuals() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut transformer = Sd3Transformer::new(&cfg, vb).unwrap();
        // Target the second attention's output projection: it is downstream of SDPA, so this test
        // exercises the MMDiT-X route without depending on the attention kernel's q/k backward.
        let targets = vec!["attn2.to_out.0".to_string()];
        let set = build_adapt_lora_targets(&mut transformer, &targets, 2, 2.0, 7, &dev).unwrap();
        let latent = Tensor::randn(0f32, 1f32, (1, 16, 8, 8), &dev).unwrap();
        let context = Tensor::randn(
            0f32,
            1f32,
            (1, cfg.context_seq_len(), cfg.joint_attention_dim),
            &dev,
        )
        .unwrap();
        let pooled = Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap();
        let timestep = Tensor::full(500f32, 1, &dev).unwrap();
        let loss1 = transformer
            .forward(&latent, &context, &pooled, &timestep)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let grads1 = loss1.backward().unwrap();
        assert!(
            grads1.get(set.vars[1].as_tensor()).is_some(),
            "the real MMDiT-X path must reach zero-initialized B"
        );
        let mut optimizer =
            TrainOptimizer::from_config("adam", set.vars.clone(), 0.05, 0.0).unwrap();
        optimizer.step(&grads1).unwrap();
        let loss2 = transformer
            .forward(&latent, &context, &pooled, &timestep)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let grads2 = loss2.backward().unwrap();
        assert!(
            grads2.get(set.vars[0].as_tensor()).is_some(),
            "the real MMDiT-X path must reach A after B's optimizer step"
        );
        assert_eq!(
            set.vars.len(),
            2,
            "the one attn2 projection has A/B factors"
        );
    }

    /// Epic 2123 depth anchoring (sc-24830) on the candle SD3.5 trainer: the tiny MMDiT + a
    /// random-init tiny 16-channel decoder + tiny DA2, through the real `step_loss`.
    mod depth_anchoring {
        use super::*;
        use candle_gen::candle_core::backprop::GradStore;
        use candle_gen::candle_core::Var;
        use candle_gen::gen_core::train::{AuxLossSchedule, DepthModelSize};

        fn schedule() -> AuxLossSchedule {
            AuxLossSchedule {
                weight: 0.5,
                t_min: 0.6,
                t_max: 0.9,
                every_n: 2,
            }
        }

        struct Fixture {
            transformer: Sd3Transformer,
            vars: Vec<Var>,
            x0: Tensor,
            conditioning: Sd3Conditioning,
            noise: Tensor,
        }

        fn fixture() -> Fixture {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let mut transformer = Sd3Transformer::new(&cfg, vb).unwrap();
            // Downstream of SDPA (as the other SD3 tests): exercises the MMDiT-X route.
            let targets = vec!["attn2.to_out.0".to_string()];
            let set =
                build_adapt_lora_targets(&mut transformer, &targets, 2, 2.0, 7, &dev).unwrap();
            for v in &set.vars {
                v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                    .unwrap();
            }
            let shape = [1usize, 16, 4, 4];
            Fixture {
                transformer,
                vars: set.vars,
                x0: Tensor::randn(0f32, 1f32, &shape, &dev).unwrap(),
                conditioning: Sd3Conditioning {
                    context: Tensor::randn(
                        0f32,
                        1f32,
                        (1, cfg.context_seq_len(), cfg.joint_attention_dim),
                        &dev,
                    )
                    .unwrap(),
                    pooled: Tensor::randn(0f32, 1f32, (1, cfg.pooled_dim), &dev).unwrap(),
                },
                noise: Tensor::randn(0f32, 1f32, &shape, &dev).unwrap(),
            }
        }

        /// Step 1 = diffusion step, step 2 = the item's 2nd window ⇒ aux-only.
        fn driver(f: &Fixture) -> (AuxDriver, BucketSchedule) {
            let path =
                candle_gen_perceptual::testing::tiny_depth_path(16, schedule(), &Device::Cpu)
                    .unwrap();
            let sched = BucketSchedule::new(1, &[], 3);
            let x0 = f.x0.clone();
            let d = AuxDriver::prepare(path, 1, |_| Ok(x0.clone()), &sched, 1, 0).unwrap();
            (d, sched)
        }

        fn run(f: &Fixture, sigma: f64, aux: Option<&AuxStep<'_>>) -> (StepLosses, GradStore) {
            let (loss, losses) = step_loss(
                &f.transformer,
                &f.x0,
                &f.conditioning,
                &f.noise,
                sigma,
                DType::F32,
                false,
                None,
                aux,
            )
            .unwrap();
            (losses, loss.backward().unwrap())
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

        /// (a)/(b): the diffusion step has no aux term; the aux-only step trains at the remapped
        /// σ with no diffusion term, total == aux and a nonzero finite LoRA B gradient. Mutations:
        /// compute the diffusion term unconditionally ⇒ red; keep the sampled σ in `plan_sigma` on
        /// an aux step ⇒ red.
        #[test]
        fn aux_only_step_trains_the_adapter_through_depth_alone() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let (sig1, s1) = plan_sigma(&d.sample(1, &sched), 0.3).unwrap();
            let s1 = s1.unwrap();
            assert_eq!(sig1, 0.3);
            assert!(s1.diffusion() && !s1.has_aux());
            let (l, _) = run(&f, sig1, Some(&s1));
            assert!(l.diffusion.is_some() && l.aux.is_none(), "{l:?}");
            assert_eq!(l.total, l.diffusion.unwrap());
            let (sig2, s2) = plan_sigma(&d.sample(2, &sched), 0.3).unwrap();
            let s2 = s2.unwrap();
            assert!(!s2.diffusion() && s2.has_aux());
            assert!(
                (0.6..=0.9).contains(&sig2),
                "remapped into the window: {sig2}"
            );
            let (l, g) = run(&f, sig2, Some(&s2));
            assert!(l.diffusion.is_none(), "{l:?}");
            let aux = l.aux.expect("aux term");
            assert!(aux.is_finite() && aux > 0.0, "{aux}");
            assert_eq!(l.total, aux);
            let gv: Vec<f32> = grad_bits(&g, &f.vars[1])
                .into_iter()
                .map(f32::from_bits)
                .collect();
            assert!(gv.iter().all(|x| x.is_finite()));
            assert!(
                gv.iter().any(|x| *x != 0.0),
                "no LoRA B gradient from the depth term"
            );
            assert_eq!(plan_sigma(&StepSample::plain(0, 0), 0.3).unwrap().0, 0.3);
        }

        /// The aux term is the depth loss of THE trainer's x0 estimate `x_t − σ·v`, recomputed
        /// independently. Mutation: `FlowX0MinusNoise` in `step_loss` ⇒ red.
        #[test]
        fn aux_term_is_the_depth_loss_of_the_recovered_x0() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let _ = d.sample(1, &sched);
            let s2 = d.sample(2, &sched).plan(0.3).unwrap().unwrap();
            let sigma = s2.noise_level() as f64;
            let (l, _) = run(&f, sigma, Some(&s2));
            let (x_t, _) = flow_match::build_batch(&f.x0, &f.noise, sigma).unwrap();
            let t = Tensor::new(&[(sigma * 1000.0) as f32], &Device::Cpu).unwrap();
            let v = f
                .transformer
                .forward(&x_t, &f.conditioning.context, &f.conditioning.pooled, &t)
                .unwrap();
            let x0 = (&x_t - (v * sigma as f32 as f64).unwrap()).unwrap();
            let want = s2
                .aux_loss(&x0)
                .unwrap()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(l.aux.unwrap().to_bits(), want.to_bits());
        }

        /// (c) Depth off is bit-identical to the pre-epic-2123 step (reproduced verbatim).
        /// Mutation: perturb the `None` combine ⇒ red.
        #[test]
        fn depth_off_is_bit_identical_to_the_legacy_step() {
            let f = fixture();
            let (l, g) = run(&f, 0.5, None);
            assert_eq!((l.diffusion, l.aux), (Some(l.total), None));
            let (x_t, target) = flow_match::build_batch(&f.x0, &f.noise, 0.5).unwrap();
            let t = Tensor::new(&[500f32], &Device::Cpu).unwrap();
            let prediction = f
                .transformer
                .forward(&x_t, &f.conditioning.context, &f.conditioning.pooled, &t)
                .unwrap();
            let loss = weighted_velocity_loss(&prediction, &target, None, false).unwrap();
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
        /// base + aux (SD3 has only the dense path; checkpointing is refused by `validate`).
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
            let tdir = tmp.path().join("transformer");
            std::fs::create_dir_all(&tdir).unwrap();
            std::fs::write(tdir.join("m.safetensors"), vec![0u8; 4096]).unwrap();
            let base = 4096u64;
            assert!(aux_memory_preflight(tmp.path(), &TrainingConfig::default(), 4, 1).is_ok());
            assert!(aux_memory_preflight(tmp.path(), &cfg, 4, base + small).is_ok());
            let e = aux_memory_preflight(tmp.path(), &cfg, 4, base + small - 1)
                .unwrap_err()
                .to_string();
            assert!(e.contains("perceptual"), "{e}");
        }

        /// (e) Both SD3.5 descriptors declare depth anchoring; depth off builds no path; a missing
        /// decoder dir names TAESD3. Mutation: `depth_anchoring: false` ⇒ red.
        #[test]
        fn descriptor_and_loader_errors() {
            assert!(large_descriptor().techniques.depth_anchoring);
            assert!(medium_descriptor().techniques.depth_anchoring);
            let dev = Device::Cpu;
            assert!(load_perceptual_path(&TrainingConfig::default(), &dev)
                .unwrap()
                .is_none());
            let tmp = tempfile::tempdir().unwrap();
            let mut cfg = depth_on();
            cfg.depth_anchoring.model_dir = Some(tmp.path().join("da2"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAESD3"), "{e}");
            cfg.perceptual_decoder_dir = Some(tmp.path().join("no-taesd3"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAESD3"), "{e}");
        }
    }
}
