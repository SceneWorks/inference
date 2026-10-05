//! Candle training for the undistilled Anima base. The trainer caches Qwen-VAE latents and Qwen3
//! states, then trains the live Cosmos DiT plus bundled `llm_adapter` conditioner through the same
//! `AdaptLinear` projections used by inference.

use std::collections::HashMap;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    self, BucketSchedule, LoadSpec, Modality, NetworkType, Precision, WeightsSource,
};
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor};
use candle_gen::train::flow_match::{
    self, combine_terms, prepared_subject_mask_weight, step_sample, step_terms,
    validate_flow_match_request, weighted_velocity_loss, AuxDriver, AuxStep, StepLosses,
};
use candle_gen::train::lora::{build_adapt_lokr_targets, build_adapt_lora_targets, AdaptLoraHost};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::perceptual::{Parameterization, PerceptualPath};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::train::taehv::TaehvConfig;
use candle_gen::{CandleError, Result};

use crate::adapt::AdaptLinear;
use crate::conditioner::AnimaTextConditioner;
use crate::config::Variant;
use crate::loader::{dit_is_packed, resolve_split_files, AnimaComponents, VAE_FILE};
use crate::text_encoder::AnimaQwen3;
use crate::tokenizer::AnimaTokenizers;
use crate::transformer::CosmosDiT;
use crate::vae::load_vae_encoder;

const LABEL: &str = "anima trainer";

pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: Variant::Base.id(),
        family: "anima",
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
        // sc-24830 (epic 2123): depth anchoring on its one (dense) loss path — the shared
        // decoded-x0 perceptual path through TAEW2.1, the Qwen-Image VAE's tiny decoder.
        // sc-24833 (epic 2123): the VAE anchor (same family decoder → FLUX.2 encoder taps, per
        // decoded frame) through the shared aux-loss builder this trainer already drives, wherever
        // depth anchoring is wired. No E-LatentLPIPS: no published weights match this latent
        // family.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            vae_anchor_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

pub struct AnimaTrainer {
    descriptor: TrainerDescriptor,
    source: WeightsSource,
    device: Device,
}

pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    if spec.precision != Precision::Bf16
        || spec.quantize.is_some()
        || dit_is_packed(&spec.weights, Variant::Base)?
    {
        return Err(CandleError::Msg(
            "anima trainer requires the dense bf16 base tier; packed/quantized weights are not trainable"
                .into(),
        ));
    }
    Ok(Box::new(AnimaTrainer {
        descriptor: trainer_descriptor(),
        source: spec.weights.clone(),
        device: candle_gen::default_device()?,
    }))
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl Trainer for AnimaTrainer {
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
        if req.config.resume {
            return Err(gen_core::Error::Unsupported(
                "anima candle trainer does not yet support resume".into(),
            ));
        }
        if req.config.gradient_checkpointing {
            return Err(gen_core::Error::Unsupported(
                "anima candle trainer does not yet support gradient checkpointing".into(),
            ));
        }
        if flow_match::parse_compute_dtype(&req.config.train_dtype) != DType::BF16 {
            return Err(gen_core::Error::Unsupported(format!(
                "anima candle trainer runs its dense base at bf16; train_dtype '{}' is unsupported",
                req.config.train_dtype
            )));
        }
        if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
            return Err(gen_core::Error::Unsupported(
                "anima candle trainer does not yet support in-training previews".into(),
            ));
        }
        if req
            .items
            .iter()
            .any(|item| item.control_image_path.is_some())
        {
            return Err(gen_core::Error::Unsupported(
                "anima candle trainer does not consume per-item control/source images".into(),
            ));
        }
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
        self.validate(req)?;
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

struct AnimaTrainHost<'a> {
    dit: &'a mut CosmosDiT,
    conditioner: &'a mut AnimaTextConditioner,
}

impl AdaptLoraHost for AnimaTrainHost<'_> {
    fn visit_adapt_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        self.dit.visit_adaptable_mut(f)?;
        self.conditioner.visit_adaptable_mut(f)
    }
}

fn encode_conditioner_inputs(
    tokenizers: &AnimaTokenizers,
    text_encoder: &AnimaQwen3,
    caption: &str,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let (ids, attention) = tokenizers.encode_qwen(caption)?;
    let input_ids = Tensor::from_vec(
        ids.into_iter().map(|id| id as u32).collect::<Vec<_>>(),
        (1, attention.len()),
        device,
    )?;
    let source = text_encoder.forward(&input_ids, dtype)?;
    let mask = Tensor::from_vec(
        attention.into_iter().map(|v| v as f32).collect::<Vec<_>>(),
        (1, source.dim(1)?, 1),
        device,
    )?
    .to_dtype(dtype)?;
    let source = source.broadcast_mul(&mask)?;
    let t5 = tokenizers.encode_t5(caption)?;
    let target_ids = Tensor::from_vec(
        t5.iter().map(|&id| id as u32).collect::<Vec<_>>(),
        (1, t5.len()),
        device,
    )?;
    Ok((source, target_ids))
}

/// One micro-step's forward+backward (epic 2123 E8): run the conditioner (→ encoder states) and the
/// DiT on `x_t = (1−σ)·x0 + σ·noise` at the shifted `sigma`, then the step's terms — the (subject-mask
/// weighted) velocity regression toward `noise − x0` when the diffusion term contributes, and on a
/// planned step with aux losses the weighted perceptual term on the x0 estimate `x_t − σ·v`, each
/// latent frame decoded by TAEW2.1. `aux = None` ⇒ exactly the pre-epic-2123 step.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    dit: &CosmosDiT,
    conditioner: &AnimaTextConditioner,
    x0: &Tensor,
    source: &Tensor,
    target_ids: &Tensor,
    sigma: f64,
    noise: &Tensor,
    mask_weight: Option<&Tensor>,
    mae: bool,
    dtype: DType,
    aux: Option<&AuxStep<'_>>,
) -> Result<(StepLosses, GradStore)> {
    let device = x0.device();
    let (x_t, target) = flow_match::build_batch(x0, noise, sigma)?;
    let encoder = conditioner.forward(source, target_ids, dtype)?;
    let sigma_tensor = Tensor::new(&[sigma as f32], device)?.to_dtype(dtype)?;
    let prediction = dit
        .forward(&x_t.to_dtype(dtype)?, &sigma_tensor, &encoder, dtype)?
        .to_dtype(DType::F32)?;
    let (diffusion_on, aux_on) = step_terms(aux);
    let diffusion = if diffusion_on {
        Some(weighted_velocity_loss(
            &prediction,
            &target,
            mask_weight,
            mae,
        )?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let x0_hat = Parameterization::FlowNoiseMinusX0 {
                sigma: sigma as f32,
            }
            .recover_x0(&x_t.to_dtype(DType::F32)?, &prediction)?;
            a.aux_loss(&latent_frames_nchw(&x0_hat)?)?
        }
        _ => None,
    };
    let (loss, losses) = combine_terms(diffusion, aux_term)?;
    let grads = loss.backward()?;
    Ok((losses, grads))
}

/// A cached `[B, C, T, h, w]` latent (`T = 1` for a still) as the decoder's NCHW batch
/// `[B·T, C, h, w]` — each latent frame decoded independently.
fn latent_frames_nchw(latent: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = latent.dims5()?;
    Ok(latent
        .permute((0, 2, 1, 3, 4))?
        .reshape((b * t, c, h, w))?
        .contiguous()?)
}

/// Anima's latent family for the shared aux-loss builder (epic 2123 E8): the Qwen-Image VAE
/// (`load_vae_encoder`, whose encode yields the per-channel `(μ − latents_mean)/latents_std`
/// latent the DiT predicts in), decoded by TAEW2.1 — upstream's TAEHV checkpoint for Qwen-Image,
/// which takes that normalized latent with no scale/shift. A still image is one `T = 1` clip.
fn anima_decoder() -> candle_gen_perceptual::DecoderSpec {
    candle_gen_perceptual::DecoderSpec::Taehv {
        name: "TAEW2.1",
        config: TaehvConfig::taew2_1(),
    }
}

/// The epic-2123 perceptual path through the shared builder: `None` when no aux loss is enabled.
fn load_perceptual_path(
    cfg: &candle_gen::gen_core::train::TrainingConfig,
    device: &Device,
) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(
        cfg,
        &candle_gen_perceptual::AuxLossContext {
            label: LABEL,
            decoder: anima_decoder(),
            device,
            latent_lpips: None,
        },
    )
}

/// Epic 2123 E7: refuse a depth job whose resident DiT (`base_bytes`, its on-disk weights — the lower
/// bound the aux models stack on) plus TAEW2.1 + the losses' frozen models at the largest bucket
/// (`entries` cached references) exceeds `budget_bytes`. No-op when no aux loss is enabled. The
/// trainer has one (dense) backward path.
fn check_perceptual_memory(
    cfg: &candle_gen::gen_core::train::TrainingConfig,
    entries: usize,
    base_bytes: u64,
    budget_bytes: u64,
) -> Result<()> {
    let edge = bucket_edges(cfg).iter().copied().max().unwrap_or(0);
    let aux = candle_gen_perceptual::perceptual_footprint(
        cfg,
        &anima_decoder(),
        candle_gen_perceptual::AuxGeometry::image(edge, entries),
    );
    if aux == 0 {
        return Ok(());
    }
    flow_match::check_aux_memory(LABEL, base_bytes, aux, budget_bytes)
}

fn shifted_sigma(cfg: &candle_gen::gen_core::train::TrainingConfig, step: u32) -> f64 {
    let sigma = flow_match::sample_unit_timestep(
        &cfg.timestep_type,
        &cfg.timestep_bias,
        flow_match::timestep_seed(cfg.seed, step),
    ) as f64;
    3.0 * sigma / (1.0 + 2.0 * sigma)
}

impl AnimaTrainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let cfg = &req.config;
        let device = &self.device;
        on_progress(TrainingProgress::Preparing);
        // Epic 2123 E7: the depth-anchoring models count against the device budget (no-op with
        // nothing enabled), BEFORE any load or caching.
        if candle_gen_perceptual::any_aux_loss(cfg) {
            let dit_path = resolve_split_files(&self.source)?
                .join("diffusion_models")
                .join(Variant::Base.dit_filename());
            let base = std::fs::metadata(&dit_path)
                .map_err(|e| {
                    CandleError::Msg(format!("{LABEL}: stat {}: {e}", dit_path.display()))
                })?
                .len();
            check_perceptual_memory(
                cfg,
                req.items.len() * bucket_edges(cfg).len(),
                base,
                flow_match::device_training_budget_bytes(device, LABEL),
            )?;
        }
        // Epic 2123 depth anchoring (sc-24830): the frozen TAEW2.1 + Depth-Anything-V2 load before
        // the caching pass, so a missing checkpoint fails fast.
        let perceptual = load_perceptual_path(cfg, device)?;
        on_progress(TrainingProgress::LoadingModel);

        let components = AnimaComponents::load(&self.source, Variant::Base, device, &[])?;
        let AnimaComponents {
            mut dit,
            mut conditioner,
            text_encoder,
            vae,
            tokenizers,
            dtype,
        } = components;
        drop(vae);
        let root = resolve_split_files(&self.source)?;
        let vae_encoder = load_vae_encoder(root.join(VAE_FILE), device)?;
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are off).
        let edges = bucket_edges(cfg);
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127).
        let mut cache = Vec::with_capacity(req.items.len() * edges.len());
        for (index, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total,
            });
            let (source, target_ids) = encode_conditioner_inputs(
                &tokenizers,
                &text_encoder,
                &item.caption,
                dtype,
                device,
            )?;
            // sc-24828: the item's subject mask is read + checked once, then resampled per bucket
            // onto that bucket's latent grid (`None` when masked loss is off).
            let mask =
                PreparedSubjectMask::load_if_enabled(LABEL, item, cfg.subject_mask_loss.as_ref())?;
            let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
            for &edge in &edges {
                let image = square_image_tensor(&square, edge, device)?;
                let x0 = vae_encoder
                    .encode(&image)?
                    .unsqueeze(2)?
                    .to_dtype(DType::F32)?;
                // `decode_square` centre-crops to a square, so the mask takes the same crop; the
                // weight is built on `x0`'s `[1, 16, 1, h, w]` shape (last two axes = latent H, W).
                let mask_weight = prepared_subject_mask_weight(
                    LABEL,
                    mask.as_ref(),
                    CropBox::center_square,
                    x0.dims(),
                    device,
                )?;
                cache.push((x0, source.clone(), target_ids.clone(), mask_weight));
            }
        }
        drop(vae_encoder);
        drop(text_encoder);
        if cache.is_empty() {
            return Err(if req.cancel.is_cancelled() {
                CandleError::Canceled
            } else {
                CandleError::Msg("anima trainer: no usable dataset items".into())
            });
        }

        let suffixes = cfg.lora_target_modules.clone();
        let set = {
            let mut host = AnimaTrainHost {
                dit: &mut dit,
                conditioner: &mut conditioner,
            };
            match cfg.network_type {
                NetworkType::Lora => build_adapt_lora_targets(
                    &mut host, &suffixes, cfg.rank, cfg.alpha, cfg.seed, device,
                )?,
                NetworkType::Lokr => build_adapt_lokr_targets(
                    &mut host,
                    &suffixes,
                    cfg.rank,
                    cfg.alpha,
                    cfg.decompose_factor,
                    cfg.seed,
                    device,
                )?,
            }
        };
        let accum = cfg.gradient_accumulation.max(1);
        let weight_decay = flow_match::effective_weight_decay(cfg);
        let mut opt = TrainOptimizer::from_config(
            &cfg.optimizer,
            set.vars.clone(),
            cfg.learning_rate,
            weight_decay,
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
        // Epic 2123 E8: references once per (item, bucket) entry; alternation keyed on the real item
        // (the trainer has no resume, so nothing to replay).
        let mut aux_driver = match perceptual {
            Some(path) => Some(AuxDriver::prepare(
                path,
                cache.len(),
                |i| latent_frames_nchw(&cache[i].0),
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
            let (x0, source, target_ids, mask_weight) = &cache[sample.entry];
            let mut sigma = shifted_sigma(cfg, step);
            // Epic 2123 E8: an aux-only step trains at the (shifted) σ remapped into the window.
            let aux = sample.plan(sigma as f32)?;
            if let Some(a) = &aux {
                sigma = a.noise_level() as f64;
            }
            let noise = flow_match::sample_noise(
                x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                device,
            )?;
            let (losses, grads) = compute_step_loss_grads(
                &dit,
                &conditioner,
                x0,
                source,
                target_ids,
                sigma,
                &noise,
                mask_weight.as_ref(),
                flow_match::is_mae(cfg),
                dtype,
                aux.as_ref(),
            )?;
            last_loss = losses.total;
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
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem};
    use candle_gen::gen_core::Quant;

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

    fn trainer() -> AnimaTrainer {
        AnimaTrainer {
            descriptor: trainer_descriptor(),
            source: WeightsSource::Dir("/unused".into()),
            device: Device::Cpu,
        }
    }

    #[test]
    fn base_descriptor_advertises_both_adapter_kinds() {
        let descriptor = trainer_descriptor();
        assert_eq!(descriptor.id, "anima_base");
        assert_eq!(descriptor.backend, "candle");
        assert!(descriptor.supports_lora && descriptor.supports_lokr);
        assert!(!descriptor.supports_control && !descriptor.supports_full_finetune);
        assert!(descriptor.techniques.subject_mask_loss);
        assert!(descriptor.techniques.resolution_buckets, "sc-2127");
    }

    #[test]
    fn validate_rejects_unimplemented_checkpointing_and_item_conditioning() {
        let trainer = trainer();
        let mut checkpointed = request();
        checkpointed.config.gradient_checkpointing = true;
        assert!(trainer.validate(&checkpointed).is_err());

        let mut conditioned = request();
        conditioned.items[0].control_image_path = Some("/control.png".into());
        assert!(trainer.validate(&conditioned).is_err());

        for dtype in ["f32", "unknown"] {
            let mut unsupported = request();
            unsupported.config.train_dtype = dtype.into();
            let error = trainer.validate(&unsupported).unwrap_err().to_string();
            assert!(error.contains("train_dtype"), "{dtype}: {error}");
        }
    }

    #[test]
    fn load_rejects_explicit_and_physical_packed_tiers() {
        let root = tempfile::tempdir().unwrap();
        let diffusion = root.path().join("diffusion_models");
        std::fs::create_dir_all(&diffusion).unwrap();
        let packed = HashMap::from([(
            "net.x_embedder.proj.1.scales".to_string(),
            Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
        )]);
        candle_gen::candle_core::safetensors::save(
            &packed,
            diffusion.join(Variant::Base.dit_filename()),
        )
        .unwrap();

        let physical = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        assert!(load_trainer(&physical)
            .err()
            .expect("physical packed tier must be rejected")
            .to_string()
            .contains("packed"));

        let mut explicit = physical;
        explicit.quantize = Some(Quant::Q8);
        assert!(load_trainer(&explicit)
            .err()
            .expect("explicit packed tier must be rejected")
            .to_string()
            .contains("packed"));
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the candle Anima step seam on a tiny synthetic Cosmos
/// DiT + conditioner (16-channel latent `[1, 16, 1, 4, 4]`, deterministic `splitmix_uniform` weights)
/// with a random-init tiny-width TAEW2.1 (16 latent channels) and a
/// random-init tiny Depth-Anything-V2, planned through the shared flow-match `AuxDriver` exactly as
/// `train_impl` plans. CPU.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use crate::config::{ConditionerConfig, DitConfig};
    use candle_gen::candle_core::Var;
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen::gen_core::train::{AuxLossSchedule, ResolutionBucket, TrainingConfig};
    use candle_gen::train::lora::LoraSet;
    use candle_gen::train::perceptual::AuxLoss;
    use candle_gen::train::taehv::{splitmix_uniform, synthetic_taehv_weights, TaehvDecoder};

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

    struct Fixture {
        dit: CosmosDiT,
        cond: AnimaTextConditioner,
        set: LoraSet,
        x0: Tensor,
        source: Tensor,
        ids: Tensor,
        noise: Tensor,
    }

    /// Deterministic synthetic weights in the checkpoint key layout (mirrors
    /// `tests/adapter_residuals.rs`): norm scales near 1, everything else `U·√(1/fan_in)`.
    fn put(map: &mut HashMap<String, Tensor>, key: &str, shape: &[usize]) {
        let seed = key.bytes().map(u64::from).sum::<u64>() % 997 + 3000;
        let fan_in = shape.get(1).copied().unwrap_or(1).max(1);
        let t = if key.ends_with("norm.weight") {
            splitmix_uniform(shape, seed, 0.05, 1.0, &Device::Cpu)
        } else {
            splitmix_uniform(shape, seed, (1.0 / fan_in as f64).sqrt(), 0.0, &Device::Cpu)
        };
        map.insert(key.to_string(), t.unwrap());
    }

    fn dit_cfg() -> DitConfig {
        DitConfig {
            in_channels: 16,
            out_channels: 16,
            num_attention_heads: 2,
            attention_head_dim: 6,
            num_layers: 2,
            mlp_ratio: 4.0,
            text_embed_dim: 8,
            adaln_lora_dim: 8,
            max_size: (128, 120, 120),
            patch_size: (1, 2, 2),
            rope_scale: (1.0, 4.0, 4.0),
            concat_padding_mask: true,
        }
    }

    fn cond_cfg() -> ConditionerConfig {
        ConditionerConfig {
            source_dim: 8,
            target_dim: 8,
            model_dim: 8,
            num_layers: 1,
            num_attention_heads: 2,
            mlp_ratio: 4.0,
            target_vocab_size: 16,
            min_sequence_length: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }

    fn dit_map(cfg: &DitConfig) -> HashMap<String, Tensor> {
        let h = cfg.hidden_size();
        let hd = cfg.attention_head_dim;
        let lora = cfg.adaln_lora_dim;
        let ctx = cfg.text_embed_dim;
        let ff = (cfg.mlp_ratio * h as f32) as usize;
        let (pt, ph, pw) = cfg.patch_size;
        let patch_in = cfg.patch_in_channels() * pt * ph * pw;
        let proj_out = ph * pw * pt * cfg.out_channels;
        let mut w = HashMap::new();
        put(&mut w, "net.x_embedder.proj.1.weight", &[h, patch_in]);
        put(&mut w, "net.t_embedder.1.linear_1.weight", &[3 * h, h]);
        put(&mut w, "net.t_embedder.1.linear_2.weight", &[3 * h, 3 * h]);
        put(&mut w, "net.t_embedding_norm.weight", &[h]);
        for i in 0..cfg.num_layers {
            let b = format!("net.blocks.{i}");
            for m in [
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ] {
                put(&mut w, &format!("{b}.{m}.1.weight"), &[lora, h]);
                put(&mut w, &format!("{b}.{m}.2.weight"), &[3 * h, lora]);
            }
            for (attn, kv_in) in [("self_attn", h), ("cross_attn", ctx)] {
                put(&mut w, &format!("{b}.{attn}.q_proj.weight"), &[h, h]);
                put(&mut w, &format!("{b}.{attn}.k_proj.weight"), &[h, kv_in]);
                put(&mut w, &format!("{b}.{attn}.v_proj.weight"), &[h, kv_in]);
                put(&mut w, &format!("{b}.{attn}.output_proj.weight"), &[h, h]);
                put(&mut w, &format!("{b}.{attn}.q_norm.weight"), &[hd]);
                put(&mut w, &format!("{b}.{attn}.k_norm.weight"), &[hd]);
            }
            put(&mut w, &format!("{b}.mlp.layer1.weight"), &[ff, h]);
            put(&mut w, &format!("{b}.mlp.layer2.weight"), &[h, ff]);
        }
        put(
            &mut w,
            "net.final_layer.adaln_modulation.1.weight",
            &[lora, h],
        );
        put(
            &mut w,
            "net.final_layer.adaln_modulation.2.weight",
            &[2 * h, lora],
        );
        put(&mut w, "net.final_layer.linear.weight", &[proj_out, h]);
        w
    }

    fn cond_map(cfg: &ConditionerConfig) -> HashMap<String, Tensor> {
        let d = cfg.model_dim;
        let hd = cfg.head_dim();
        let ff = (cfg.mlp_ratio * d as f32) as usize;
        let mut w = HashMap::new();
        put(
            &mut w,
            "llm_adapter.embed.weight",
            &[cfg.target_vocab_size, d],
        );
        for i in 0..cfg.num_layers {
            let b = format!("llm_adapter.blocks.{i}");
            put(&mut w, &format!("{b}.norm_self_attn.weight"), &[d]);
            put(&mut w, &format!("{b}.norm_cross_attn.weight"), &[d]);
            put(&mut w, &format!("{b}.norm_mlp.weight"), &[d]);
            for attn in ["self_attn", "cross_attn"] {
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    put(&mut w, &format!("{b}.{attn}.{p}.weight"), &[d, d]);
                }
                put(&mut w, &format!("{b}.{attn}.q_norm.weight"), &[hd]);
                put(&mut w, &format!("{b}.{attn}.k_norm.weight"), &[hd]);
            }
            put(&mut w, &format!("{b}.mlp.0.weight"), &[ff, d]);
            put(&mut w, &format!("{b}.mlp.0.bias"), &[ff]);
            put(&mut w, &format!("{b}.mlp.2.weight"), &[d, ff]);
            put(&mut w, &format!("{b}.mlp.2.bias"), &[d]);
        }
        put(&mut w, "llm_adapter.out_proj.weight", &[cfg.target_dim, d]);
        put(&mut w, "llm_adapter.out_proj.bias", &[cfg.target_dim]);
        put(&mut w, "llm_adapter.norm.weight", &[cfg.target_dim]);
        w
    }

    fn fixture() -> Fixture {
        let dev = Device::Cpu;
        let vb = VarBuilder::from_tensors(dit_map(&dit_cfg()), DType::F32, &dev);
        let mut dit = CosmosDiT::new(&vb.pp("net"), dit_cfg()).unwrap();
        let vb = VarBuilder::from_tensors(cond_map(&cond_cfg()), DType::F32, &dev);
        let mut cond = AnimaTextConditioner::new(&vb.pp("llm_adapter"), cond_cfg()).unwrap();
        let set = {
            let mut host = AnimaTrainHost {
                dit: &mut dit,
                conditioner: &mut cond,
            };
            build_adapt_lora_targets(&mut host, &[], 4, 4.0, 7, &dev).unwrap()
        };
        assert!(!set.vars.is_empty());
        let x0 = splitmix_uniform(&[1, 16, 1, 4, 4], 1, 1.0, 0.0, &dev).unwrap();
        let noise = splitmix_uniform(&[1, 16, 1, 4, 4], 2, 1.0, 0.0, &dev).unwrap();
        let source = splitmix_uniform(&[1, 6, 8], 3, 1.0, 0.0, &dev).unwrap();
        let ids = Tensor::from_vec(vec![1u32, 5, 9, 3], (1, 4), &dev).unwrap();
        Fixture {
            dit,
            cond,
            set,
            x0,
            source,
            ids,
            noise,
        }
    }

    fn step(f: &Fixture, sigma: f64, aux: Option<&AuxStep<'_>>) -> (StepLosses, GradStore) {
        compute_step_loss_grads(
            &f.dit,
            &f.cond,
            &f.x0,
            &f.source,
            &f.ids,
            sigma,
            &f.noise,
            None,
            false,
            DType::F32,
            aux,
        )
        .unwrap()
    }

    fn driver(f: &Fixture) -> (AuxDriver, BucketSchedule) {
        let sched = BucketSchedule::new(
            1,
            &[ResolutionBucket {
                resolution: 32,
                repeats: 1,
            }],
            7,
        );
        let d = AuxDriver::prepare(path(), 1, |_| latent_frames_nchw(&f.x0), &sched, 1, 0).unwrap();
        (d, sched)
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

    /// AC (a)+(b): the driver's step 2 is a depth step — no diffusion term, total == the weighted
    /// depth term, a nonzero finite gradient through the decoded x0 on the adapter factors — and
    /// step 1 carries no depth term. Mutation: compute the diffusion term unconditionally ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only() {
        let f = fixture();
        let (mut d, sched) = driver(&f);
        let s1 = d.sample(1, &sched).plan(0.5).unwrap().unwrap();
        let (diff, _) = step(&f, 0.5, Some(&s1));
        assert_eq!(diff.aux, None);
        assert_eq!(Some(diff.total), diff.diffusion);
        let s2 = d.sample(2, &sched).plan(0.5).unwrap().unwrap();
        assert!(!s2.diffusion());
        let (depth, g) = step(&f, s2.noise_level() as f64, Some(&s2));
        assert_eq!(depth.diffusion, None);
        let a = depth.aux.expect("depth term");
        assert!(a > 0.0 && a.is_finite(), "{a}");
        assert!((depth.total - a).abs() <= 1e-6 * a.abs());
        let gsum: f32 = f
            .set
            .vars
            .iter()
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
        assert!(gsum > 0.0 && gsum.is_finite(), "adapter grad {gsum}");
    }

    /// AC (c): depth off ⇒ bit-identical to the pre-epic-2123 loop body (reproduced here), and a
    /// diffusion-only planned step equals it too. Mutation: scale the diffusion loss (×1.0001) ⇒ red.
    #[test]
    fn depth_off_is_bit_identical_to_the_legacy_step() {
        assert!(
            load_perceptual_path(&TrainingConfig::default(), &Device::Cpu)
                .unwrap()
                .is_none()
        );
        let f = fixture();
        for (i, v) in f.set.vars.iter().enumerate() {
            v.set(
                &splitmix_uniform(
                    v.as_tensor().dims(),
                    900 + i as u64,
                    0.02,
                    0.0,
                    &Device::Cpu,
                )
                .unwrap(),
            )
            .unwrap();
        }
        let (off, g_off) = step(&f, 0.5, None);
        let (x_t, target) = flow_match::build_batch(&f.x0, &f.noise, 0.5).unwrap();
        let encoder = f.cond.forward(&f.source, &f.ids, DType::F32).unwrap();
        let sigma_tensor = Tensor::new(&[0.5f32], &Device::Cpu).unwrap();
        let prediction = f
            .dit
            .forward(&x_t, &sigma_tensor, &encoder, DType::F32)
            .unwrap();
        let loss = weighted_velocity_loss(
            &prediction.to_dtype(DType::F32).unwrap(),
            &target,
            None,
            false,
        )
        .unwrap();
        let legacy = loss.to_scalar::<f32>().unwrap();
        let g_legacy = loss.backward().unwrap();
        assert_eq!(off.total.to_bits(), legacy.to_bits());
        assert_eq!(
            grad_bits(&g_off, &f.set.vars),
            grad_bits(&g_legacy, &f.set.vars)
        );
        let (mut d, sched) = driver(&f);
        let s1 = d.sample(1, &sched).plan(0.5).unwrap().unwrap();
        let (on, g_on) = step(&f, 0.5, Some(&s1));
        assert_eq!(on, off);
        assert_eq!(
            grad_bits(&g_on, &f.set.vars),
            grad_bits(&g_off, &f.set.vars)
        );
    }

    /// AC (d), E7: depth grows the guarded footprint (more for Large) and the guard refuses at a
    /// synthetic budget between base and base+aux. Mutation: compare `base` alone ⇒ red.
    #[test]
    fn memory_guard_counts_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule();
        let fp = |c: &TrainingConfig| {
            candle_gen_perceptual::perceptual_footprint(
                c,
                &anima_decoder(),
                candle_gen_perceptual::AuxGeometry::image(1024, 1),
            )
        };
        let small = fp(&on);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = fp(&on);
        assert!(small > 0 && large > small + (1u64 << 30), "{small} {large}");
        let base = 4u64 << 30;
        assert!(check_perceptual_memory(&TrainingConfig::default(), 1, base, base).is_ok());
        assert!(check_perceptual_memory(&on, 1, base, base + large / 2).is_err());
        assert!(check_perceptual_memory(&on, 1, base, base + large + (1 << 30)).is_ok());
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
