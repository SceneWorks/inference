//! Candle training for the undistilled Anima base. The trainer caches Qwen-VAE latents and Qwen3
//! states, then trains the live Cosmos DiT plus bundled `llm_adapter` conditioner through the same
//! `AdaptLinear` projections used by inference.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, NetworkType, Precision, WeightsSource};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{self, validate_flow_match_request, velocity_loss};
use candle_gen::train::lora::{build_adapt_lokr_targets, build_adapt_lora_targets, AdaptLoraHost};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::schedule::schedule_updates;
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
        let edge = bucket_resolution(cfg.resolution);
        let total = req.items.len() as u32;
        let mut cache = Vec::with_capacity(req.items.len());
        for (index, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total,
            });
            let image = load_image_tensor(&item.image_path, edge, device)?;
            let x0 = vae_encoder
                .encode(&image)?
                .unsqueeze(2)?
                .to_dtype(DType::F32)?;
            let (source, target_ids) = encode_conditioner_inputs(
                &tokenizers,
                &text_encoder,
                &item.caption,
                dtype,
                device,
            )?;
            cache.push((x0, source, target_ids));
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
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, source, target_ids) = &cache[(step as usize - 1) % cache.len()];
            let sigma = shifted_sigma(cfg, step);
            let noise = flow_match::sample_noise(
                x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                device,
            )?;
            let (x_t, target) = flow_match::build_batch(x0, &noise, sigma)?;
            let encoder = conditioner.forward(source, target_ids, dtype)?;
            let sigma_tensor = Tensor::new(&[sigma as f32], device)?.to_dtype(dtype)?;
            let prediction = dit.forward(&x_t.to_dtype(dtype)?, &sigma_tensor, &encoder, dtype)?;
            let loss = velocity_loss(
                &prediction.to_dtype(DType::F32)?,
                &target,
                flow_match::is_mae(cfg),
            )?;
            last_loss = loss.to_scalar::<f32>()?;
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
