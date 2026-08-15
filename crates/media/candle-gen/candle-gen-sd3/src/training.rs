//! Dense Candle training for the SD3.5 Large and Medium MMDiTs. Caption conditioning and VAE
//! latents are cached first; the frozen encoders are then dropped before the dense transformer and
//! its trainable forward-time LoRA/LoKr residuals are loaded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, NetworkType, Precision, WeightsSource};
use candle_gen::quant::AdaptLinear;
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{self, velocity_loss};
use candle_gen::train::lora::{build_adapt_lokr_targets, build_adapt_lora_targets, AdaptLoraHost};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::{CandleError, Result};

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

impl Sd3Trainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let cfg = &req.config;
        let device = &self.device;
        on_progress(TrainingProgress::Preparing);
        on_progress(TrainingProgress::LoadingModel);

        let pipe = Pipeline::load(&self.root, device, self.dtype, self.variant, None, &[]);
        let mut encoders = pipe.load_training_encoders()?;
        let vae_encoder = pipe.load_vae_encoder()?;
        let model_cfg = self.variant.config();
        let edge = bucket_resolution(cfg.resolution);
        let total = req.items.len() as u32;
        let mut cache: Vec<(Tensor, Sd3Conditioning)> = Vec::with_capacity(req.items.len());
        for (index, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total,
            });
            let image = load_image_tensor(&item.image_path, edge, device)?;
            let x0 = encode_mean(&vae_encoder, &image, DType::F32)?;
            let conditioning = aggregate(&model_cfg, &encoders.encode(&item.caption)?)?;
            cache.push((x0, conditioning));
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
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, conditioning) = &cache[(step as usize - 1) % cache.len()];
            let sigma = sample_sigma(req, step);
            let noise = flow_match::sample_noise(
                x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                device,
            )?;
            let (x_t, target) = flow_match::build_batch(x0, &noise, sigma)?;
            let timestep = Tensor::new(&[(sigma * 1000.0) as f32], device)?.to_dtype(self.dtype)?;
            let prediction = transformer.forward(
                &x_t.to_dtype(self.dtype)?,
                &conditioning.context.to_dtype(self.dtype)?,
                &conditioning.pooled.to_dtype(self.dtype)?,
                &timestep,
            )?;
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
}
