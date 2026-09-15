//! Candle LoRA/LoKr training for Kolors. Kolors shares the SDXL U-Net adapter surface, but its
//! conditioning is deliberately not the SDXL dual-CLIP path: captions are encoded by the snapshot's
//! ChatGLM3 tokenizer/model, projected from 4096 to 2048, and paired with Kolors' 5632-wide pooled
//! plus size embedding.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{linear, Module};
use candle_gen::diffusion_schedule::{
    KOLORS_BETA_END as BETA_END, KOLORS_BETA_START as BETA_START,
    KOLORS_TRAIN_STEPS as NUM_TRAIN_TIMESTEPS,
};
use candle_gen::gen_core::sampling::AlphaSchedule;
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, NetworkType, Precision, WeightsSource};
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{effective_weight_decay, noise_seed, sample_noise};
use candle_gen::train::lora::{
    build_lokr_targets, build_lora_targets, save_lokr, save_lora_peft, AdapterKind, LoraSet,
    SDXL_ATTN_TARGETS, SDXL_PEFT_PREFIX,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::{CandleError, Result};
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
    }
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
        validate_request(req).map_err(Into::into)
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

fn accumulation_divisor(micro_step: u32, configured: u32) -> u32 {
    let configured = configured.max(1);
    let pending = micro_step % configured;
    if pending == 0 {
        configured
    } else {
        pending
    }
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

fn epsilon_loss(prediction: &Tensor, noise: &Tensor, mae: bool) -> Result<Tensor> {
    let diff = (prediction.to_dtype(DType::F32)? - noise.to_dtype(DType::F32)?)?;
    Ok(if mae {
        diff.abs()?.mean_all()?
    } else {
        diff.sqr()?.mean_all()?
    })
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
        let edge = bucket_resolution(cfg.resolution);
        on_progress(TrainingProgress::Preparing);
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
        let mut cache = Vec::with_capacity(req.items.len());
        for (index, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total: req.items.len() as u32,
            });
            let image = load_image_tensor(&item.image_path, edge, device)?;
            let x0 = vae.encode_mean(&image)?.detach();
            let (context, pooled) = cache_caption(&caption_encoder, &item.caption)?;
            cache.push((x0, context, pooled));
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
        let time_ids = build_time_ids(device, 1, edge, edge)?.to_dtype(dtype)?;
        let mae = matches!(cfg.loss_type.to_ascii_lowercase().as_str(), "mae" | "l1");
        let stem = file_stem(&req.file_name).to_string();

        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, context, pooled) = &cache[(step as usize - 1) % cache.len()];
            let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(step as u64));
            let timestep = rng.random_range(0..NUM_TRAIN_TIMESTEPS);
            let noise = sample_noise(x0.dims(), noise_seed(cfg.seed, step), device)?;
            let noisy = ddpm_noise(&schedule, x0, &noise, timestep)?.to_dtype(dtype)?;
            let projected = context_projection.forward(&context.to_dtype(dtype)?)?;
            let prediction = unet.forward_instantid(
                &noisy,
                timestep as f64,
                &projected,
                &pooled.to_dtype(dtype)?,
                &time_ids,
                None,
                None,
            )?;
            let loss = epsilon_loss(&prediction, &noise, mae)?;
            last_loss = loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            accumulate_grads(&mut accumulated, grads, &set.vars)?;
            steps_run = step;

            if step % accum == 0 || step == cfg.steps {
                optimizer.set_lr_scaled(lr_multiplier(cfg.lr_scheduler, update, updates, warmup));
                let mut grads = accumulated
                    .take()
                    .expect("an update has accumulated gradients");
                let divisor = accumulation_divisor(step, accum);
                scale_grads(&mut grads, &set.vars, 1.0 / divisor as f64)?;
                clip_grad_norm(&mut grads, &set.vars, 1.0)?;
                optimizer.step(&grads)?;
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
}
