//! Native Candle training for the Mage-Flow base. Adapter runs train the exact dotted projection
//! surface consumed by inference; full runs use a separate owned-parameter loader and publish a
//! complete reloadable transformer component, never an adapter-shaped artifact.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, LoadSpec, Modality, NetworkType, Precision, WeightsSource};
use candle_gen::quant::AdaptLinear;
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{
    self, effective_weight_decay, noise_seed, sample_noise, save_adapter,
    validate_flow_match_request, velocity_loss,
};
use candle_gen::train::lora::{
    build_adapt_lokr_targets, build_adapt_lora_targets, AdaptLoraHost, LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::schedule::{lr_multiplier, schedule_updates};
use candle_gen::{CandleError, Result};

use crate::config::{self, MageConfig, VAE_DOWNSAMPLE};
use crate::rope::{ImgShape, PackLayout};
use crate::{resolve_component_dirs, MageComponentDirs, MageTextEncoder, MageTransformer, MageVae};

const LABEL: &str = "mage_flow_base trainer";
const TRANSFORMER_WEIGHTS: &str = "diffusion_pytorch_model.safetensors";
const TRANSFORMER_CONFIG: &str = "config.json";
const DEFAULT_TARGETS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: config::BASE_MODEL_ID,
        family: config::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        supports_full_finetune: true,
    }
}

pub struct MageTrainer {
    descriptor: TrainerDescriptor,
    dirs: MageComponentDirs,
    device: Device,
}

pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: expected a snapshot directory"
            )))
        }
    };
    if spec.precision != Precision::Bf16 || spec.quantize.is_some() {
        return Err(CandleError::Msg(format!(
            "{LABEL}: requires the dense bf16 base tier"
        )));
    }
    let dirs = resolve_component_dirs(root, spec).map_err(|e| CandleError::Msg(e.to_string()))?;
    for (name, dir) in [
        ("transformer", &dirs.transformer),
        ("text_encoder", &dirs.text_encoder),
    ] {
        if packed_component(dir)? {
            return Err(CandleError::Msg(format!(
                "{LABEL}: {name} at {} is packed/quantized; install the dense bf16 tier",
                dir.display()
            )));
        }
    }
    Ok(Box::new(MageTrainer {
        descriptor: trainer_descriptor(),
        dirs,
        device: candle_gen::default_device()?,
    }))
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl Trainer for MageTrainer {
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

fn packed_component(dir: &Path) -> Result<bool> {
    let config = dir.join(TRANSFORMER_CONFIG);
    let text = match std::fs::read_to_string(&config) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: read {}: {error}",
                config.display()
            )))
        }
    };
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        CandleError::Msg(format!("{LABEL}: parse {}: {error}", config.display()))
    })?;
    Ok(candle_gen::quant::PackedConfig::from_config(&value).is_some())
}

fn validate_request(req: &TrainingRequest) -> Result<()> {
    // Full tuning legitimately has no adapter rank, but every other flow-match knob has identical
    // semantics and must go through the same fail-closed scheduler/bias/loss validation.
    let mut normalized;
    let validated = if req.config.full_finetune && req.config.rank == 0 {
        normalized = req.clone();
        normalized.config.rank = 1;
        &normalized
    } else {
        req
    };
    validate_flow_match_request(validated, LABEL)?;
    let requested_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);
    if req.config.full_finetune && requested_dtype != DType::F32 {
        return Err(CandleError::Msg(format!(
            "{LABEL}: full fine-tuning uses f32 master weights and currently requires train_dtype=f32; got '{}'",
            req.config.train_dtype
        )));
    }
    if req.config.gradient_checkpointing {
        return Err(CandleError::Msg(format!(
            "{LABEL}: gradient checkpointing is not yet implemented"
        )));
    }
    if req.config.resume {
        return Err(CandleError::Msg(format!(
            "{LABEL}: resume is not yet implemented"
        )));
    }
    if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
        return Err(CandleError::Msg(format!(
            "{LABEL}: in-training previews are not yet implemented"
        )));
    }
    if req
        .items
        .iter()
        .any(|item| item.control_image_path.is_some())
    {
        return Err(CandleError::Msg(format!(
            "{LABEL}: control images are not supported"
        )));
    }
    Ok(())
}

impl AdaptLoraHost for MageTransformer {
    fn visit_adapt_lora_mut(
        &mut self,
        visitor: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        self.visit_adaptable_mut(&mut |path, linear| {
            visitor(path, linear).map_err(|error| candle_core::Error::Msg(error.to_string()))
        })?;
        Ok(())
    }
}

fn target_suffixes(req: &TrainingRequest) -> Vec<String> {
    if req.config.lora_target_modules.is_empty() {
        DEFAULT_TARGETS
            .iter()
            .map(|value| value.to_string())
            .collect()
    } else {
        req.config.lora_target_modules.clone()
    }
}

/// Build Mage's flow-match input in the requested transformer compute dtype while retaining an f32
/// velocity target for the shared loss. The VAE cache is BF16 and the seeded prior is F32; aligning
/// both operands before blending avoids mixed-dtype tensor addition without widening the cache.
fn build_training_batch(
    latent: &Tensor,
    noise: &Tensor,
    sigma: f64,
    compute_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let latent_compute = latent.to_dtype(compute_dtype)?;
    let noise_compute = noise.to_dtype(compute_dtype)?;
    let x_t = ((latent_compute * (1.0 - sigma))? + (noise_compute * sigma)?)?;
    let target = (noise.to_dtype(DType::F32)? - latent.to_dtype(DType::F32)?)?;
    Ok((x_t, target))
}

struct CachedSample {
    latent: Tensor,
    text: Tensor,
    layout: PackLayout,
}

fn cache_samples(
    dirs: &MageComponentDirs,
    req: &TrainingRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<Vec<CachedSample>> {
    on_progress(TrainingProgress::LoadingModel);
    let text_encoder =
        MageTextEncoder::load_component_with_quant(&dirs.text_encoder, false, None, device)?;
    let vae = MageVae::load_full(&dirs.vae, device)?;
    let edge = bucket_resolution(req.config.resolution);
    let grid = edge as usize / VAE_DOWNSAMPLE;
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
        let latent = vae
            .encode_sample(&image, req.config.seed.wrapping_add(index as u64))?
            .permute((0, 2, 3, 1))?
            .reshape((1, grid * grid, config::LATENT_CHANNELS))?
            .detach();
        let text = text_encoder.encode(&item.caption)?.detach();
        let layout =
            PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![text.dim(1)?])?;
        cache.push(CachedSample {
            latent,
            text,
            layout,
        });
    }
    if cache.is_empty() {
        return Err(if req.cancel.is_cancelled() {
            CandleError::Canceled
        } else {
            CandleError::Msg(format!("{LABEL}: no usable dataset items"))
        });
    }
    Ok(cache)
}

enum TrainSurface {
    Adapter(LoraSet),
    Full(Vec<(String, Var)>),
}

impl TrainSurface {
    fn vars(&self) -> Vec<Var> {
        match self {
            Self::Adapter(set) => set.vars.clone(),
            Self::Full(named) => named.iter().map(|(_, var)| var.clone()).collect(),
        }
    }
}

fn save_full_checkpoint(
    named: &[(String, Var)],
    source_dir: &Path,
    output_dir: &Path,
) -> Result<PathBuf> {
    std::fs::create_dir_all(output_dir).map_err(|error| {
        CandleError::Msg(format!("{LABEL}: create {}: {error}", output_dir.display()))
    })?;
    let mut entries = Vec::with_capacity(named.len());
    for (name, var) in named {
        entries.push((
            name.clone(),
            var.as_tensor()
                .to_dtype(DType::BF16)?
                .to_device(&Device::Cpu)?
                .contiguous()?,
        ));
    }
    let metadata = HashMap::from([
        ("networkType".to_string(), "full".to_string()),
        ("family".to_string(), config::FAMILY.to_string()),
    ]);
    let path = output_dir.join(TRANSFORMER_WEIGHTS);
    safetensors07::serialize_to_file(entries, Some(metadata), &path)
        .map_err(|error| CandleError::Msg(format!("{LABEL}: save {}: {error}", path.display())))?;
    std::fs::copy(
        source_dir.join(TRANSFORMER_CONFIG),
        output_dir.join(TRANSFORMER_CONFIG),
    )
    .map_err(|error| CandleError::Msg(format!("{LABEL}: copy config: {error}")))?;
    Ok(path)
}

impl MageTrainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        on_progress(TrainingProgress::Preparing);
        let cache = cache_samples(&self.dirs, req, &self.device, on_progress)?;
        let cfg_text = std::fs::read_to_string(self.dirs.transformer.join(TRANSFORMER_CONFIG))
            .map_err(|error| CandleError::Msg(format!("{LABEL}: read config: {error}")))?;
        let dit_cfg = MageConfig::from_json(&cfg_text)?;
        let compute_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);
        let (transformer, surface) = if req.config.full_finetune {
            let (transformer, named) =
                MageTransformer::load_trainable(&self.dirs.transformer, &dit_cfg, &self.device)?;
            (transformer, TrainSurface::Full(named))
        } else {
            let mut transformer = MageTransformer::load_dtype(
                &self.dirs.transformer,
                &dit_cfg,
                compute_dtype,
                &self.device,
            )?;
            let suffixes = target_suffixes(req);
            let set = match req.config.network_type {
                NetworkType::Lora => build_adapt_lora_targets(
                    &mut transformer,
                    &suffixes,
                    req.config.rank,
                    req.config.alpha,
                    req.config.seed,
                    &self.device,
                )?,
                NetworkType::Lokr => build_adapt_lokr_targets(
                    &mut transformer,
                    &suffixes,
                    req.config.rank,
                    req.config.alpha,
                    req.config.decompose_factor,
                    req.config.seed,
                    &self.device,
                )?,
            };
            (transformer, TrainSurface::Adapter(set))
        };
        let vars = surface.vars();
        let mut optimizer = TrainOptimizer::from_config(
            &req.config.optimizer,
            vars.clone(),
            req.config.learning_rate,
            effective_weight_decay(&req.config),
        )?;
        let accum = req.config.gradient_accumulation.max(1);
        let (updates, warmup) =
            schedule_updates(req.config.steps, accum, req.config.lr_warmup_steps);
        let mae = matches!(
            req.config.loss_type.to_ascii_lowercase().as_str(),
            "mae" | "l1"
        );
        let stem = file_stem(&req.file_name).to_string();
        let mut accumulated = None;
        let mut update = 0;
        let mut steps_run = 0;
        let mut last_loss = 0.0;

        for step in 1..=req.config.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let sample = &cache[(step as usize - 1) % cache.len()];
            let sigma = flow_match::sample_unit_timestep(
                &req.config.timestep_type,
                &req.config.timestep_bias,
                flow_match::timestep_seed(req.config.seed, step),
            ) as f64;
            let noise = sample_noise(
                sample.latent.dims(),
                noise_seed(req.config.seed, step),
                &self.device,
            )?;
            let (x_t, target) = build_training_batch(&sample.latent, &noise, sigma, compute_dtype)?;
            let sigma_tensor = Tensor::new(&[sigma as f32], &self.device)?;
            let prediction = transformer.forward(
                &x_t,
                &sample.text.to_dtype(compute_dtype)?,
                &sigma_tensor,
                &sample.layout,
            )?;
            let loss = velocity_loss(&prediction, &target, mae)?;
            last_loss = loss.to_scalar::<f32>()?;
            let grads = loss.backward()?;
            accumulate_grads(&mut accumulated, grads, &vars)?;
            steps_run = step;
            if step % accum == 0 || step == req.config.steps {
                optimizer.set_lr_scaled(lr_multiplier(
                    req.config.lr_scheduler,
                    update,
                    updates,
                    warmup,
                ));
                let mut grads = accumulated
                    .take()
                    .expect("an optimizer update has accumulated gradients");
                let window = if step % accum == 0 {
                    accum
                } else {
                    step % accum
                };
                scale_grads(&mut grads, &vars, 1.0 / window as f64)?;
                clip_grad_norm(&mut grads, &vars, 1.0)?;
                optimizer.step(&grads)?;
                update += 1;
            }
            on_progress(TrainingProgress::Training {
                step,
                total: req.config.steps,
                loss: last_loss,
            });
            if req.config.save_every > 0
                && step % req.config.save_every == 0
                && step != req.config.steps
            {
                match &surface {
                    TrainSurface::Adapter(set) => {
                        std::fs::create_dir_all(&req.output_dir).map_err(|error| {
                            CandleError::Msg(format!("{LABEL}: create output: {error}"))
                        })?;
                        save_adapter(
                            set,
                            &HashMap::from([("family".into(), config::FAMILY.into())]),
                            &req.output_dir.join(checkpoint_filename(&stem, step)),
                        )?;
                    }
                    TrainSurface::Full(named) => {
                        save_full_checkpoint(
                            named,
                            &self.dirs.transformer,
                            &req.output_dir.join(format!("{stem}-step{step:06}")),
                        )?;
                    }
                }
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }
        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }
        on_progress(TrainingProgress::Saving);
        let output = match &surface {
            TrainSurface::Adapter(set) => {
                std::fs::create_dir_all(&req.output_dir).map_err(|error| {
                    CandleError::Msg(format!("{LABEL}: create output: {error}"))
                })?;
                let path = req.output_dir.join(&req.file_name);
                save_adapter(
                    set,
                    &HashMap::from([("family".into(), config::FAMILY.into())]),
                    &path,
                )?;
                path
            }
            TrainSurface::Full(named) => {
                save_full_checkpoint(named, &self.dirs.transformer, &req.output_dir)?
            }
        };
        Ok(TrainingOutput {
            adapter_path: output,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::backprop::GradStore;
    use candle_gen::gen_core::{AdapterKind as RuntimeAdapterKind, AdapterSpec};

    use super::*;
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem};

    fn values(count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| (((index * 13 + 5) % 29) as f32 - 14.0) * scale)
            .collect()
    }

    fn insert_linear(map: &mut HashMap<String, Tensor>, path: &str, input: usize, output: usize) {
        map.insert(
            format!("{path}.weight"),
            Tensor::from_vec(values(input * output, 0.002), (output, input), &Device::Cpu).unwrap(),
        );
        map.insert(
            format!("{path}.bias"),
            Tensor::from_vec(values(output, 0.001), output, &Device::Cpu).unwrap(),
        );
    }

    fn tiny_config() -> MageConfig {
        MageConfig {
            in_channels: 4,
            out_channels: 4,
            context_in_dim: 8,
            hidden_size: 128,
            num_heads: 1,
            depth: 1,
            axes_dim: config::AXES_DIM,
            checkpoint: false,
            patch_size: 1,
        }
    }

    fn tiny_transformer_dir() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        let mut map = HashMap::new();
        insert_linear(&mut map, "img_in", 4, 128);
        map.insert(
            "txt_norm.weight".into(),
            Tensor::ones(8, DType::F32, &Device::Cpu).unwrap(),
        );
        insert_linear(&mut map, "txt_in", 8, 128);
        insert_linear(
            &mut map,
            "time_text_embed.timestep_embedder.linear_1",
            256,
            128,
        );
        insert_linear(
            &mut map,
            "time_text_embed.timestep_embedder.linear_2",
            128,
            128,
        );
        let block = "transformer_blocks.0";
        insert_linear(&mut map, &format!("{block}.img_mod.1"), 128, 768);
        insert_linear(&mut map, &format!("{block}.txt_mod.1"), 128, 768);
        for name in [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ] {
            insert_linear(&mut map, &format!("{block}.attn.{name}"), 128, 128);
        }
        for name in [
            "norm_q.weight",
            "norm_k.weight",
            "norm_added_q.weight",
            "norm_added_k.weight",
        ] {
            map.insert(
                format!("{block}.attn.{name}"),
                Tensor::ones(128, DType::F32, &Device::Cpu).unwrap(),
            );
        }
        for stream in ["img", "txt"] {
            insert_linear(
                &mut map,
                &format!("{block}.{stream}_mlp.net.0.proj"),
                128,
                256,
            );
            insert_linear(&mut map, &format!("{block}.{stream}_mlp.net.2"), 256, 128);
        }
        insert_linear(&mut map, "norm_out.linear", 128, 256);
        insert_linear(&mut map, "proj_out", 128, 4);
        candle_core::safetensors::save(
            &map,
            temp.path().join("diffusion_pytorch_model.safetensors"),
        )
        .unwrap();
        std::fs::write(temp.path().join(TRANSFORMER_CONFIG), "{}").unwrap();
        temp
    }

    fn tiny_inputs() -> (Tensor, Tensor, Tensor, PackLayout) {
        (
            Tensor::from_vec(values(4, 0.1), (1, 1, 4), &Device::Cpu).unwrap(),
            Tensor::from_vec(values(8, 0.1), (1, 1, 8), &Device::Cpu).unwrap(),
            Tensor::new(&[0.5f32], &Device::Cpu).unwrap(),
            PackLayout::generation(vec![ImgShape::latent(1, 1)], vec![1]).unwrap(),
        )
    }

    #[test]
    fn bf16_cached_latent_and_f32_noise_build_a_finite_typed_batch() {
        let latent = Tensor::new(&[[-1.0f32, 0.25, 2.0]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let noise = Tensor::new(&[[0.5f32, -0.75, 1.25]], &Device::Cpu).unwrap();
        let (x_t, target) = build_training_batch(&latent, &noise, 0.4, DType::BF16).unwrap();

        assert_eq!(x_t.dtype(), DType::BF16);
        assert_eq!(target.dtype(), DType::F32);
        for tensor in [&x_t, &target] {
            assert!(tensor
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .all(f32::is_finite));
        }

        let target_values = target.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(target_values, vec![1.5, -1.0, -0.75]);
    }

    fn full_request() -> TrainingRequest {
        let config = TrainingConfig {
            full_finetune: true,
            rank: 0,
            train_dtype: "f32".into(),
            ..Default::default()
        };
        TrainingRequest {
            items: vec![TrainingItem::captioned(
                "/image.png".into(),
                "caption".into(),
            )],
            config,
            output_dir: "/out".into(),
            file_name: "full.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn full_finetune_preserves_rank_zero_but_rejects_invalid_flow_match_knobs() {
        assert!(validate_request(&full_request()).is_ok());
        for mutate in [
            |request: &mut TrainingRequest| request.config.timestep_type = "mystery".into(),
            |request: &mut TrainingRequest| request.config.timestep_bias = "mystery".into(),
            |request: &mut TrainingRequest| request.config.loss_type = "huber".into(),
        ] {
            let mut request = full_request();
            mutate(&mut request);
            assert!(validate_request(&request).is_err());
        }
    }

    #[test]
    fn dtype_contract_accepts_both_adapter_precisions_and_requires_f32_for_full() {
        for dtype in ["bf16", "bfloat16", "f32", "unknown"] {
            let mut adapter = full_request();
            adapter.config.full_finetune = false;
            adapter.config.rank = 2;
            adapter.config.train_dtype = dtype.into();
            assert!(validate_request(&adapter).is_ok(), "adapter {dtype}");
        }

        let mut default_bf16_full = full_request();
        default_bf16_full.config.train_dtype = "bf16".into();
        let error = validate_request(&default_bf16_full)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires train_dtype=f32"), "{error}");
        assert!(validate_request(&full_request()).is_ok());
    }

    fn step_surface(model: &MageTransformer, vars: &[Var]) -> (GradStore, Vec<f32>) {
        let (image, text, sigma, layout) = tiny_inputs();
        let output = model.forward(&image, &text, &sigma, &layout).unwrap();
        let flat = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let target = Tensor::ones(output.dims(), DType::F32, &Device::Cpu).unwrap();
        let loss = (output.to_dtype(DType::F32).unwrap() - target)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        if !vars.is_empty() {
            assert!(vars.iter().any(|var| grads.get(var.as_tensor()).is_some()));
        }
        (grads, flat)
    }

    #[test]
    fn lora_and_lokr_train_save_and_apply_on_actual_mage_projection() {
        for (network, runtime_kind) in [
            (NetworkType::Lora, RuntimeAdapterKind::Lora),
            (NetworkType::Lokr, RuntimeAdapterKind::Lokr),
        ] {
            let fixture = tiny_transformer_dir();
            let cfg = tiny_config();
            let mut training =
                MageTransformer::load_dtype(fixture.path(), &cfg, DType::F32, &Device::Cpu)
                    .unwrap();
            let suffixes = vec!["proj_out".to_string()];
            let set = match network {
                NetworkType::Lora => {
                    build_adapt_lora_targets(&mut training, &suffixes, 2, 2.0, 7, &Device::Cpu)
                        .unwrap()
                }
                NetworkType::Lokr => {
                    build_adapt_lokr_targets(&mut training, &suffixes, 2, 2.0, -1, 7, &Device::Cpu)
                        .unwrap()
                }
            };
            let (grads, _) = step_surface(&training, &set.vars);
            let mut optimizer =
                TrainOptimizer::from_config("adam", set.vars.clone(), 1e-2, 0.0).unwrap();
            optimizer.step(&grads).unwrap();

            let adapter = fixture.path().join(format!("{network:?}.safetensors"));
            save_adapter(
                &set,
                &HashMap::from([("family".into(), config::FAMILY.into())]),
                &adapter,
            )
            .unwrap();
            let stored = candle_core::safetensors::load(&adapter, &Device::Cpu).unwrap();
            assert!(stored.keys().all(|key| key.starts_with("proj_out.")));

            let baseline =
                MageTransformer::load_dtype(fixture.path(), &cfg, DType::F32, &Device::Cpu)
                    .unwrap();
            let (_, before) = step_surface(&baseline, &[]);
            let mut applied =
                MageTransformer::load_dtype(fixture.path(), &cfg, DType::F32, &Device::Cpu)
                    .unwrap();
            let report = candle_gen::quant::install_dotted_adapters(
                "mage",
                &[AdapterSpec::new(adapter, 1.0, runtime_kind)],
                &Device::Cpu,
                |visitor| applied.visit_adaptable_mut(visitor),
            )
            .unwrap();
            assert_eq!(report.applied, 1);
            let (_, after) = step_surface(&applied, &[]);
            assert_ne!(before, after, "saved {network:?} did not affect inference");
        }
    }

    #[test]
    fn full_surface_updates_separated_weights_and_round_trips_production_loader() {
        let fixture = tiny_transformer_dir();
        let cfg = tiny_config();
        let (model, named) =
            MageTransformer::load_trainable(fixture.path(), &cfg, &Device::Cpu).unwrap();
        let original = |name: &str| {
            named
                .iter()
                .find(|(key, _)| key == name)
                .unwrap()
                .1
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let img_before = original("img_in.weight");
        let out_before = original("proj_out.weight");
        let vars = named.iter().map(|(_, var)| var.clone()).collect::<Vec<_>>();
        let (grads, _) = step_surface(&model, &vars);
        for name in ["img_in.weight", "proj_out.weight"] {
            let var = named
                .iter()
                .find(|(key, _)| key == name)
                .unwrap()
                .1
                .as_tensor();
            let grad = grads
                .get(var)
                .expect("separated live base weight has a gradient");
            assert!(
                grad.abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    > 0.0
            );
        }
        let mut optimizer = TrainOptimizer::from_config("adam", vars, 1e-2, 0.0).unwrap();
        optimizer.step(&grads).unwrap();
        assert_ne!(img_before, original("img_in.weight"));
        assert_ne!(out_before, original("proj_out.weight"));

        let output = tempfile::tempdir().unwrap();
        let weights = save_full_checkpoint(&named, fixture.path(), output.path()).unwrap();
        assert_eq!(weights.file_name().unwrap(), TRANSFORMER_WEIGHTS);
        let saved = candle_core::safetensors::load(&weights, &Device::Cpu).unwrap();
        assert_eq!(saved.len(), named.len());
        for (name, var) in &named {
            let tensor = saved
                .get(name)
                .expect("complete original key set was saved");
            assert_eq!(tensor.dims(), var.dims());
            assert_eq!(tensor.dtype(), DType::BF16);
        }
        let _reloaded = MageTransformer::load(output.path(), &cfg, &Device::Cpu)
            .expect("full checkpoint reloads through the production Mage transformer loader");
    }
}
