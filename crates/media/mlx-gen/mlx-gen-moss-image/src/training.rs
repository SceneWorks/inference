//! Native MOSS transformer-only LoRA/LoKr. Frozen Qwen3.5 conditioning and full FLUX2 VAE
//! encoding are cached before functional autograd; only the explicit adapter-factor map is a leaf.
use crate::training_memory::{
    self as admission, EstimateRequest, FootprintFacts, MemoryEstimate, Phase,
};
use crate::{
    loader::{read_weights, Sources},
    text_encoder::Encoder,
    transformer::Transformer,
};
use mlx_gen::gen_core::{self, moss_image::*, moss_image_training as contract, BucketSchedule};
use mlx_gen::train::{
    checkpoint,
    dataset::{bucket_edges, center_crop_square},
    lora::{
        accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
        build_lora_targets, LoraParams, TrainAdapter,
    },
    schedule::{lr_multiplier, schedule_updates},
};
use mlx_gen::{
    CancelFlag, Error, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{
    random,
    transforms::{eval, keyed_value_and_grad},
    Array, Dtype,
};
use std::path::Path;

struct LoadedTrainer {
    descriptor: TrainerDescriptor,
    variant: MossImageVariant,
    sources: Sources,
    encoder: Option<Encoder>,
    vae: Flux2Vae,
    transformer: Transformer,
    facts: FootprintFacts,
    memory_budget_override: Option<u64>,
    decode_bytes: u64,
}
fn descriptor(variant: MossImageVariant) -> TrainerDescriptor {
    TrainerDescriptor {
        id: variant.provider_id(),
        family: "moss_image",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        supports_full_finetune: false,
        max_reference_images: 0,
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}
fn base_descriptor() -> TrainerDescriptor {
    descriptor(MossImageVariant::Base)
}
fn sft_descriptor() -> TrainerDescriptor {
    descriptor(MossImageVariant::Sft)
}
pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load(spec, MossImageVariant::Base)
}
pub fn load_sft(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load(spec, MossImageVariant::Sft)
}
/// Request-aware lazy trainer. Header facts and validation never allocate model tensors.
/// The override is a native integration/test budget, not a nonexistent generic request cap.
pub struct MossImageTrainer {
    descriptor: TrainerDescriptor,
    variant: MossImageVariant,
    sources: Sources,
    facts: FootprintFacts,
    memory_budget_override: Option<u64>,
    loaded: Option<LoadedTrainer>,
}
fn admission_error(error: admission::AdmissionError) -> Error {
    Error::Unsupported(error.to_string())
}
fn estimate_request(facts: &FootprintFacts, req: &TrainingRequest) -> Result<MemoryEstimate> {
    let shape = EstimateRequest::from_request(req).map_err(admission_error)?;
    admission::estimate(facts, &shape).map_err(admission_error)
}
fn budget(override_bytes: Option<u64>) -> Result<u64> {
    match override_bytes {
        Some(bytes) => Ok(admission::resolve_budget(Some(bytes), 0)),
        None => admission::device_budget_bytes().map_err(admission_error),
    }
}
/// The real eager-loading boundary: no encoder/VAE/backbone allocation runs before admission.
fn admitted_load<T>(
    estimate: &MemoryEstimate,
    budget: u64,
    load: impl FnOnce() -> Result<T>,
) -> Result<T> {
    estimate.check_all(budget).map_err(admission_error)?;
    load()
}
impl MossImageTrainer {
    pub fn load(variant: MossImageVariant, spec: &LoadSpec) -> Result<Self> {
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(
                "MOSS trainer requires the exact unadapted Base/SFT checkpoint".into(),
            ));
        }
        let sources = Sources::prepare(variant, spec)?;
        let facts = FootprintFacts::from_sources(&sources).map_err(admission_error)?;
        Ok(Self {
            descriptor: descriptor(variant),
            variant,
            sources,
            facts,
            memory_budget_override: None,
            loaded: None,
        })
    }
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
        if let Some(loaded) = &mut self.loaded {
            loaded.memory_budget_override = bytes;
        }
    }
}
impl Trainer for MossImageTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_floors(&self.descriptor, req)?;
        self.sources.ensure_unchanged()?;
        estimate_request(&self.facts, req)?
            .check_all(budget(self.memory_budget_override)?)
            .map_err(admission_error)?;
        let shape = EstimateRequest::from_request(req)
            .map_err(admission_error)?
            .with_image_headers(req)
            .map_err(admission_error)?;
        admission::estimate(&self.facts, &shape)
            .map_err(admission_error)?
            .check_all(budget(self.memory_budget_override)?)
            .map_err(admission_error)?;
        Ok(())
    }
    fn train(
        &mut self,
        req: &TrainingRequest,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        if self.loaded.is_none() {
            let shape = EstimateRequest::from_request(req)
                .map_err(admission_error)?
                .with_image_headers(req)
                .map_err(admission_error)?;
            let estimate = admission::estimate(&self.facts, &shape).map_err(admission_error)?;
            self.loaded = Some(admitted_load(
                &estimate,
                budget(self.memory_budget_override)?,
                || {
                    load_weights(
                        self.sources.clone(),
                        self.variant,
                        self.facts.clone(),
                        self.memory_budget_override,
                        shape.decode_bytes,
                    )
                },
            )?);
        }
        self.loaded
            .as_mut()
            .expect("admitted model initialized above")
            .train(req, progress)
    }
}
fn load(spec: &LoadSpec, variant: MossImageVariant) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(MossImageTrainer::load(variant, spec)?))
}
mlx_gen::register_trainer! { pub(crate) const BASE_REGISTRATION = base_descriptor => load_base }
mlx_gen::register_trainer! { pub(crate) const SFT_REGISTRATION = sft_descriptor => load_sft }
fn load_weights(
    sources: Sources,
    variant: MossImageVariant,
    facts: FootprintFacts,
    memory_budget_override: Option<u64>,
    decode_bytes: u64,
) -> Result<LoadedTrainer> {
    let cancel = CancelFlag::new();
    mlx_rs::memory::clear_cache();
    let encoder = Encoder::load(&sources, &cancel)?;
    let weights = read_weights(
        sources
            .ordinary
            .iter()
            .filter(|a| a.relative == "vae/diffusion_pytorch_model.safetensors"),
        |k| {
            k.starts_with("encoder.")
                || k.starts_with("decoder.")
                || k.starts_with("quant_conv.")
                || k.starts_with("post_quant_conv.")
                || k == "bn.running_mean"
                || k == "bn.running_var"
        },
        &cancel,
    )?;
    let vae = Flux2Vae::from_weights_with_decoder_config(
        &weights,
        &Flux2VaeDecoderConfig {
            block_out_channels: [96, 192, 384, 384],
        },
    )?;
    weights.materialize_accessed()?;
    if !weights.unused_keys().is_empty() {
        return Err(Error::Unsupported(format!(
            "unconsumed MOSS training VAE tensors {:?}",
            weights.unused_keys()
        )));
    }
    let transformer = sources.transformer(MossImageAcceleration::Off, &cancel)?;
    sources.ensure_unchanged()?;
    Ok(LoadedTrainer {
        facts,
        memory_budget_override,
        decode_bytes,
        descriptor: descriptor(variant),
        variant,
        sources,
        encoder: Some(encoder),
        vae,
        transformer,
    })
}
fn validate_floors(d: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    gen_core::train::validate_control_request(d, req)?;
    gen_core::train::validate_full_finetune_request(d, req)?;
    gen_core::train::validate_training_techniques(d, req)?;
    gen_core::train::validate_edit_request(d, req)?;
    let c = &req.config;
    if req.items.is_empty()
        || c.steps == 0
        || c.rank == 0
        || c.batch_size == 0
        || c.gradient_accumulation == 0
    {
        return Err(Error::Msg(
            "MOSS requires a nonempty dataset and positive steps/rank/batch/accumulation".into(),
        )
        .into());
    }
    if !c.alpha.is_finite()
        || c.alpha <= 0.
        || !c.learning_rate.is_finite()
        || c.learning_rate <= 0.
    {
        return Err(
            Error::Msg("MOSS alpha and learning rate must be finite and positive".into()).into(),
        );
    }
    if !TrainOptimizer::is_supported(&c.optimizer) {
        return Err(
            Error::Unsupported(format!("unsupported MOSS optimizer {}", c.optimizer)).into(),
        );
    }
    if !["bf16", "bfloat16", "f32", "float32", "fp32"]
        .contains(&c.train_dtype.to_ascii_lowercase().as_str())
    {
        return Err(Error::Unsupported("MOSS training dtype must be bf16 or f32 (FP32 adapter masters/residuals in both modes)".into()).into());
    }
    if ![
        "",
        "default",
        contract::SAMPLER,
        "moss_stage3",
        "sigmoid",
        "linear",
        "uniform",
        "weighted",
    ]
    .contains(&c.timestep_type.as_str())
    {
        return Err(Error::Unsupported(format!(
            "unknown MOSS timestep policy {}",
            c.timestep_type
        ))
        .into());
    }
    if ![
        "balanced",
        "none",
        "neutral",
        "high",
        "high_noise",
        "favor_high_noise",
        "low",
        "low_noise",
        "favor_low_noise",
    ]
    .contains(&c.timestep_bias.as_str())
    {
        return Err(Error::Unsupported("unknown MOSS timestep bias".into()).into());
    }
    if !["mse", "l2", "mae", "l1"].contains(&c.loss_type.as_str()) {
        return Err(Error::Unsupported("MOSS loss must be mse/l2 or mae/l1".into()).into());
    }
    contract::target_paths(42, &c.lora_target_modules)?;
    for (k, v) in &c.model_options {
        if k != "vae_posterior" || v.as_str() != Some(contract::POSTERIOR) {
            return Err(Error::Unsupported(format!(
                "unsupported MOSS training option {k}={v}; posterior policy is {}",
                contract::POSTERIOR
            ))
            .into());
        }
    }
    if c.resume && c.save_every > 0 && !c.save_every.is_multiple_of(c.gradient_accumulation) {
        return Err(Error::Unsupported(
            "MOSS resume checkpoints must fall on complete accumulation windows".into(),
        )
        .into());
    }
    if c.sample_every > 0 && !c.sample_prompts.is_empty() && c.sample_steps == 0 {
        return Err(
            Error::Unsupported("MOSS preview requires positive sample_steps".into()).into(),
        );
    }
    if req.file_name.is_empty()
        || Path::new(&req.file_name)
            .file_name()
            .and_then(|s| s.to_str())
            != Some(req.file_name.as_str())
    {
        return Err(
            Error::Unsupported("MOSS adapter filename must be a single filename".into()).into(),
        );
    }
    Ok(())
}

/// Raw NHWC32 -> channel-first (c,ph,pw) packing. BN normalization is applied by the caller.
pub(crate) fn pack_mean(mean: &Array) -> Result<Array> {
    let s = mean.shape();
    if s.len() != 4 || s[3] != 32 || s[1] % 2 != 0 || s[2] % 2 != 0 {
        return Err(Error::Msg(
            "MOSS VAE mean must be NHWC32 with even spatial dimensions".into(),
        ));
    }
    Ok(mean
        .reshape(&[s[0], s[1] / 2, 2, s[2] / 2, 2, 32])?
        .transpose_axes(&[0, 5, 2, 4, 1, 3])?
        .reshape(&[s[0], 128, s[1] / 2, s[2] / 2])?)
}
fn encode_image(vae: &Flux2Vae, image: &mlx_gen::media::Image, edge: u32) -> Result<Array> {
    let pixels = mlx_gen::img2img::preprocess_init_image(image, edge, edge)?
        .transpose_axes(&[0, 2, 3, 1])?;
    vae.bn_normalize_nchw(&pack_mean(&vae.encode_mean(&pixels)?)?)
}
/// Common periodic/final portable export; request words are independent of caption injection.
pub(super) fn save(
    model: &mut Transformer,
    adapter: &TrainAdapter,
    params: &LoraParams,
    variant: MossImageVariant,
    req: &TrainingRequest,
    path: &Path,
) -> Result<()> {
    let cfg = &req.config;
    let mut metadata = contract::metadata(variant, sampler_name(cfg));
    if !req.trigger_words.is_empty() {
        metadata.insert(
            "triggerWords".into(),
            serde_json::Value::from(req.trigger_words.clone()).to_string(),
        );
    }
    let extra: Vec<_> = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    crate::adapters::save_validated(model, adapter, params, cfg, &extra, path)
}
fn caption(item: &str, trigger: Option<&str>) -> String {
    match trigger.filter(|s| !s.trim().is_empty()) {
        Some(t) if !item.contains(t) => format!("{t}, {item}"),
        _ => item.to_string(),
    }
}
fn sampler_name(c: &TrainingConfig) -> &str {
    match c.timestep_type.as_str() {
        "" | "default" | "moss_stage3" => contract::SAMPLER,
        s => s,
    }
}
fn sigma(c: &TrainingConfig, seed: u64) -> Result<f32> {
    let u = contract::unit_draw(seed);
    let mut t = if sampler_name(c) == contract::SAMPLER {
        contract::stage3_sigma(u)?
    } else {
        let key = random::key(seed)?;
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        match c.timestep_type.as_str() {
            "uniform" | "linear" => u as f32,
            "weighted" => {
                (u as f32
                    + sigmoid(random::normal::<f32>(&[1], None, None, Some(&key))?.item::<f32>()))
                    / 2.
            }
            _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&key))?.item::<f32>()),
        }
        .clamp(0.001, 0.999)
    };
    match c.timestep_bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t = t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t *= t,
        _ => {}
    }
    Ok(t)
}

/// The raw MOSS velocity convention, established independently from its Euler generation step.
fn flow_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    Ok((
        add(
            multiply(x0, Array::from_f32(1. - sigma))?,
            multiply(noise, Array::from_f32(sigma))?,
        )?,
        subtract(noise, x0)?,
        sigma * 1000.,
    ))
}

/// The production differentiable seam. Factors install *inside* the traced closure. Per-block
/// checkpoint segments receive all adapter arrays as explicit inputs, never detached captures.
#[allow(clippy::too_many_arguments)]
pub(crate) fn loss_grads(
    transformer: &mut Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    cfg: &TrainingConfig,
    x0: &Array,
    context: &Array,
    noise: &Array,
    sigma: f32,
    mask: Option<&Array>,
    cancel: &CancelFlag,
) -> Result<(f32, LoraParams)> {
    crate::pipeline::check(cancel)?;
    let (x, target, time) = flow_batch(x0, noise, sigma)?;
    transformer.training = true;
    transformer.training_checkpoint = cfg.gradient_checkpointing;
    let loss = |p: LoraParams| -> MlxResult<Vec<Array>> {
        transformer
            .install_training_adapter(adapter, &p, cfg.alpha, cfg.rank as f32)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let pred = transformer
            .forward(&x, &[time], std::slice::from_ref(context), &[], cancel)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let difference = subtract(pred.as_dtype(Dtype::Float32)?, &target)?;
        Ok(vec![mlx_gen::train::loss::reduce_loss(
            &difference,
            mask,
            matches!(cfg.loss_type.as_str(), "mae" | "l1"),
        )?])
    };
    let mut loss = loss;
    let function = |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> { loss(p) };
    let result = {
        let mut vg = keyed_value_and_grad(function);
        vg(params.clone(), 0)
    };
    transformer.training = false;
    transformer.training_checkpoint = false;
    crate::pipeline::check(cancel)?;
    let (values, grads) = result?;
    eval(values.iter().chain(grads.values()))?;
    for (key, gradient) in &grads {
        if !gradient.is_finite()?.all(None)?.item::<bool>() {
            return Err(Error::Msg(format!("MOSS nonfinite adapter gradient {key}")));
        }
    }
    let loss = values[0].item::<f32>();
    if !loss.is_finite() {
        return Err(Error::Msg("MOSS training produced nonfinite loss".into()));
    }
    Ok((loss, grads))
}
// Appendix A.2 Eq3 averages over image tokens, not captions or equally weighted images.
// Accumulation windows may contain different bucket areas, so carry the actual token denominator.
fn token_count(latent: &Array) -> Result<u32> {
    let shape = latent.shape();
    if shape.len() != 4 || shape[2] <= 0 || shape[3] <= 0 {
        return Err(Error::Msg(
            "MOSS loss requires a nonempty NCHW latent grid".into(),
        ));
    }
    (shape[2] as u32)
        .checked_mul(shape[3] as u32)
        .ok_or_else(|| Error::Msg("MOSS token count overflow".into()))
}
fn token_weighted_grads(grads: LoraParams, tokens: u32) -> Result<LoraParams> {
    Ok(grads
        .into_iter()
        .map(|(key, value)| Ok((key, multiply(value, Array::from_f32(tokens as f32))?)))
        .collect::<MlxResult<_>>()?)
}

struct Entry {
    latent: Array,
    context: Array,
    mask: Option<Array>,
}
impl Trainer for LoadedTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_floors(&self.descriptor, req)
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
impl LoadedTrainer {
    fn admit(&self, req: &TrainingRequest, phase: Phase) -> Result<()> {
        let mut shape = EstimateRequest::from_request(req).map_err(admission_error)?;
        shape.decode_bytes = self.decode_bytes;
        admission::estimate(&self.facts, &shape)
            .map_err(admission_error)?
            .check(phase, budget(self.memory_budget_override)?)
            .map_err(admission_error)
    }
    fn admit_image(&self, req: &TrainingRequest, path: &Path) -> Result<()> {
        let mut shape = EstimateRequest::from_request(req).map_err(admission_error)?;
        shape.decode_bytes = self
            .decode_bytes
            .max(admission::image_decode_bytes(path).map_err(admission_error)?);
        admission::estimate(&self.facts, &shape)
            .map_err(admission_error)?
            .check(Phase::Cache, budget(self.memory_budget_override)?)
            .map_err(admission_error)
    }
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let c = &req.config;
        use sha2::{Digest, Sha256};
        let mut identity_config = c.clone();
        identity_config.resume = false;
        identity_config.model_options.insert(
            "vae_posterior".into(),
            serde_json::json!(contract::POSTERIOR),
        );
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(format!(
                "moss-resume-v1:{}:{}:{identity_config:?}",
                self.variant.revision(),
                checkpoint::request_fingerprint(req)?
            ))
        );
        let identity = checkpoint::ResumeIdentity {
            config: c,
            request_fingerprint: &fingerprint,
        };
        self.sources.ensure_unchanged()?;
        self.admit(req, Phase::Cast)?;
        let dtype = if matches!(
            c.train_dtype.to_ascii_lowercase().as_str(),
            "bf16" | "bfloat16"
        ) {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        self.transformer.cast_training_weights(dtype)?;
        progress(TrainingProgress::Preparing);
        progress(TrainingProgress::LoadingModel);
        self.admit(req, Phase::Cache)?;
        let edges = bucket_edges(c);
        let mut cache = Vec::new();
        let encoder=self.encoder.as_ref().ok_or_else(||Error::Msg("MOSS trainer instance already consumed its conditioning cache; reload for a new dataset".into()))?;
        for (i, item) in req.items.iter().enumerate() {
            crate::pipeline::check(&req.cancel)?;
            self.admit_image(req, &item.image_path)?;
            progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total: req.items.len() as u32,
            });
            let rgb = image::open(&item.image_path)
                .map_err(|e| Error::Msg(e.to_string()))?
                .to_rgb8();
            let image = center_crop_square(&mlx_gen::media::Image {
                width: rgb.width(),
                height: rgb.height(),
                pixels: rgb.into_raw(),
            });
            let (context, _) = encoder.encode(
                &caption(&item.caption, c.trigger_word.as_deref()),
                4096,
                &req.cancel,
            )?;
            let mask = mlx_gen::train::subject_mask::PreparedSubjectMask::load_if_enabled(
                "MOSS",
                item,
                c.subject_mask_loss.as_ref(),
            )?;
            for edge in &edges {
                crate::pipeline::check(&req.cancel)?;
                let latent = encode_image(&self.vae, &image, *edge)?;
                let mask = mlx_gen::train::loss::prepared_subject_mask_weight(
                    "MOSS",
                    mask.as_ref(),
                    mlx_gen::train::subject_mask::CropBox::center_square,
                    latent.shape(),
                )?;
                eval([&latent, &context])?;
                cache.push(Entry {
                    latent,
                    context: context.clone(),
                    mask,
                });
            }
        }
        let mut previews = Vec::new();
        if c.sample_every > 0 {
            for prompt in c.sample_prompts.iter().take(4) {
                previews.push((prompt.clone(), encoder.encode(prompt, 4096, &req.cancel)?.0));
            }
        }
        let negative = if previews.is_empty() {
            None
        } else {
            Some(encoder.encode("", 4096, &req.cancel)?.0)
        };
        self.encoder = None;
        mlx_rs::memory::clear_cache();
        self.admit(req, Phase::Backward)?;
        let paths = contract::target_paths(42, &c.lora_target_modules)?;
        let (adapter, mut params) = match c.network_type {
            NetworkType::Lora => {
                let (targets, p) =
                    build_lora_targets(&mut self.transformer, &paths, c.rank as i32, c.seed)?;
                (TrainAdapter::Lora { targets }, p)
            }
            NetworkType::Lokr => {
                let (targets, p) = build_lokr_targets(
                    &mut self.transformer,
                    &paths,
                    c.rank as i32,
                    c.decompose_factor,
                    c.seed,
                )?;
                (TrainAdapter::Lokr { targets }, p)
            }
        };
        let mut opt = TrainOptimizer::from_config(
            &c.optimizer,
            c.learning_rate,
            if c.optimizer.eq_ignore_ascii_case("adam") {
                0.
            } else {
                c.weight_decay
            },
        )?;
        let accum = c.gradient_accumulation;
        let (updates, warmup) = schedule_updates(c.steps, accum, c.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| Error::Msg("invalid adapter filename".into()))?;
        let (mut start, mut update) = (0, 0);
        if c.resume {
            if let Some((path, _)) = checkpoint::find_latest_resume(&req.output_dir, stem) {
                let (loaded, meta) =
                    checkpoint::load_resume_with_identity(&path, &mut opt, identity)?;
                if loaded.len() != params.len()
                    || params
                        .iter()
                        .any(|(k, v)| loaded.get(k).is_none_or(|p| p.shape() != v.shape()))
                {
                    return Err(Error::Unsupported(
                        "MOSS resume factor keys/shapes differ from requested adapter".into(),
                    ));
                }
                if meta.step >= c.steps {
                    return Err(Error::Unsupported("MOSS resume already reached the requested training steps; no new training would run".into()));
                }
                params = loaded;
                start = meta.step;
                update = meta.update_idx;
            }
        }
        let schedule = BucketSchedule::new(req.items.len(), &c.training_buckets(), c.seed);
        let mut accumulated = None;
        let mut accumulated_tokens = 0u32;
        let mut final_loss = 0.;
        let mut steps = start;
        for step in start + 1..=c.steps {
            crate::pipeline::check(&req.cancel)?;
            self.admit(req, Phase::Backward)?;
            let mut micro = None;
            let mut total_loss = 0.;
            let mut total_tokens = 0u32;
            for batch in 0..c.batch_size {
                let index = (step as usize - 1) * c.batch_size as usize + batch as usize;
                let entry = &cache[schedule.cache_index(index)];
                let seed = c.seed.wrapping_add(index as u64 * 2 + 1);
                let noise = random::normal::<f32>(
                    entry.latent.shape(),
                    None,
                    None,
                    Some(&random::key(seed)?),
                )?;
                let (loss, grads) = loss_grads(
                    &mut self.transformer,
                    &params,
                    &adapter,
                    c,
                    &entry.latent,
                    &entry.context,
                    &noise,
                    sigma(c, seed ^ 0xd1b54a32d192ed03)?,
                    entry.mask.as_ref(),
                    &req.cancel,
                )?;
                let tokens = token_count(&entry.latent)?;
                total_tokens = total_tokens
                    .checked_add(tokens)
                    .ok_or_else(|| Error::Msg("MOSS batch token count overflow".into()))?;
                total_loss += loss * tokens as f32;
                accumulate_grads(&mut micro, token_weighted_grads(grads, tokens)?)?;
            }
            final_loss = total_loss / total_tokens as f32;
            accumulated_tokens = accumulated_tokens
                .checked_add(total_tokens)
                .ok_or_else(|| Error::Msg("MOSS accumulation token count overflow".into()))?;
            accumulate_grads(&mut accumulated, micro.expect("positive batch size"))?;
            steps = step;
            if step % accum == 0 || step == c.steps {
                opt.set_lr_scaled(lr_multiplier(c.lr_scheduler, update, updates, warmup));

                let avg = average_grads(
                    accumulated.take().expect("completed accumulation window"),
                    accumulated_tokens,
                )?;
                accumulated_tokens = 0;
                adapter_optimizer_update(&mut opt, &mut params, &avg, c, update, c.seed)?;
                update += 1;
            }
            progress(TrainingProgress::Training {
                step,
                total: c.steps,
                loss: final_loss,
            });
            if c.save_every > 0 && step % c.save_every == 0 && step != c.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                save(
                    &mut self.transformer,
                    &adapter,
                    &params,
                    self.variant,
                    req,
                    &req.output_dir
                        .join(checkpoint::checkpoint_filename(stem, step)),
                )?;
                // No partial accumulation is persisted as a valid resumable checkpoint.
                if step % accum == 0 {
                    checkpoint::save_resume_with_identity(
                        &req.output_dir,
                        stem,
                        step,
                        update,
                        &opt,
                        &params,
                        identity,
                    )?;
                }
                progress(TrainingProgress::Checkpoint { step });
            }
            if c.sample_every > 0 && step % c.sample_every == 0 && !previews.is_empty() {
                self.admit(req, Phase::Preview)?;
                self.transformer.install_training_adapter(
                    &adapter,
                    &params,
                    c.alpha,
                    c.rank as f32,
                )?;
                self.previews(
                    c,
                    step,
                    &previews,
                    negative.as_ref().expect("preview conditioning"),
                    &req.cancel,
                    progress,
                )?;
            }
        }
        if steps == 0 {
            return Err(Error::Canceled);
        }
        crate::pipeline::check(&req.cancel)?;
        progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let path = req.output_dir.join(&req.file_name);
        save(
            &mut self.transformer,
            &adapter,
            &params,
            self.variant,
            req,
            &path,
        )?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps,
            final_loss,
        })
    }
    fn previews(
        &self,
        c: &TrainingConfig,
        step: u32,
        prompts: &[(String, Array)],
        negative: &Array,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<()> {
        for (i, (prompt, context)) in prompts.iter().enumerate() {
            let edge = mlx_gen::train::dataset::bucket_resolution(c.resolution);
            let request = mlx_gen::GenerationRequest {
                prompt: prompt.clone(),
                width: edge,
                height: edge,
                steps: Some(c.sample_steps),
                guidance: Some(c.sample_guidance_scale),
                moss_image: Some(MossImageOptions {
                    uncond_skip_layers: Some(Vec::new()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let resolved = MossImageResolved::resolve(self.variant, &request)?;
            let values = crate::pipeline::normal(
                c.seed.wrapping_add(i as u64),
                128 * (edge as usize / 16).pow(2),
            );
            let noise =
                Array::from_slice(&values, &[1, 128, (edge / 16) as i32, (edge / 16) as i32]);
            let latent = crate::pipeline::Sampling {
                transformer: &self.transformer,
                resolved: &resolved,
                positive: std::slice::from_ref(context),
                negative: std::slice::from_ref(negative),
                dataspace: None,
            }
            .run(&noise, cancel, &mut |_| {})?;
            let decoded = self
                .vae
                .decode_packed_latents(&latent.transpose_axes(&[0, 2, 3, 1])?)?;
            let image = mlx_gen::image::decoded_to_image(&decoded.transpose_axes(&[0, 3, 1, 2])?)?;
            progress(TrainingProgress::Sample {
                step,
                index: i as u32 + 1,
                total: prompts.len() as u32,
                prompt: prompt.clone(),
                image,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_token_normalization_preserves_different_bucket_areas() {
        let _device = crate::cpu_test_device();
        let mut total = None;
        for (gradient, tokens) in [(3f32, 2), (11f32, 6)] {
            let grads = [(
                std::rc::Rc::<str>::from("factor"),
                Array::from_f32(gradient),
            )]
            .into_iter()
            .collect();
            accumulate_grads(&mut total, token_weighted_grads(grads, tokens).unwrap()).unwrap();
        }
        let average = average_grads(total.unwrap(), 8).unwrap();
        assert_eq!(average["factor"].item::<f32>(), 9.);
        assert_eq!(
            token_count(&Array::zeros::<f32>(&[1, 128, 2, 3]).unwrap()).unwrap(),
            6
        );
        assert!(token_count(&Array::zeros::<f32>(&[1, 2, 3]).unwrap()).is_err());
    }
    #[test]
    fn flow_batch_matches_unnegated_velocity_and_thousand_scale_time() {
        let _device = crate::cpu_test_device();
        let (x, target, time) = flow_batch(
            &Array::from_slice(&[2f32, -4.], &[2]),
            &Array::from_slice(&[6f32, 8.], &[2]),
            0.25,
        )
        .unwrap();
        assert_eq!(
            crate::pipeline::logical_f32_readback(&x).unwrap(),
            vec![3., -1.]
        );
        assert_eq!(
            crate::pipeline::logical_f32_readback(&target).unwrap(),
            vec![4., 12.]
        );
        assert_eq!(time, 250.);
    }
    #[test]
    fn packing_uses_channel_then_patch_coordinates_not_lens_token_layout() {
        let _device = crate::cpu_test_device();
        let values: Vec<_> = (0..2 * 4 * 32).map(|i| i as f32).collect();
        let packed = pack_mean(&Array::from_slice(&values, &[1, 2, 4, 32])).unwrap();
        assert_eq!(packed.shape(), &[1, 128, 1, 2]);
        let out = crate::pipeline::logical_f32_readback(&packed).unwrap();
        for c in 0..32 {
            for ph in 0..2 {
                for pw in 0..2 {
                    for w in 0..2 {
                        assert_eq!(
                            out[(c * 4 + ph * 2 + pw) * 2 + w],
                            values[(ph * 4 + w * 2 + pw) * 32 + c]
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod review_admission_tests {
    use super::*;
    #[test]
    fn refused_working_set_never_enters_the_eager_loader() {
        let (facts, shape) = admission::tests::tiny();
        let estimate = admission::estimate(&facts, &shape).unwrap();
        let loads = std::cell::Cell::new(0);
        let result = admitted_load(&estimate, estimate.peak() - 1, || {
            loads.set(loads.get() + 1);
            Ok(())
        });
        assert!(
            matches!(result,Err(Error::Unsupported(message)) if message.contains("structural working-set"))
        );
        assert_eq!(
            loads.get(),
            0,
            "eager allocation ran before structural admission"
        );
        admitted_load(&estimate, estimate.peak(), || {
            loads.set(loads.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(loads.get(), 1);
    }
    #[test]
    fn concrete_trainer_override_refuses_before_missing_model_or_dataset_can_load() {
        let _device = crate::cpu_test_device();
        let (facts, _) = admission::tests::tiny();
        let sources = Sources {
            selection: None,
            ordinary: Vec::new(),
            optional: Vec::new(),
            config: facts.config.clone(),
        };
        let mut trainer = MossImageTrainer {
            descriptor: descriptor(MossImageVariant::Base),
            variant: MossImageVariant::Base,
            sources,
            facts,
            memory_budget_override: None,
            loaded: None,
        };
        trainer.set_memory_budget_override(Some(0));
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("not-created");
        let request = TrainingRequest {
            items: vec![gen_core::TrainingItem::captioned(
                temp.path().join("not-opened.png"),
                "caption".into(),
            )],
            config: TrainingConfig {
                resolution: 32,
                rank: 1,
                ..Default::default()
            },
            output_dir: output.clone(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        };
        let result = trainer.train(&request, &mut |_| {
            panic!("no model/cache/progress before refusal")
        });
        assert!(
            matches!(result,Err(gen_core::Error::Unsupported(message)) if message.contains("structural working-set")),
            "request reached missing assets instead of admission"
        );
        assert!(trainer.loaded.is_none());
        assert!(!output.exists());
    }
}
