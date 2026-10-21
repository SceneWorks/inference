//! Native Candle/CUDA transformer-only MOSS Base/SFT training. Source checkpoint precision is
//! independent of compute precision; frozen Qwen/VAE conditioning is cached before backward.
use crate::{
    loader::{load_weights, Sources},
    text_encoder::{self, Conditioning},
    training_memory as admission,
    transformer::{check_cancel, Geometry, Transformer},
};
use candle_gen::candle_core::{backprop::GradStore, DType, Device, Tensor};
use candle_gen::gen_core::{
    self, moss_image::*, moss_image_training as contract, BucketSchedule, CancelFlag, LoadSpec,
    Modality, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::train::{
    checkpoint,
    dataset::{bucket_edges, decode_square, square_image_tensor},
    flow_match::{prepared_subject_mask_weight, sample_noise},
    lora::{self, LoraSet},
    optim::{accumulate_grads, scale_grads, TrainOptimizer},
    schedule::{lr_multiplier, schedule_updates},
};
use candle_gen::{CandleError, Result};
use std::{collections::HashMap, path::Path};

fn descriptor(variant: MossImageVariant) -> TrainerDescriptor {
    TrainerDescriptor {
        id: variant.provider_id(),
        family: "moss_image",
        backend: "candle",
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
fn load_base(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(MossImageTrainer::load_on_device(
        MossImageVariant::Base,
        spec,
        candle_gen::default_device()?,
    )?))
}
fn load_sft(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(MossImageTrainer::load_on_device(
        MossImageVariant::Sft,
        spec,
        candle_gen::default_device()?,
    )?))
}
candle_gen::register_trainer! { pub(crate) const BASE_REGISTRATION = base_descriptor => load_base }
candle_gen::register_trainer! { pub(crate) const SFT_REGISTRATION = sft_descriptor => load_sft }

/// Header-only lazy trainer. Device identity is supplied by the allocator, never inferred from a
/// physical index. The optional native budget is also the explicit CPU numerical-oracle boundary.
pub struct MossImageTrainer {
    descriptor: TrainerDescriptor,
    variant: MossImageVariant,
    sources: Sources,
    facts: admission::FootprintFacts,
    device: Device,
    memory_budget_override: Option<u64>,
}
fn admission_error(e: admission::AdmissionError) -> CandleError {
    CandleError::Unsupported(e.to_string())
}
fn admitted_load<T>(
    estimate: &admission::MemoryEstimate,
    budget: u64,
    load: impl FnOnce() -> Result<T>,
) -> Result<T> {
    estimate.check_all(budget).map_err(admission_error)?;
    load()
}
impl MossImageTrainer {
    pub fn load_on_device(
        variant: MossImageVariant,
        spec: &LoadSpec,
        device: Device,
    ) -> Result<Self> {
        if !spec.adapters.is_empty() {
            return Err(CandleError::Msg(
                "MOSS training requires the exact unadapted Base/SFT parent checkpoint".into(),
            ));
        }
        let sources = Sources::selected(variant, spec, MossImageAcceleration::Off)?;
        let facts = admission::FootprintFacts::from_sources(&sources).map_err(admission_error)?;
        Ok(Self {
            descriptor: descriptor(variant),
            variant,
            sources,
            facts,
            device,
            memory_budget_override: None,
        })
    }
    /// Zero is a zero budget, never a guard bypass. This is not a generic request user-cap field.
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
    }
    fn budget(&self) -> Result<u64> {
        match self.memory_budget_override {
            Some(bytes) => Ok(admission::resolve_budget(Some(bytes), 0)),
            None => admission::device_budget_bytes(&self.device).map_err(admission_error),
        }
    }
    fn estimate(&self, req: &TrainingRequest) -> Result<admission::MemoryEstimate> {
        let shape = admission::EstimateRequest::from_request(req)
            .map_err(admission_error)?
            .with_image_headers(req)
            .map_err(admission_error)?;
        admission::estimate(&self.facts, &shape).map_err(admission_error)
    }
}
impl Trainer for MossImageTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)?;
        self.sources.ensure_unchanged()?;
        self.estimate(req)?
            .check_all(self.budget()?)
            .map_err(admission_error)?;
        Ok(())
    }
    fn train(
        &mut self,
        req: &TrainingRequest,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        self.train_impl(req, progress).map_err(Into::into)
    }
}
fn validate_request(d: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    gen_core::train::validate_control_request(d, req)?;
    gen_core::train::validate_full_finetune_request(d, req)?;
    gen_core::train::validate_training_techniques(d, req)?;
    gen_core::train::validate_edit_request(d, req)?;
    let c = &req.config;
    let refuse = |s: &str| gen_core::Error::Unsupported(format!("MOSS training: {s}"));
    if req.items.is_empty()
        || c.steps == 0
        || c.rank == 0
        || c.batch_size == 0
        || c.gradient_accumulation == 0
    {
        return Err(refuse(
            "requires dataset and positive steps/rank/batch/accumulation",
        ));
    }
    if !c.alpha.is_finite()
        || c.alpha <= 0.
        || !c.learning_rate.is_finite()
        || c.learning_rate <= 0.
        || !c.weight_decay.is_finite()
        || c.weight_decay < 0.
    {
        return Err(refuse(
            "alpha/lr must be finite positive and weight decay finite nonnegative",
        ));
    }
    if !TrainOptimizer::is_supported(&c.optimizer) {
        return Err(refuse(&format!("unsupported optimizer {}", c.optimizer)));
    }
    compute_dtype(c)?;
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
        return Err(refuse(&format!(
            "unknown timestep policy {}",
            c.timestep_type
        )));
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
        return Err(refuse("unknown timestep bias"));
    }
    if !["mse", "l2", "mae", "l1"].contains(&c.loss_type.as_str()) {
        return Err(refuse("loss must be mse/l2 or mae/l1"));
    }
    if c.decompose_factor != -1 && c.decompose_factor <= 0 {
        return Err(refuse("LoKr decomposition must be -1 (auto) or positive"));
    }
    contract::target_paths(42, &c.lora_target_modules)?;
    for (key, value) in &c.model_options {
        if key != "vae_posterior" || value.as_str() != Some(contract::POSTERIOR) {
            return Err(refuse(&format!(
                "unsupported model option {key}={value}; posterior is {}",
                contract::POSTERIOR
            )));
        }
    }
    if c.resume && c.save_every > 0 && !c.save_every.is_multiple_of(c.gradient_accumulation) {
        return Err(refuse(
            "resumable checkpoints must fall on complete accumulation windows",
        ));
    }
    if c.sample_every > 0
        && !c.sample_prompts.is_empty()
        && (c.sample_steps == 0 || !c.sample_guidance_scale.is_finite())
    {
        return Err(refuse(
            "preview requires positive steps and finite guidance",
        ));
    }
    if req.file_name.is_empty()
        || Path::new(&req.file_name)
            .file_name()
            .and_then(|s| s.to_str())
            != Some(req.file_name.as_str())
    {
        return Err(refuse("adapter filename must be a single filename"));
    }
    Ok(())
}
fn compute_dtype(c: &TrainingConfig) -> gen_core::Result<DType> {
    match c.train_dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f32" | "float32" | "fp32" => Ok(DType::F32),
        _ => Err(gen_core::Error::Unsupported(
            "MOSS compute dtype must be bf16 or f32; both use FP32 adapter masters/residuals"
                .into(),
        )),
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
    let mut s = if sampler_name(c) == contract::SAMPLER {
        contract::stage3_sigma(u)?
    } else {
        let normal =
            || -> Result<f32> { Ok(sample_noise(&[1], seed, &Device::Cpu)?.to_vec1::<f32>()?[0]) };
        let sigmoid = |z: f32| 1. / (1. + (-z).exp());
        match c.timestep_type.as_str() {
            "uniform" | "linear" => u as f32,
            "weighted" => (u as f32 + sigmoid(normal()?)) / 2.,
            _ => sigmoid(normal()?),
        }
        .clamp(0.001, 0.999)
    };
    match c.timestep_bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => s = s.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => s *= s,
        _ => {}
    }
    Ok(s)
}
fn flow_batch(x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<(Tensor, Tensor, f32)> {
    Ok((
        ((x0 * f64::from(1. - sigma))? + (noise * f64::from(sigma))?)?,
        noise.sub(x0)?,
        1000. * sigma,
    ))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn loss_grads(
    model: &Transformer,
    set: &LoraSet,
    c: &TrainingConfig,
    x0: &Tensor,
    context: &Tensor,
    noise: &Tensor,
    sigma: f32,
    mask: Option<&Tensor>,
    cancel: &CancelFlag,
) -> Result<(f32, GradStore)> {
    let (x, target, time) = flow_batch(x0, noise, sigma)?;
    let result = model.loss_and_grads(
        &x,
        time,
        context,
        &target,
        mask,
        matches!(c.loss_type.as_str(), "mae" | "l1"),
        c.gradient_checkpointing,
        &set.vars,
        cancel,
    )?;
    // A mathematically zero gradient is valid (including zero-initialized LoRA A), but
    // a missing leaf means the graph was detached. Never silently publish a partial update.
    for (name, var) in set.named_vars() {
        if result.1.get(var.as_tensor()).is_none() {
            return Err(CandleError::Msg(format!(
                "MOSS autograd omitted trainable factor {name}; no adapter update was performed"
            )));
        }
    }
    Ok(result)
}
struct Entry {
    latent: Tensor,
    context: Tensor,
    mask: Option<Tensor>,
}
fn caption(text: &str, trigger: Option<&str>) -> String {
    match trigger.filter(|t| !t.trim().is_empty()) {
        Some(t) if !text.contains(t) => format!("{t}, {text}"),
        _ => text.into(),
    }
}
pub(super) fn save(
    model: &Transformer,
    set: &LoraSet,
    variant: MossImageVariant,
    req: &TrainingRequest,
    path: &Path,
) -> Result<()> {
    let c = &req.config;
    for (_, var) in set.named_vars() {
        crate::adapters::finite(var.as_tensor(), "saved factor")?;
    }
    // Validate derived products, not only their finite source factors, after the final update too.
    model.validate_residuals()?;
    let mut meta: HashMap<_, _> = contract::metadata(variant, sampler_name(c))
        .into_iter()
        .collect();
    if !req.trigger_words.is_empty() {
        meta.insert(
            "triggerWords".into(),
            serde_json::to_string(&req.trigger_words)
                .map_err(|e| CandleError::Msg(e.to_string()))?,
        );
    }
    match set.kind {
        lora::AdapterKind::Lora => lora::save_lora_peft(set, "", &meta, path),
        lora::AdapterKind::Lokr => lora::save_lokr(set, &meta, path),
    }
}
fn token_count(latent: &Tensor) -> Result<u64> {
    let (batch, _, h, w) = latent.dims4()?;
    (batch as u64)
        .checked_mul(h as u64)
        .and_then(|n| n.checked_mul(w as u64))
        .filter(|n| *n > 0)
        .ok_or_else(|| CandleError::Msg("MOSS token count overflow/empty grid".into()))
}
/// Image-token weighting spans the entire optimizer window, including variable bucket areas,
/// all batch items, and the final incomplete accumulation window.
#[derive(Default)]
struct GradientWindow {
    gradients: Option<GradStore>,
    tokens: u64,
}
impl GradientWindow {
    fn add(
        &mut self,
        mut gradients: GradStore,
        vars: &[candle_gen::candle_core::Var],
        tokens: u64,
    ) -> Result<()> {
        if tokens == 0 {
            return Err(CandleError::Msg(
                "MOSS cannot accumulate an empty token grid".into(),
            ));
        }
        self.tokens = self
            .tokens
            .checked_add(tokens)
            .ok_or_else(|| CandleError::Msg("MOSS accumulation token overflow".into()))?;
        scale_grads(&mut gradients, vars, tokens as f64)?;
        accumulate_grads(&mut self.gradients, gradients, vars)
    }
    fn take(&mut self, vars: &[candle_gen::candle_core::Var]) -> Result<GradStore> {
        if self.tokens == 0 {
            return Err(CandleError::Msg(
                "MOSS cannot normalize an empty optimizer window".into(),
            ));
        }
        let mut gradients = self
            .gradients
            .take()
            .ok_or_else(|| CandleError::Msg("MOSS missing accumulation gradients".into()))?;
        scale_grads(&mut gradients, vars, 1. / self.tokens as f64)?;
        self.tokens = 0;
        Ok(gradients)
    }
}
impl MossImageTrainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let c = &req.config;
        let estimate = self.estimate(req)?;
        // Snapshot once, before owned allocations. Comparing the total plan with later remaining
        // free memory would charge this run's own weights twice.
        let budget = self.budget()?;
        estimate.check_all(budget).map_err(admission_error)?;
        self.sources.ensure_unchanged()?;
        check_cancel(&req.cancel)?;
        progress(TrainingProgress::Preparing);
        progress(TrainingProgress::LoadingModel);
        estimate
            .check(admission::Phase::Load, budget)
            .map_err(admission_error)?;
        let (encoder, vae) = admitted_load(&estimate, budget, || {
            Ok((
                self.sources.text(&self.device)?,
                self.sources.training_vae(&self.device)?,
            ))
        })?;
        estimate
            .check(admission::Phase::Cache, budget)
            .map_err(admission_error)?;
        let edges = bucket_edges(c);
        let mut cache = Vec::new();
        for (index, item) in req.items.iter().enumerate() {
            check_cancel(&req.cancel)?;
            self.sources.ensure_unchanged()?;
            progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total: req.items.len() as u32,
            });
            self.estimate(req)?
                .check(admission::Phase::Cache, budget)
                .map_err(admission_error)?;
            let square = decode_square(&item.image_path)?;
            let context = text_encoder::strings(
                &self.sources,
                &encoder,
                &[caption(&item.caption, c.trigger_word.as_deref())],
                4096,
                &self.device,
                &req.cancel,
            )?
            .rows
            .remove(0)
            .detach();
            let subject = gen_core::train::subject_mask::PreparedSubjectMask::load_if_enabled(
                "MOSS",
                item,
                c.subject_mask_loss.as_ref(),
            )?;
            for edge in &edges {
                check_cancel(&req.cancel)?;
                let latent = vae
                    .encode_packed(&square_image_tensor(&square, *edge, &self.device)?)?
                    .detach();
                crate::adapters::finite(&latent, "cached latent")?;
                let mask = prepared_subject_mask_weight(
                    "MOSS",
                    subject.as_ref(),
                    gen_core::train::subject_mask::CropBox::center_square,
                    latent.dims(),
                    &self.device,
                )?;
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
                previews.push((
                    prompt.clone(),
                    text_encoder::strings(
                        &self.sources,
                        &encoder,
                        std::slice::from_ref(prompt),
                        4096,
                        &self.device,
                        &req.cancel,
                    )?
                    .rows
                    .remove(0)
                    .detach(),
                ));
            }
        }
        let negative = if previews.is_empty() {
            None
        } else {
            Some(
                text_encoder::strings(
                    &self.sources,
                    &encoder,
                    &[String::new()],
                    4096,
                    &self.device,
                    &req.cancel,
                )?
                .rows
                .remove(0)
                .detach(),
            )
        };
        self.device.synchronize()?;
        drop(encoder);
        estimate
            .check(admission::Phase::Cast, budget)
            .map_err(admission_error)?;
        let weights = load_weights(
            &self.sources.assets(MossImageAcceleration::Off)?,
            MossImageAssetRole::Transformer,
            &self.device,
            compute_dtype(c)?,
            &[""],
        )?;
        let mut transformer =
            Transformer::load(&weights, Geometry::from(&self.sources.config), None)?;
        drop(weights);
        estimate
            .check(admission::Phase::Backward, budget)
            .map_err(admission_error)?;
        let set = transformer.training_adapter(c)?;
        let mut opt = TrainOptimizer::from_config(
            &c.optimizer,
            set.vars.clone(),
            c.learning_rate,
            if c.optimizer.eq_ignore_ascii_case("adam") {
                0.
            } else {
                c.weight_decay
            },
        )?;
        let stem = checkpoint::file_stem(&req.file_name);
        let fingerprint = resume_fingerprint(self.variant, req)?;
        let (mut start, mut update) = (0, 0);
        if c.resume {
            if let Some((path, _)) = checkpoint::find_latest_resume(&req.output_dir, stem) {
                let meta = checkpoint::load_resume(&path, &mut opt, &set, c, &fingerprint)?;
                if meta.step >= c.steps {
                    return Err(CandleError::Msg(
                        "MOSS resume already reached requested steps; no new training would run"
                            .into(),
                    ));
                }
                start = meta.step;
                update = meta.update_idx;
                transformer.validate_residuals()?;
            }
        }
        let schedule = BucketSchedule::new(req.items.len(), &c.training_buckets(), c.seed);
        let (updates, warmup) =
            schedule_updates(c.steps, c.gradient_accumulation, c.lr_warmup_steps);
        let mut window = GradientWindow::default();
        let mut final_loss = 0.;
        for step in start + 1..=c.steps {
            check_cancel(&req.cancel)?;
            estimate
                .check(admission::Phase::Backward, budget)
                .map_err(admission_error)?;
            let mut batch_loss = 0.;
            let mut batch_tokens = 0u64;
            for batch in 0..c.batch_size {
                let index = (step as usize - 1)
                    .checked_mul(c.batch_size as usize)
                    .and_then(|n| n.checked_add(batch as usize))
                    .ok_or_else(|| CandleError::Msg("MOSS sample index overflow".into()))?;
                let entry = &cache[schedule.cache_index(index)];
                let seed = c
                    .seed
                    .wrapping_add((index as u64).wrapping_mul(2).wrapping_add(1));
                let noise = sample_noise(entry.latent.dims(), seed, &self.device)?;
                let (loss, grads) = loss_grads(
                    &transformer,
                    &set,
                    c,
                    &entry.latent,
                    &entry.context,
                    &noise,
                    sigma(c, seed ^ 0xd1b54a32d192ed03)?,
                    entry.mask.as_ref(),
                    &req.cancel,
                )?;
                let tokens = token_count(&entry.latent)?;
                window.add(grads, &set.vars, tokens)?;
                batch_tokens = batch_tokens
                    .checked_add(tokens)
                    .ok_or_else(|| CandleError::Msg("MOSS batch token overflow".into()))?;
                batch_loss += loss * tokens as f32;
            }
            final_loss = batch_loss / batch_tokens as f32;
            if step % c.gradient_accumulation == 0 || step == c.steps {
                let mut grads = window.take(&set.vars)?;
                opt.set_lr_scaled(lr_multiplier(c.lr_scheduler, update, updates, warmup));
                lora::adapter_optimizer_step(&mut opt, &mut grads, &set, c, update, c.seed)?;
                transformer.validate_residuals()?;
                update += 1;
            }
            progress(TrainingProgress::Training {
                step,
                total: c.steps,
                loss: final_loss,
            });
            if c.save_every > 0 && step % c.save_every == 0 && step != c.steps {
                std::fs::create_dir_all(&req.output_dir)
                    .map_err(|e| CandleError::Msg(e.to_string()))?;
                save(
                    &transformer,
                    &set,
                    self.variant,
                    req,
                    &req.output_dir
                        .join(checkpoint::checkpoint_filename(stem, step)),
                )?;
                if step % c.gradient_accumulation == 0 {
                    checkpoint::save_resume(
                        &req.output_dir,
                        stem,
                        step,
                        update,
                        &opt,
                        &set,
                        c,
                        &fingerprint,
                    )?;
                }
                progress(TrainingProgress::Checkpoint { step });
            }
            if c.sample_every > 0 && step % c.sample_every == 0 && !previews.is_empty() {
                estimate
                    .check(admission::Phase::Preview, budget)
                    .map_err(admission_error)?;
                for (index, (prompt, context)) in previews.iter().enumerate() {
                    let edge = candle_gen::train::dataset::bucket_resolution(c.resolution);
                    let request = gen_core::GenerationRequest {
                        prompt: prompt.clone(),
                        width: edge,
                        height: edge,
                        steps: Some(c.sample_steps),
                        guidance: Some(c.sample_guidance_scale),
                        seed: Some(c.seed.wrapping_add(index as u64)),
                        cancel: req.cancel.clone(),
                        moss_image: Some(MossImageOptions {
                            uncond_skip_layers: Some(Vec::new()),
                            ..Default::default()
                        }),
                        ..Default::default()
                    };
                    let resolved = MossImageResolved::resolve(self.variant, &request)?;
                    let positive = Conditioning {
                        rows: vec![context.clone()],
                        receipts: vec![],
                    };
                    let negative = Conditioning {
                        rows: vec![negative
                            .as_ref()
                            .expect("preview conditioning cached")
                            .clone()],
                        receipts: vec![],
                    };
                    transformer.freeze_residuals(true);
                    let preview = (|| -> Result<_> {
                        let sampled = crate::pipeline::sample_graph(
                            &transformer,
                            None,
                            &request,
                            &resolved,
                            MossImageAdvancedInputs::default(),
                            &positive,
                            Some(&negative),
                            &mut |_| {},
                        )?;
                        let mut images =
                            crate::pipeline::decode(&vae, &sampled.latents, &request, &mut |_| {})?;
                        Ok(images.remove(0))
                    })();
                    transformer.freeze_residuals(false);
                    progress(TrainingProgress::Sample {
                        step,
                        index: index as u32 + 1,
                        total: previews.len() as u32,
                        prompt: prompt.clone(),
                        image: preview?,
                    });
                }
            }
        }
        check_cancel(&req.cancel)?;
        self.sources.ensure_unchanged()?;
        progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir).map_err(|e| CandleError::Msg(e.to_string()))?;
        let path = req.output_dir.join(&req.file_name);
        save(&transformer, &set, self.variant, req, &path)?;
        Ok(TrainingOutput {
            adapter_path: path,
            steps: c.steps,
            final_loss,
        })
    }
}
fn resume_fingerprint(variant: MossImageVariant, req: &TrainingRequest) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut config = req.config.clone();
    config.resume = false;
    config.model_options.insert(
        "vae_posterior".into(),
        serde_json::json!(contract::POSTERIOR),
    );
    Ok(format!(
        "{:x}",
        Sha256::digest(format!(
            "moss-resume-v1:{}:{}:{config:?}",
            variant.revision(),
            gen_core::train::resume::request_fingerprint(req)?
        ))
    ))
}

#[cfg(test)]
mod control_tests {
    use super::*;
    #[test]
    fn admission_refuses_before_the_real_eager_loading_closure() {
        use std::cell::Cell;
        let (facts, shape) = admission::tests::tiny();
        let estimate = admission::estimate(&facts, &shape).unwrap();
        let called = Cell::new(false);
        let result = admitted_load(&estimate, estimate.peak() - 1, || {
            called.set(true);
            Ok(())
        });
        assert!(matches!(result, Err(CandleError::Unsupported(_))));
        assert!(!called.get());
        admitted_load(&estimate, estimate.peak(), || {
            called.set(true);
            Ok(())
        })
        .unwrap();
        assert!(called.get());
    }
    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![gen_core::TrainingItem::captioned(
                "not-loaded.png".into(),
                "subject".into(),
            )],
            config: TrainingConfig {
                train_dtype: "f32".into(),
                steps: 3,
                rank: 2,
                alpha: 0.375,
                batch_size: 2,
                gradient_accumulation: 2,
                ..Default::default()
            },
            output_dir: "unused".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }
    #[test]
    fn controls_are_preserved_or_actionably_refused_not_defaulted() {
        let mut req = request();
        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let descriptor = descriptor(variant);
            for dtype in ["f32", "fp32", "float32", "bf16", "bfloat16"] {
                req.config.train_dtype = dtype.into();
                validate_request(&descriptor, &req).unwrap();
                assert_eq!(
                    compute_dtype(&req.config).unwrap(),
                    if dtype.starts_with('b') {
                        DType::BF16
                    } else {
                        DType::F32
                    }
                );
            }
            for policy in [
                contract::SAMPLER,
                "uniform",
                "linear",
                "sigmoid",
                "weighted",
            ] {
                req.config.timestep_type = policy.into();
                validate_request(&descriptor, &req).unwrap();
                let a = sigma(&req.config, 73).unwrap();
                assert_eq!(a, sigma(&req.config, 73).unwrap());
                assert!(a.is_finite() && (0.0..=1.0).contains(&a));
            }
            req.config.train_dtype = "fp16".into();
            assert!(validate_request(&descriptor, &req).is_err());
            req.config.train_dtype = "f32".into();
            req.config.lora_target_modules = vec!["q".into(), "typo".into()];
            assert!(validate_request(&descriptor, &req).is_err());
            req.config.lora_target_modules.clear();
            req.config
                .model_options
                .insert("unrecognized".into(), serde_json::json!(true));
            assert!(validate_request(&descriptor, &req).is_err());
            req.config.model_options.clear();
            req.config.model_options.insert(
                "vae_posterior".into(),
                serde_json::json!(contract::POSTERIOR),
            );
            validate_request(&descriptor, &req).unwrap();
            for value in [0., f32::NAN, f32::INFINITY] {
                req.config.alpha = value;
                assert!(validate_request(&descriptor, &req).is_err());
            }
            req.config.alpha = 0.375;
            req.config.optimizer = "unknown".into();
            assert!(validate_request(&descriptor, &req).is_err());
            req.config.optimizer = "adamw".into();
            req.config.timestep_type = "wrong".into();
            assert!(validate_request(&descriptor, &req).is_err());
            req.config.timestep_type = contract::SAMPLER.into();
        }
    }
    #[test]
    fn objective_uses_raw_velocity_and_thousand_sigma() {
        let x = Tensor::from_vec(vec![2f32, -3.], (1, 2, 1, 1), &Device::Cpu).unwrap();
        let n = Tensor::from_vec(vec![4f32, 5.], (1, 2, 1, 1), &Device::Cpu).unwrap();
        let (sample, target, time) = flow_batch(&x, &n, 0.25).unwrap();
        assert_eq!(crate::loader::tensor_f32(&sample).unwrap(), [2.5, -1.]);
        assert_eq!(crate::loader::tensor_f32(&target).unwrap(), [2., 8.]);
        assert_eq!(time, 250.);
    }
    #[test]
    fn token_denominator_counts_bucket_area_and_batch() {
        assert_eq!(
            token_count(&Tensor::zeros((2, 4, 3, 5), DType::F32, &Device::Cpu).unwrap()).unwrap(),
            30
        );
        let variable = candle_gen::candle_core::Var::zeros((1,), DType::F32, &Device::Cpu).unwrap();
        let vars = vec![variable.clone()];
        let mut window = GradientWindow::default();
        let gradients = |value: f32| {
            let mut store = GradStore::default();
            store.insert(
                variable.as_tensor(),
                Tensor::new(&[value], &Device::Cpu).unwrap(),
            );
            store
        };
        // Different bucket areas in one batch/window must not become equal-weight images.
        window.add(gradients(3.), &vars, 2).unwrap();
        window.add(gradients(11.), &vars, 6).unwrap();
        let averaged = window.take(&vars).unwrap();
        assert_eq!(
            averaged
                .get(variable.as_tensor())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [9.]
        );
        assert_eq!(window.tokens, 0);
        assert!(window.take(&vars).is_err());
        // The final incomplete window uses its own denominator, not the configured accumulation.
        window.add(gradients(7.), &vars, 3).unwrap();
        assert_eq!(
            window
                .take(&vars)
                .unwrap()
                .get(variable.as_tensor())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [7.]
        );
    }
}
