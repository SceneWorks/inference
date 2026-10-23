//! Native Candle/CUDA transformer-only MOSS Base/SFT training. Source checkpoint precision is
//! independent of compute precision; frozen Qwen/VAE conditioning is cached before backward.
#[cfg(feature = "test-support")]
mod audit;
mod recovery;
#[cfg(feature = "test-support")]
use gen_core_testkit::moss_training_probe::{RunProbe, TrainingProbe};

use crate::{
    loader::{load_weights_cancellable, Sources},
    text_encoder::{self, Conditioning},
    training_memory as admission,
    transformer::{check_cancel, Geometry, Transformer},
};
use candle_gen::candle_core::{backprop::GradStore, DType, Device, Tensor};
use candle_gen::gen_core::{
    self, moss_image::*, moss_image_training as contract, CancelFlag, LoadSpec, Modality, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::train::{
    checkpoint,
    dataset::square_image_tensor,
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
    #[cfg(feature = "test-support")]
    test_probe: Option<TrainingProbe>,
    descriptor: TrainerDescriptor,
    variant: MossImageVariant,
    sources: Sources,
    facts: admission::FootprintFacts,
    device: Device,
    memory_budget_override: Option<u64>,
}
struct Preparation {
    plan: contract::recovery::DataPlan,
    estimate: admission::MemoryEstimate,
    budget: u64,
    host_budget: u64,
    #[cfg(feature = "test-support")]
    evaluation: Option<audit::Snapshot>,
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
        Self::from_sources(variant, sources, device)
    }
    pub(crate) fn from_sources(
        variant: MossImageVariant,
        sources: Sources,
        device: Device,
    ) -> Result<Self> {
        let facts = admission::FootprintFacts::from_sources(&sources).map_err(admission_error)?;
        Ok(Self {
            descriptor: descriptor(variant),
            variant,
            sources,
            facts,
            device,
            memory_budget_override: None,
            #[cfg(feature = "test-support")]
            test_probe: None,
        })
    }
    /// Zero is a zero budget, never a guard bypass. This is not a generic request user-cap field.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn set_test_probe(&mut self, probe: Option<TrainingProbe>) {
        self.test_probe = probe;
    }
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_memory_plan(&self, req: &TrainingRequest) -> Result<serde_json::Value> {
        let estimate = self.prepare(req)?.estimate;
        Ok(
            serde_json::json!({"load_bytes":estimate.load,"cast_bytes":estimate.cast,"cache_bytes":estimate.cache,"backward_bytes":estimate.backward,"preview_bytes":estimate.preview,"checkpoint_bytes":estimate.checkpoint,"encoder_only_peak_bytes":admission::encoder_work(&self.facts.encoder_config,4096).map_err(admission_error)?.checked_add(self.facts.encoder_bytes).and_then(|n|n.checked_add(self.facts.largest_source_shard)).ok_or_else(||CandleError::Msg("encoder-only peak overflow".into()))?,"host_staging_peak_bytes":estimate.host_peak_bytes,"tensor_peak_bytes":estimate.peak(),"context_tokens":estimate.context_token_bound,"source_shard_bytes":self.facts.largest_source_shard,"encoder_resident_bytes":self.facts.encoder_bytes,"vae_resident_bytes":self.facts.vae_bytes,"trainable_elements":estimate.trainable_elements,"cache_bytes_persistent":estimate.cache_bytes}),
        )
    }
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
    }
    fn budget(&self, req: &TrainingRequest) -> Result<u64> {
        // Only the explicitly selected CPU numerical oracle has no CUDA physical discovery.
        // A runtime override is a lower ceiling on a real device, never a VRAM substitute.
        let (physical, total) = match (&self.device, self.memory_budget_override) {
            // Explicit CPU numerical-reference domain; this is not physical GPU evidence.
            (Device::Cpu, Some(bytes)) => (admission::resolve_budget(Some(bytes), 0), bytes),
            _ => {
                let (discovered, total) =
                    admission::device_memory_budget(&self.device).map_err(admission_error)?;
                (
                    self.memory_budget_override
                        .map_or(discovered, |limit| discovered.min(limit)),
                    total,
                )
            }
        };
        gen_core::train::effective_memory_budget_with_fraction(
            physical,
            Some(total),
            req.config.memory_limit_bytes,
            req.config.memory_limit_fraction,
        )
        .map_err(Into::into)
    }
    fn basic(&self, req: &TrainingRequest) -> Result<()> {
        validate_request(&self.descriptor, req)?;
        contract::target_paths(
            self.sources.config.num_layers,
            &req.config.lora_target_modules,
        )?;
        self.sources.ensure_unchanged()?;
        check_cancel(&req.cancel)
    }
    fn prepare(&self, req: &TrainingRequest) -> Result<Preparation> {
        let budget = self.budget(req)?;
        let minimal = self
            .facts
            .backbone_elements
            .checked_mul(2)
            .ok_or_else(|| CandleError::Msg("minimum backbone residency overflow".into()))?;
        if minimal > budget {
            return Err(CandleError::Unsupported(
                "frozen backbone alone exceeds the physical device/user cap".into(),
            ));
        }
        let available = contract::recovery::host::available_bytes()?;
        let host_budget = if self.device.is_cpu() {
            available.min(budget)
        } else {
            available
        };
        let plan = contract::recovery::DataPlan::from_request_budget(req, host_budget)?;
        let mut shape = admission::EstimateRequest::from_request(req)
            .map_err(admission_error)?
            .with_plan(req, &plan)
            .map_err(admission_error)?;
        shape.context_tokens = text_encoder::token_bound(&self.sources, req)?;
        #[cfg(feature = "test-support")]
        let evaluation = self
            .test_probe
            .as_ref()
            .map(|probe| audit::Snapshot::new(req, probe.evaluation(), host_budget))
            .transpose()?;
        #[cfg(feature = "test-support")]
        if let Some(evaluation) = &evaluation {
            let tokens = text_encoder::token_bound(&self.sources, &evaluation.request)?;
            let geometry = &evaluation.plan.entries[0];
            shape.context_tokens = shape.context_tokens.max(tokens);
            shape.evaluation_context_tokens = tokens;
            shape.evaluation_canvas = Some((u64::from(geometry.width), u64::from(geometry.height)));
            shape.audit_workspace_bytes = 64 * 1024 * 1024;
            shape.decode_bytes = shape
                .decode_bytes
                .max(evaluation.plan.decode_buffer_bound()?);
            shape.dataset_metadata_bytes = shape
                .dataset_metadata_bytes
                .checked_add(contract::recovery::DataPlan::metadata_bound(
                    &evaluation.request,
                )?)
                .ok_or_else(|| CandleError::Msg("evaluation metadata overflow".into()))?;
        }
        let estimate = admission::estimate(&self.facts, &shape).map_err(admission_error)?;
        estimate.check_all(budget).map_err(admission_error)?;
        estimate
            .check_host(host_budget, self.device.is_cpu())
            .map_err(admission_error)?;
        contract::recovery::validate_resume_plan(
            req,
            &self.sources.source_identity(self.variant),
            &plan,
        )?;
        Ok(Preparation {
            plan,
            estimate,
            budget,
            host_budget,
            #[cfg(feature = "test-support")]
            evaluation,
        })
    }
}
impl Trainer for MossImageTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        self.basic(req)?;
        self.prepare(req)?;
        Ok(())
    }
    fn train(
        &mut self,
        req: &TrainingRequest,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.basic(req)?;
        let prepared = self.prepare(req)?;
        #[cfg(feature = "test-support")]
        let probe = self
            .test_probe
            .as_ref()
            .map(|p| p.begin(self.device.is_cpu()))
            .transpose()?;
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.bind_request(
                req,
                &prepared.plan,
                &self.sources.source_identity(self.variant),
            )?;
        }
        #[cfg(feature = "test-support")]
        let result = self.train_impl(req, progress, prepared, probe.clone());
        #[cfg(not(feature = "test-support"))]
        let result = self.train_impl(req, progress, prepared);
        // Drain the actual owned context even on cancellation/error, after run locals drop.
        let synchronized = self.device.synchronize().map_err(CandleError::from);
        let result = match synchronized {
            Ok(()) => result,
            Err(error) => Err(error),
        };
        #[cfg(feature = "test-support")]
        if let Some(probe) = probe {
            probe.finish(result.as_ref().err().map(ToString::to_string))?;
        }
        result.map_err(Into::into)
    }
}
fn validate_request(d: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    gen_core::train::validate_control_request(d, req)?;
    gen_core::train::validate_full_finetune_request(d, req)?;
    gen_core::train::validate_training_techniques(d, req)?;
    gen_core::train::validate_edit_request(d, req)?;
    let c = &req.config;
    if !req.file_name.ends_with(".safetensors") {
        return Err(gen_core::Error::Unsupported(
            "MOSS adapter output must use .safetensors".into(),
        ));
    }
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
    for (key, value) in &c.model_options {
        if key != "vae_posterior" || value.as_str() != Some(contract::POSTERIOR) {
            return Err(refuse(&format!(
                "unsupported model option {key}={value}; posterior is {}",
                contract::POSTERIOR
            )));
        }
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
fn encode_entry(
    req: &TrainingRequest,
    geometry: &contract::recovery::EntryGeometry,
    plan: &contract::recovery::DataPlan,
    host_budget: u64,
    vae: &candle_gen_flux2::vae::Flux2Vae,
    device: &Device,
) -> Result<(Tensor, Option<Tensor>)> {
    check_cancel(&req.cancel)?;
    let source = plan.decode_rgb(geometry.item, host_budget)?;
    let pixels = if req.config.aspect_ratio_buckets.is_empty() {
        let crop = geometry.crop_box();
        let square =
            image::imageops::crop_imm(&source, crop.x, crop.y, crop.width, crop.height).to_image();
        square_image_tensor(&square, geometry.width, device)?
    } else {
        let row = gen_core::AspectRatioBucket {
            width: geometry.width,
            height: geometry.height,
            repeats: 1,
        };
        let rgb = gen_core::train::aspect_ratio::resize_rgb(&source, row)?;
        Tensor::from_vec(
            rgb,
            (1, geometry.height as usize, geometry.width as usize, 3),
            &Device::Cpu,
        )?
        .permute((0, 3, 1, 2))?
        .affine(1. / 127.5, -1.)?
        .to_device(device)?
    };
    check_cancel(&req.cancel)?;
    let latent = vae.encode_packed(&pixels)?.detach();
    check_cancel(&req.cancel)?;
    crate::adapters::finite(&latent, "encoded training latent")?;
    let subject = plan.subject_mask(req, geometry.item, host_budget)?;
    let mask = prepared_subject_mask_weight(
        "MOSS",
        subject.as_ref(),
        |_, _| geometry.crop_box(),
        latent.dims(),
        device,
    )?;
    if req.config.subject_mask_loss.is_some() && mask.is_none() {
        return Err(CandleError::Msg(
            "MOSS configured subject mask was not produced".into(),
        ));
    }
    Ok((latent, mask))
}
fn caption(text: &str, trigger: Option<&str>) -> String {
    match trigger.filter(|t| !t.trim().is_empty()) {
        Some(t) if !text.contains(t) => format!("{t}, {text}"),
        _ => text.into(),
    }
}
#[cfg(test)]
pub(super) fn save(
    model: &Transformer,
    set: &LoraSet,
    variant: MossImageVariant,
    req: &TrainingRequest,
    path: &Path,
) -> Result<()> {
    save_with_source(model, set, variant, req, path, None)
}
fn save_with_source(
    model: &Transformer,
    set: &LoraSet,
    variant: MossImageVariant,
    req: &TrainingRequest,
    path: &Path,
    identity: Option<&contract::recovery::RunIdentity>,
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
    if let Some(identity) = identity {
        meta.insert(
            "source_checkpoint_identity".into(),
            identity.source_checkpoint.clone(),
        );
        meta.insert(
            "dataset_fingerprint".into(),
            identity.dataset_fingerprint.clone(),
        );
        meta.insert(
            "effective_training_controls".into(),
            identity.effective_config.clone(),
        );
        meta.insert(
            "training_stream_version".into(),
            contract::recovery::STREAM_VERSION.into(),
        );
    }
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
        prepared: Preparation,
        #[cfg(feature = "test-support")] probe: Option<RunProbe>,
    ) -> Result<TrainingOutput> {
        let c = &req.config;

        let plan = prepared.plan;
        let estimate = prepared.estimate;
        let budget = prepared.budget;
        let host_budget = prepared.host_budget;
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.update(|r| r.context_bound = estimate.context_token_bound)?;
        }
        self.sources.ensure_unchanged()?;
        check_cancel(&req.cancel)?;
        progress(TrainingProgress::Preparing);
        progress(TrainingProgress::LoadingModel);
        estimate
            .check(admission::Phase::Load, budget)
            .map_err(admission_error)?;
        let (encoder, vae) = admitted_load(&estimate, budget, || {
            Ok((
                self.sources.text_cancellable(&self.device, &req.cancel)?,
                self.sources
                    .training_vae_cancellable(&self.device, &req.cancel)?,
            ))
        })?;
        estimate
            .check(admission::Phase::Cache, budget)
            .map_err(admission_error)?;

        #[cfg(feature = "test-support")]
        let te_before = if probe.is_some() {
            Some(audit::digest(
                encoder
                    .frozen_training_tensors()
                    .map_err(|e| CandleError::Msg(e.to_string()))?,
            )?)
        } else {
            None
        };
        #[cfg(feature = "test-support")]
        let vae_before = if probe.is_some() {
            Some(audit::digest(vae.frozen_training_tensors())?)
        } else {
            None
        };
        #[cfg(feature = "test-support")]
        let evaluation = probe
            .as_ref()
            .map(|p| {
                audit::prepare(
                    &self.sources,
                    prepared
                        .evaluation
                        .as_ref()
                        .expect("admitted heldout snapshot"),
                    p,
                    &encoder,
                    &vae,
                    &self.device,
                    host_budget,
                )
            })
            .transpose()?;
        let fingerprint = contract::recovery::effective_identity(
            req,
            &self.sources.source_identity(self.variant),
            &plan,
        )?;
        let mut cache = Vec::with_capacity(plan.entries.len());
        let mut contexts = Vec::with_capacity(req.items.len());
        for (index, item) in req.items.iter().enumerate() {
            check_cancel(&req.cancel)?;
            progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total: req.items.len() as u32,
            });
            contexts.push(if c.cache_text_embeddings {
                Some(
                    text_encoder::strings(
                        &self.sources,
                        &encoder,
                        &[caption(&item.caption, c.trigger_word.as_deref())],
                        4096,
                        &self.device,
                        &req.cancel,
                    )?
                    .rows
                    .remove(0)
                    .detach(),
                )
            } else {
                None
            });
            for geometry in plan
                .entries
                .iter()
                .filter(|geometry| geometry.item == contexts.len() - 1)
            {
                check_cancel(&req.cancel)?;
                self.sources.ensure_unchanged()?;
                estimate
                    .check(admission::Phase::Cache, budget)
                    .map_err(admission_error)?;
                cache.push(if c.cache_latents {
                    Some(encode_entry(
                        req,
                        geometry,
                        &plan,
                        host_budget,
                        &vae,
                        &self.device,
                    )?)
                } else {
                    None
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
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.frozen(
                "text_encoder",
                te_before.as_ref().expect("observed encoder"),
                &audit::digest(
                    encoder
                        .frozen_training_tensors()
                        .map_err(|e| CandleError::Msg(e.to_string()))?,
                )?,
            )?;
            probe.frozen(
                "vae",
                vae_before.as_ref().expect("observed VAE"),
                &audit::digest(vae.frozen_training_tensors())?,
            )?;
            for context in contexts
                .iter()
                .flatten()
                .chain(previews.iter().map(|(_, c)| c))
                .chain(negative.iter())
            {
                let tokens = context.dim(1)? as u64;
                probe.update(|r| r.actual_context_max = r.actual_context_max.max(tokens))?;
            }
        }
        self.device.synchronize()?;
        let encoder = if c.cache_text_embeddings {
            None
        } else {
            Some(encoder)
        };
        let vae = if c.cache_latents && previews.is_empty() {
            None
        } else {
            Some(vae)
        };
        estimate
            .check(admission::Phase::Cast, budget)
            .map_err(admission_error)?;
        let weights = load_weights_cancellable(
            &self.sources.assets(MossImageAcceleration::Off)?,
            MossImageAssetRole::Transformer,
            &self.device,
            compute_dtype(c)?,
            &[""],
            &req.cancel,
        )?;
        let mut transformer =
            Transformer::load(&weights, Geometry::from(&self.sources.config), None)?;
        #[cfg(feature = "test-support")]
        let frozen_base = if probe.is_some() {
            Some(audit::base_handles(&weights)?)
        } else {
            None
        };
        #[cfg(feature = "test-support")]
        let base_before = frozen_base
            .as_ref()
            .map(|values| audit::digest(values.iter().cloned()))
            .transpose()?;
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
        let (mut start, mut update) = (0, 0);
        let mut window = GradientWindow::default();
        let mut final_loss = 0.;
        if c.resume {
            estimate
                .check(admission::Phase::Checkpoint, budget)
                .map_err(admission_error)?;
            let (state, pending) =
                recovery::load_complete(req, stem, &plan, &fingerprint, &mut opt, &set)?;
            start = state.step;
            update = state.update;
            final_loss = state.final_loss;
            window = pending;
            transformer.validate_residuals()?;
        }
        let (updates, warmup) =
            schedule_updates(c.steps, c.gradient_accumulation, c.lr_warmup_steps);

        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            let factors = audit::factors(&set)?;
            let loss =
                evaluation
                    .as_ref()
                    .expect("heldout prepared")
                    .loss(&transformer, &set, req)?;
            probe.update(|r| {
                r.factor_before = factors;
                r.evaluation_before = Some(loss);
            })?;
        }
        for step in start + 1..=c.steps {
            check_cancel(&req.cancel)?;
            plan.verify_inputs()?;
            estimate
                .check(admission::Phase::Backward, budget)
                .map_err(admission_error)?;
            let mut batch_loss = 0.;
            let mut batch_tokens = 0u64;
            let draw = plan.batch(step - 1)?;
            for (offset, &entry_index) in draw.entries.iter().enumerate() {
                let geometry = &plan.entries[entry_index];
                let uncached;
                let (latent, mask) = if let Some(cached) = &cache[entry_index] {
                    (&cached.0, cached.1.as_ref())
                } else {
                    uncached = encode_entry(
                        req,
                        geometry,
                        &plan,
                        host_budget,
                        vae.as_ref().expect("uncached latents retain VAE"),
                        &self.device,
                    )?;
                    (&uncached.0, uncached.1.as_ref())
                };
                let fresh_context;
                let context = if let Some(context) = &contexts[geometry.item] {
                    context
                } else {
                    let item = &req.items[geometry.item];
                    fresh_context = text_encoder::strings(
                        &self.sources,
                        encoder.as_ref().expect("uncached text retains encoder"),
                        &[caption(&item.caption, c.trigger_word.as_deref())],
                        4096,
                        &self.device,
                        &req.cancel,
                    )?
                    .rows
                    .remove(0)
                    .detach();
                    &fresh_context
                };
                let index = draw
                    .sample_cursor
                    .checked_add(offset as u64)
                    .ok_or_else(|| CandleError::Msg("MOSS sample draw overflow".into()))?;
                let seed = c.seed.wrapping_add(index.wrapping_mul(2).wrapping_add(1));
                let noise = sample_noise(latent.dims(), seed, &self.device)?;
                let (loss, grads) = loss_grads(
                    &transformer,
                    &set,
                    c,
                    latent,
                    context,
                    &noise,
                    sigma(c, seed ^ 0xd1b54a32d192ed03)?,
                    mask,
                    &req.cancel,
                )?;
                #[cfg(feature = "test-support")]
                if let Some(probe) = &probe {
                    audit::gradients(probe, &grads, &set)?;
                    let tokens = context.dim(1)? as u64;
                    probe.update(|r| r.actual_context_max = r.actual_context_max.max(tokens))?;
                }
                #[cfg(feature = "test-support")]
                if let (Some(probe), Some(mask)) = (&probe, mask) {
                    audit::mask(probe, mask, latent)?;
                }
                let tokens = token_count(latent)?;
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
            #[cfg(feature = "test-support")]
            if let Some(probe) = &probe {
                probe.update(|r| r.microsteps.push(step))?;
            }
            progress(TrainingProgress::Training {
                step,
                total: c.steps,
                loss: final_loss,
            });
            if c.save_every > 0 && step % c.save_every == 0 && step != c.steps {
                estimate
                    .check(admission::Phase::Checkpoint, budget)
                    .map_err(admission_error)?;
                std::fs::create_dir_all(&req.output_dir)
                    .map_err(|e| CandleError::Msg(e.to_string()))?;
                save_with_source(
                    &transformer,
                    &set,
                    self.variant,
                    req,
                    &req.output_dir
                        .join(checkpoint::checkpoint_filename(stem, step)),
                    Some(&fingerprint),
                )?;
                let state = contract::recovery::RunState::after_step(
                    req,
                    &plan,
                    &fingerprint,
                    step,
                    update,
                    window.tokens,
                    final_loss,
                )?;
                recovery::save_complete(req, stem, &state, &opt, &set, &window)?;
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
                        let mut images = crate::pipeline::decode(
                            vae.as_ref().expect("preview retains VAE"),
                            &sampled.latents,
                            &request,
                            &mut |_| {},
                        )?;
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

        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            let loss =
                evaluation
                    .as_ref()
                    .expect("heldout prepared")
                    .loss(&transformer, &set, req)?;
            let factors = audit::factors(&set)?;
            probe.update(|r| {
                r.factor_after = factors;
                r.evaluation_after = Some(loss);
            })?;
            probe.frozen(
                "backbone",
                base_before.as_ref().expect("observed backbone"),
                &audit::digest(
                    frozen_base
                        .as_ref()
                        .expect("observed backbone")
                        .iter()
                        .cloned(),
                )?,
            )?;
            if let Some(encoder) = &encoder {
                probe.frozen(
                    "text_encoder",
                    te_before.as_ref().expect("observed encoder"),
                    &audit::digest(
                        encoder
                            .frozen_training_tensors()
                            .map_err(|e| CandleError::Msg(e.to_string()))?,
                    )?,
                )?;
            }
            if let Some(vae) = &vae {
                probe.frozen(
                    "vae",
                    vae_before.as_ref().expect("observed VAE"),
                    &audit::digest(vae.frozen_training_tensors())?,
                )?;
            }
        }
        check_cancel(&req.cancel)?;
        self.sources.ensure_unchanged()?;
        plan.verify_inputs()?;
        estimate
            .check(admission::Phase::Checkpoint, budget)
            .map_err(admission_error)?;
        progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir).map_err(|e| CandleError::Msg(e.to_string()))?;
        let path = req.output_dir.join(&req.file_name);
        save_with_source(
            &transformer,
            &set,
            self.variant,
            req,
            &path,
            Some(&fingerprint),
        )?;
        if c.save_every > 0 || c.resume {
            let state = contract::recovery::RunState::after_step(
                req,
                &plan,
                &fingerprint,
                c.steps,
                update,
                window.tokens,
                final_loss,
            )?;
            recovery::save_complete(req, stem, &state, &opt, &set, &window)?;
        }
        Ok(TrainingOutput {
            adapter_path: path,
            steps: c.steps,
            final_loss,
        })
    }
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
