//! Native MOSS transformer-only LoRA/LoKr. Frozen Qwen3.5 conditioning and full FLUX2 VAE
//! encoding are cached before functional autograd; only the explicit adapter-factor map is a leaf.
#[cfg(feature = "test-support")]
mod audit;
mod optimizer;
mod recovery;
#[cfg(feature = "test-support")]
use gen_core_testkit::moss_training_probe::{RunProbe, TrainingProbe};
pub(crate) use optimizer::MossOptimizer as TrainOptimizer;

use crate::training_memory::{
    self as admission, EstimateRequest, FootprintFacts, MemoryEstimate, Phase,
};
use crate::{
    loader::{read_weights, Sources},
    text_encoder::Encoder,
    transformer::Transformer,
};
use mlx_gen::gen_core::{self, moss_image::*, moss_image_training as contract};
use mlx_gen::train::{
    checkpoint,
    dataset::center_crop_square,
    lora::{accumulate_grads, build_lokr_targets, build_lora_targets, LoraParams, TrainAdapter},
    schedule::{lr_multiplier, schedule_updates},
};
use mlx_gen::{
    CancelFlag, Error, LoadSpec, Modality, NetworkType, Result, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
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

#[cfg(test)]
std::thread_local! {
    // Actual eager-loader entries on this test thread, not inferred retained trainer state.
    static EAGER_LOAD_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct LoadedTrainer {
    #[cfg(feature = "test-support")]
    test_probe: Option<RunProbe>,
    descriptor: TrainerDescriptor,
    variant: MossImageVariant,
    sources: Sources,
    encoder: Option<Encoder>,
    vae: Option<Flux2Vae>,
    transformer: Transformer,
    facts: FootprintFacts,
    budget_bytes: u64,
    shape: EstimateRequest,
    plan: contract::recovery::DataPlan,
    host_budget: u64,
    #[cfg(feature = "test-support")]
    evaluation_snapshot: Option<audit::Snapshot>,
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
    #[cfg(feature = "test-support")]
    test_probe: Option<TrainingProbe>,
    descriptor: TrainerDescriptor,
    variant: MossImageVariant,
    sources: Sources,
    facts: FootprintFacts,
    memory_budget_override: Option<u64>,
}
struct Preparation {
    plan: contract::recovery::DataPlan,
    shape: EstimateRequest,
    estimate: MemoryEstimate,
    budget: u64,
    host_budget: u64,
    #[cfg(feature = "test-support")]
    evaluation: Option<audit::Snapshot>,
}
fn admission_error(error: admission::AdmissionError) -> Error {
    Error::Unsupported(error.to_string())
}

fn budget(override_bytes: Option<u64>) -> Result<u64> {
    if override_bytes == Some(0) {
        return Ok(0);
    }
    let physical = admission::device_budget_bytes().map_err(admission_error)?;
    Ok(admission::resolve_budget(override_bytes, physical).min(physical))
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
        let sources = Sources::selected(variant, spec, MossImageAcceleration::Off)?;
        Self::from_sources(variant, sources)
    }
    pub(crate) fn from_sources(variant: MossImageVariant, sources: Sources) -> Result<Self> {
        let facts = FootprintFacts::from_sources(&sources).map_err(admission_error)?;
        Ok(Self {
            descriptor: descriptor(variant),
            variant,
            sources,
            facts,
            memory_budget_override: None,
            #[cfg(feature = "test-support")]
            test_probe: None,
        })
    }
    fn basic(&self, req: &TrainingRequest) -> Result<()> {
        validate_floors(&self.descriptor, req)?;
        contract::target_paths(
            self.sources.config.num_layers,
            &req.config.lora_target_modules,
        )?;
        self.sources.ensure_unchanged()?;
        crate::pipeline::check(&req.cancel)
    }
    fn prepare(&self, req: &TrainingRequest) -> Result<Preparation> {
        let available = contract::recovery::host::available_bytes()?;
        let total = req
            .config
            .memory_limit_fraction
            .map(|_| self.physical_total())
            .transpose()?;
        let budget = gen_core::train::effective_memory_budget_with_fraction(
            self.budget()?.min(available),
            total,
            req.config.memory_limit_bytes,
            req.config.memory_limit_fraction,
        )?;
        let minimal = self
            .facts
            .backbone_elements
            .checked_mul(2)
            .ok_or_else(|| Error::Unsupported("minimum backbone residency overflow".into()))?;
        if minimal > budget {
            return Err(Error::Unsupported(
                "frozen backbone alone exceeds physical unified/user memory cap".into(),
            ));
        }
        let host_budget = budget;
        contract::recovery::admit_conditioning_preprocess(
            req,
            host_budget,
            self.facts.host_metadata_bytes,
        )?;
        let plan = contract::recovery::DataPlan::from_request_budget(req, host_budget)?;
        let mut shape = EstimateRequest::from_request(req)
            .map_err(admission_error)?
            .with_plan(req, &plan)
            .map_err(admission_error)?;
        shape.context_tokens = crate::text_encoder::token_bound(&self.sources, req, host_budget)?;
        #[cfg(feature = "test-support")]
        let evaluation = self
            .test_probe
            .as_ref()
            .map(|probe| audit::Snapshot::new(req, probe.evaluation(), host_budget))
            .transpose()?;
        #[cfg(feature = "test-support")]
        if let Some(evaluation) = &evaluation {
            let tokens =
                crate::text_encoder::token_bound(&self.sources, &evaluation.request, host_budget)?;
            let geometry = &evaluation.plan.entries[0];
            shape.context_tokens = shape.context_tokens.max(tokens);
            shape.evaluation_context_tokens = tokens;
            shape.evaluation_canvas = Some((u64::from(geometry.width), u64::from(geometry.height)));
            shape.audit_workspace_bytes =
                gen_core_testkit::moss_training_probe::AUDIT_WORKSPACE_BYTES;
            shape.decode_bytes = shape
                .decode_bytes
                .max(evaluation.plan.decode_buffer_bound()?);
            shape.dataset_metadata_bytes = shape
                .dataset_metadata_bytes
                .checked_add(contract::recovery::DataPlan::metadata_bound(
                    &evaluation.request,
                )?)
                .ok_or_else(|| Error::Unsupported("evaluation metadata overflow".into()))?;
        }
        let estimate = admission::estimate(&self.facts, &shape).map_err(admission_error)?;
        estimate.check_all(budget).map_err(admission_error)?;
        contract::recovery::validate_resume_plan(
            req,
            &self.sources.source_identity(self.variant),
            &plan,
        )?;
        Ok(Preparation {
            plan,
            shape,
            estimate,
            budget,
            host_budget,
            #[cfg(feature = "test-support")]
            evaluation,
        })
    }
    fn physical_total(&self) -> Result<u64> {
        #[cfg(feature = "test-support")]
        if self.sources.is_fixture() {
            crate::test_support::ensure_cpu()?;
            // Explicit CPU reference envelope, not an MLX/Metal physical-memory measurement.
            return Ok(crate::test_support::HOST_CAP_BYTES);
        }
        contract::recovery::host::total_bytes().map_err(Into::into)
    }
    fn budget(&self) -> Result<u64> {
        #[cfg(feature = "test-support")]
        if self.sources.is_fixture() {
            crate::test_support::ensure_cpu()?;
            return Ok(admission::resolve_budget(
                self.memory_budget_override,
                crate::test_support::HOST_CAP_BYTES,
            )
            .min(crate::test_support::HOST_CAP_BYTES));
        }
        budget(self.memory_budget_override)
    }
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
            serde_json::json!({"load_bytes":estimate.load,"cast_bytes":estimate.cast,"cache_bytes":estimate.cache,"backward_bytes":estimate.backward,"preview_bytes":estimate.preview,"checkpoint_bytes":estimate.checkpoint,"encoder_only_peak_bytes":admission::encoder_work(&self.facts.encoder_config,4096).map_err(admission_error)?.checked_add(self.facts.encoder_bytes).and_then(|n|n.checked_add(self.facts.largest_source_shard)).ok_or_else(||Error::Unsupported("encoder-only peak overflow".into()))?,"host_staging_peak_bytes":0,"unified_memory":true,"tensor_peak_bytes":estimate.peak(),"context_tokens":estimate.context_token_bound,"source_shard_bytes":self.facts.largest_source_shard,"encoder_resident_bytes":self.facts.encoder_bytes,"vae_resident_bytes":self.facts.vae_bytes,"trainable_elements":estimate.trainable_elements,"cache_bytes_persistent":estimate.cache_bytes}),
        )
    }
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
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
            .map(|p| p.begin(self.sources.is_fixture()))
            .transpose()?;
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.bind_request(
                req,
                &prepared.plan,
                &self.sources.source_identity(self.variant),
            )?;
        }
        let result = (|| {
            crate::pipeline::check(&req.cancel)?;
            let estimate = prepared.estimate.clone();
            let run_budget = prepared.budget;
            let mut loaded = admitted_load(&estimate, run_budget, || {
                load_weights(
                    self.sources.clone(),
                    self.variant,
                    self.facts.clone(),
                    prepared,
                    &req.cancel,
                )
            })?;
            #[cfg(feature = "test-support")]
            {
                loaded.test_probe = probe.clone();
            }
            loaded.train(req, progress)
        })();
        let cleanup = self.sources.drain_cache().map_err(gen_core::Error::from);
        let result = match cleanup {
            Ok(()) => result,
            Err(error) => Err(error),
        };
        #[cfg(feature = "test-support")]
        if let Some(probe) = probe {
            probe.finish(result.as_ref().err().map(ToString::to_string))?;
        }
        result
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
    prepared: Preparation,
    cancel: &CancelFlag,
) -> Result<LoadedTrainer> {
    #[cfg(test)]
    EAGER_LOAD_ENTRIES.with(|entries| entries.set(entries.get() + 1));
    crate::pipeline::check(cancel)?;
    sources.clear_cache()?;
    let encoder = Encoder::load(&sources, cancel)?;
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
        cancel,
    )?;
    let vae = Flux2Vae::from_weights_with_decoder_config(
        &weights,
        &Flux2VaeDecoderConfig {
            block_out_channels: if sources.is_fixture() {
                [32; 4]
            } else {
                [96, 192, 384, 384]
            },
        },
    )?;
    weights.materialize_accessed()?;
    if !weights.unused_keys().is_empty() {
        return Err(Error::Unsupported(format!(
            "unconsumed MOSS training VAE tensors {:?}",
            weights.unused_keys()
        )));
    }
    let transformer = sources.transformer(MossImageAcceleration::Off, cancel)?;
    sources.ensure_unchanged()?;
    Ok(LoadedTrainer {
        #[cfg(feature = "test-support")]
        test_probe: None,
        facts,
        budget_bytes: prepared.budget,
        shape: prepared.shape,
        plan: prepared.plan,
        host_budget: prepared.host_budget,
        #[cfg(feature = "test-support")]
        evaluation_snapshot: prepared.evaluation,
        descriptor: descriptor(variant),
        variant,
        sources,
        encoder: Some(encoder),
        vae: Some(vae),
        transformer,
    })
}
fn validate_floors(d: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
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
    for (k, v) in &c.model_options {
        if k != "vae_posterior" || v.as_str() != Some(contract::POSTERIOR) {
            return Err(Error::Unsupported(format!(
                "unsupported MOSS training option {k}={v}; posterior policy is {}",
                contract::POSTERIOR
            ))
            .into());
        }
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
#[cfg(test)]
pub(super) fn save(
    model: &mut Transformer,
    adapter: &TrainAdapter,
    params: &LoraParams,
    variant: MossImageVariant,
    req: &TrainingRequest,
    path: &Path,
) -> Result<()> {
    save_with_source(model, adapter, params, variant, req, path, None)
}
fn save_with_source(
    model: &mut Transformer,
    adapter: &TrainAdapter,
    params: &LoraParams,
    variant: MossImageVariant,
    req: &TrainingRequest,
    path: &Path,
    identity: Option<&contract::recovery::RunIdentity>,
) -> Result<()> {
    let cfg = &req.config;
    let mut metadata = contract::metadata(variant, sampler_name(cfg));
    if let Some(identity) = identity {
        metadata.insert(
            "source_checkpoint_identity".into(),
            identity.source_checkpoint.clone(),
        );
        metadata.insert(
            "dataset_fingerprint".into(),
            identity.dataset_fingerprint.clone(),
        );
        metadata.insert(
            "effective_training_controls".into(),
            identity.effective_config.clone(),
        );
        metadata.insert(
            "training_stream_version".into(),
            contract::recovery::STREAM_VERSION.into(),
        );
    }
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
fn token_count(latent: &Array) -> Result<u64> {
    let shape = latent.shape();
    if shape.len() != 4 || shape[2] <= 0 || shape[3] <= 0 {
        return Err(Error::Msg(
            "MOSS loss requires a nonempty NCHW latent grid".into(),
        ));
    }
    (shape[2] as u64)
        .checked_mul(shape[3] as u64)
        .ok_or_else(|| Error::Msg("MOSS token count overflow".into()))
}
fn token_weighted_grads(grads: LoraParams, tokens: u64) -> Result<LoraParams> {
    Ok(grads
        .into_iter()
        .map(|(key, value)| Ok((key, multiply(value, Array::from_f32(tokens as f32))?)))
        .collect::<MlxResult<_>>()?)
}

fn average_grads(grads: LoraParams, tokens: u64) -> Result<LoraParams> {
    if tokens == 0 {
        return Err(Error::Msg("MOSS empty token denominator".into()));
    }
    Ok(grads
        .into_iter()
        .map(|(key, value)| Ok((key, multiply(value, Array::from_f32(1. / tokens as f32))?)))
        .collect::<MlxResult<_>>()?)
}
fn encode_entry(
    req: &TrainingRequest,
    geometry: &contract::recovery::EntryGeometry,
    plan: &contract::recovery::DataPlan,
    host_budget: u64,
    vae: &Flux2Vae,
) -> Result<(Array, Option<Array>)> {
    crate::pipeline::check(&req.cancel)?;
    let source = plan.decode_rgb(geometry.item, host_budget)?;
    let latent = if req.config.aspect_ratio_buckets.is_empty() {
        let image = center_crop_square(&mlx_gen::media::Image {
            width: source.width(),
            height: source.height(),
            pixels: source.into_raw(),
        });
        encode_image(vae, &image, geometry.width)?
    } else {
        let row = gen_core::AspectRatioBucket {
            width: geometry.width,
            height: geometry.height,
            repeats: 1,
        };
        let pixels = gen_core::train::aspect_ratio::resize_rgb(&source, row)?;
        let pixels = Array::from_slice(
            &pixels,
            &[1, geometry.height as i32, geometry.width as i32, 3],
        );
        let normalized = pixels
            .divide(Array::from_f32(127.5))?
            .subtract(Array::from_f32(1.))?;
        vae.bn_normalize_nchw(&pack_mean(&vae.encode_mean(&normalized)?)?)?
    };
    eval([&latent])?;
    crate::pipeline::check(&req.cancel)?;
    if !latent.is_finite()?.all(None)?.item::<bool>() {
        return Err(Error::Unsupported("MOSS nonfinite encoded latent".into()));
    }
    let mask = plan.subject_mask(req, geometry.item, host_budget)?;
    let mask = mlx_gen::train::loss::prepared_subject_mask_weight(
        "MOSS",
        mask.as_ref(),
        |_, _| geometry.crop_box(),
        latent.shape(),
    )?;
    if req.config.subject_mask_loss.is_some() && mask.is_none() {
        return Err(Error::Unsupported(
            "MOSS configured subject mask was not produced".into(),
        ));
    }
    Ok((latent, mask))
}
impl Trainer for LoadedTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_floors(&self.descriptor, req)?;
        contract::target_paths(
            self.sources.config.num_layers,
            &req.config.lora_target_modules,
        )?;
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
impl LoadedTrainer {
    fn admit(&self, _req: &TrainingRequest, phase: Phase) -> Result<()> {
        admission::estimate(&self.facts, &self.shape)
            .map_err(admission_error)?
            .check(phase, self.budget_bytes)
            .map_err(admission_error)
    }
    fn admit_image(&self, req: &TrainingRequest, _path: &Path) -> Result<()> {
        self.plan.verify_inputs()?;
        self.admit(req, Phase::Cache)
    }
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let c = &req.config;
        #[cfg(feature = "test-support")]
        let probe = self.test_probe.clone();
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.update(|r| r.context_bound = self.shape.context_tokens)?;
        }

        let plan = &self.plan;
        plan.verify_inputs()?;
        let fingerprint = contract::recovery::effective_identity(
            req,
            &self.sources.source_identity(self.variant),
            plan,
        )?;
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
        self.transformer
            .cast_training_weights_with_cancel(dtype, &req.cancel)?;
        progress(TrainingProgress::Preparing);
        progress(TrainingProgress::LoadingModel);
        self.admit(req, Phase::Cache)?;

        #[cfg(feature = "test-support")]
        let base_before = if probe.is_some() {
            Some(audit::digest(
                self.transformer.frozen_training_arrays(),
                &req.cancel,
            )?)
        } else {
            None
        };
        #[cfg(feature = "test-support")]
        let te_before = if probe.is_some() {
            Some(audit::digest(
                self.encoder
                    .as_ref()
                    .expect("run encoder")
                    .frozen_training_arrays(),
                &req.cancel,
            )?)
        } else {
            None
        };
        #[cfg(feature = "test-support")]
        let vae_before = if probe.is_some() {
            Some(audit::digest(
                self.vae
                    .as_ref()
                    .expect("run VAE")
                    .frozen_training_arrays()?,
                &req.cancel,
            )?)
        } else {
            None
        };
        #[cfg(feature = "test-support")]
        let evaluation = probe
            .as_ref()
            .map(|p| {
                audit::prepare(
                    &self.sources,
                    self.evaluation_snapshot
                        .as_ref()
                        .expect("admitted heldout snapshot"),
                    p,
                    self.encoder.as_ref().expect("run encoder"),
                    self.vae.as_ref().expect("run VAE"),
                    self.host_budget,
                )
            })
            .transpose()?;
        let mut cache = Vec::with_capacity(plan.entries.len());
        let mut contexts = Vec::with_capacity(req.items.len());
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("MOSS missing run-owned encoder".into()))?;
        for (i, item) in req.items.iter().enumerate() {
            crate::pipeline::check(&req.cancel)?;
            progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total: req.items.len() as u32,
            });
            contexts.push(if c.cache_text_embeddings {
                Some(
                    encoder
                        .encode(
                            &caption(&item.caption, c.trigger_word.as_deref()),
                            4096,
                            &req.cancel,
                        )?
                        .0,
                )
            } else {
                None
            });
            for geometry in plan
                .entries
                .iter()
                .filter(|geometry| geometry.item == contexts.len() - 1)
            {
                crate::pipeline::check(&req.cancel)?;
                self.admit_image(req, &req.items[geometry.item].image_path)?;
                cache.push(if c.cache_latents {
                    Some(encode_entry(
                        req,
                        geometry,
                        plan,
                        self.host_budget,
                        self.vae.as_ref().expect("loaded VAE"),
                    )?)
                } else {
                    None
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
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.frozen(
                "text_encoder",
                te_before.as_ref().expect("observed encoder"),
                &audit::digest(
                    self.encoder
                        .as_ref()
                        .expect("run encoder")
                        .frozen_training_arrays(),
                    &req.cancel,
                )?,
            )?;
            probe.frozen(
                "vae",
                vae_before.as_ref().expect("observed VAE"),
                &audit::digest(
                    self.vae
                        .as_ref()
                        .expect("run VAE")
                        .frozen_training_arrays()?,
                    &req.cancel,
                )?,
            )?;
            for context in contexts
                .iter()
                .flatten()
                .chain(previews.iter().map(|(_, c)| c))
                .chain(negative.iter())
            {
                let tokens = u64::try_from(context.shape()[1])
                    .map_err(|_| Error::Msg("invalid conditioning shape".into()))?;
                probe.update(|r| r.actual_context_max = r.actual_context_max.max(tokens))?;
            }
        }
        if c.cache_text_embeddings {
            self.encoder = None;
        }
        if c.cache_latents && previews.is_empty() {
            self.vae = None;
        }
        self.sources.clear_cache()?;
        self.admit(req, Phase::Backward)?;
        let paths = contract::target_paths(self.sources.config.num_layers, &c.lora_target_modules)?;
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
        let stem = req
            .file_name
            .strip_suffix(".safetensors")
            .ok_or_else(|| Error::Msg("invalid adapter filename".into()))?;
        let (mut start, mut update) = (0, 0);
        let mut accumulated = None;
        let mut accumulated_tokens = 0u64;
        let mut final_loss = 0.;
        if c.resume {
            self.admit(req, Phase::Checkpoint)?;
            let (state, loaded, pending) =
                recovery::load_complete(req, stem, plan, &fingerprint, &mut opt, &params)?;
            params = loaded;
            accumulated = pending;
            accumulated_tokens = state.accumulated_tokens;
            start = state.step;
            update = state.update;
            final_loss = state.final_loss;
        }
        let mut steps = start;

        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            if let Some(target) = probe.run_start_export_target()? {
                self.admit(req, Phase::Checkpoint)?;
                save_with_source(
                    &mut self.transformer,
                    &adapter,
                    &params,
                    self.variant,
                    req,
                    &target,
                    Some(&fingerprint),
                )?;
                probe.record_run_start_export(&target, start, update, self.variant)?;
            }
            let factors = audit::factors(&params, &req.cancel)?;
            let loss = evaluation.as_ref().expect("heldout prepared").loss(
                &mut self.transformer,
                &params,
                &adapter,
                req,
            )?;
            let observation = evaluation
                .as_ref()
                .expect("heldout prepared")
                .observation(loss, &req.cancel)?;
            probe.update(|r| {
                r.factor_before = factors;
                r.evaluation_before = Some(loss);
                r.prepared_evaluation
                    .get_or_insert_with(Default::default)
                    .before = Some(observation);
            })?;
        }
        for step in start + 1..=c.steps {
            crate::pipeline::check(&req.cancel)?;
            plan.verify_inputs()?;
            self.admit(req, Phase::Backward)?;
            let mut micro = None;
            let mut total_loss = 0.;
            let mut total_tokens = 0u64;
            let draw = plan.batch(step - 1)?;
            for (offset, &entry_index) in draw.entries.iter().enumerate() {
                let geometry = &plan.entries[entry_index];
                let uncached;
                let (latent, mask) = if let Some(cached) = &cache[entry_index] {
                    (&cached.0, cached.1.as_ref())
                } else {
                    self.admit_image(req, &req.items[geometry.item].image_path)?;
                    uncached = encode_entry(
                        req,
                        geometry,
                        plan,
                        self.host_budget,
                        self.vae.as_ref().expect("uncached latents retain VAE"),
                    )?;
                    (&uncached.0, uncached.1.as_ref())
                };
                let fresh_context;
                let context = if let Some(context) = &contexts[geometry.item] {
                    context
                } else {
                    let item = &req.items[geometry.item];
                    fresh_context = self
                        .encoder
                        .as_ref()
                        .expect("uncached text retains encoder")
                        .encode(
                            &caption(&item.caption, c.trigger_word.as_deref()),
                            4096,
                            &req.cancel,
                        )?
                        .0;
                    &fresh_context
                };
                let index = draw
                    .sample_cursor
                    .checked_add(offset as u64)
                    .ok_or_else(|| Error::Msg("MOSS sample draw overflow".into()))?;
                let seed = c.seed.wrapping_add(index.wrapping_mul(2).wrapping_add(1));
                let noise =
                    random::normal::<f32>(latent.shape(), None, None, Some(&random::key(seed)?))?;
                let (loss, grads) = loss_grads(
                    &mut self.transformer,
                    &params,
                    &adapter,
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
                    audit::gradients(probe, &grads)?;
                    let tokens = u64::try_from(context.shape()[1])
                        .map_err(|_| Error::Msg("invalid conditioning shape".into()))?;
                    probe.update(|r| r.actual_context_max = r.actual_context_max.max(tokens))?;
                }
                #[cfg(feature = "test-support")]
                if let (Some(probe), Some(mask)) = (&probe, mask) {
                    audit::mask(probe, mask, latent)?;
                }
                let tokens = token_count(latent)?;
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
                opt.update(&mut params, &avg, c, update, c.seed)?;
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
                self.admit(req, Phase::Checkpoint)?;
                std::fs::create_dir_all(&req.output_dir)?;
                save_with_source(
                    &mut self.transformer,
                    &adapter,
                    &params,
                    self.variant,
                    req,
                    &req.output_dir
                        .join(checkpoint::checkpoint_filename(stem, step)),
                    Some(&fingerprint),
                )?;
                let state = contract::recovery::RunState::after_step(
                    req,
                    plan,
                    &fingerprint,
                    step,
                    update,
                    accumulated_tokens,
                    final_loss,
                )?;
                recovery::save_complete(req, stem, &state, &opt, &params, accumulated.as_ref())?;
                #[cfg(feature = "test-support")]
                if let Some(probe) = &probe {
                    probe.record_checkpoint(
                        &contract::recovery::resume_bundle_path(
                            &req.output_dir,
                            &req.file_name,
                            step,
                        )?,
                        &req.output_dir
                            .join(checkpoint::checkpoint_filename(stem, step)),
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

        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            let loss = evaluation.as_ref().expect("heldout prepared").loss(
                &mut self.transformer,
                &params,
                &adapter,
                req,
            )?;
            let factors = audit::factors(&params, &req.cancel)?;
            let observation = evaluation
                .as_ref()
                .expect("heldout prepared")
                .observation(loss, &req.cancel)?;
            probe.update(|r| {
                r.factor_after = factors;
                r.evaluation_after = Some(loss);
                r.prepared_evaluation
                    .get_or_insert_with(Default::default)
                    .after = Some(observation);
            })?;
            probe.frozen(
                "backbone",
                base_before.as_ref().expect("observed backbone"),
                &audit::digest(self.transformer.frozen_training_arrays(), &req.cancel)?,
            )?;
            if let Some(encoder) = &self.encoder {
                probe.frozen(
                    "text_encoder",
                    te_before.as_ref().expect("observed encoder"),
                    &audit::digest(encoder.frozen_training_arrays(), &req.cancel)?,
                )?;
            }
            if let Some(vae) = &self.vae {
                probe.frozen(
                    "vae",
                    vae_before.as_ref().expect("observed VAE"),
                    &audit::digest(vae.frozen_training_arrays()?, &req.cancel)?,
                )?;
            }
        }
        plan.verify_inputs()?;
        if steps == 0 {
            return Err(Error::Canceled);
        }
        crate::pipeline::check(&req.cancel)?;
        self.admit(req, Phase::Checkpoint)?;
        progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let path = req.output_dir.join(&req.file_name);
        save_with_source(
            &mut self.transformer,
            &adapter,
            &params,
            self.variant,
            req,
            &path,
            Some(&fingerprint),
        )?;
        if c.save_every > 0 || c.resume {
            let state = contract::recovery::RunState::after_step(
                req,
                plan,
                &fingerprint,
                c.steps,
                update,
                accumulated_tokens,
                final_loss,
            )?;
            recovery::save_complete(req, stem, &state, &opt, &params, accumulated.as_ref())?;
            #[cfg(feature = "test-support")]
            if let Some(probe) = &probe {
                probe.record_checkpoint(
                    &contract::recovery::resume_bundle_path(
                        &req.output_dir,
                        &req.file_name,
                        c.steps,
                    )?,
                    &path,
                )?;
            }
        }
        #[cfg(feature = "test-support")]
        if let Some(probe) = &probe {
            probe.record_final_export(&path)?;
        }
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
                .as_ref()
                .expect("preview retains VAE")
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
    fn public_trainer_factories_require_backbone_before_shared_assets() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("parent");
        let shared = temp.path().join("shared");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&shared).unwrap();
        let unrelated = root.join("prepared-literal");
        std::fs::write(&unrelated, b"source identity only").unwrap();

        for variant in [MossImageVariant::Base, MossImageVariant::Sft] {
            let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(root.clone()));
            spec.components.insert(
                MOSS_SHARED_SNAPSHOT_COMPONENT.into(),
                gen_core::WeightsSource::Dir(shared.clone()),
            );
            spec.prepare_with_file_pins([gen_core::PinnedWeightsFile::pin(&unrelated).unwrap()])
                .unwrap();

            // Generation still defers the backbone until a request selects its acceleration.
            let deferred = Sources::prepare(variant, &spec).err().unwrap().to_string();
            assert!(
                deferred.contains(
                    &shared
                        .join("text_encoder/config.json")
                        .display()
                        .to_string()
                ),
                "{variant:?}: {deferred}"
            );

            // Exercise the actual registry factories, not synthetic from_sources fixtures.
            // Missing prepared identity fails before a header read, device query or model load.
            EAGER_LOAD_ENTRIES.with(|entries| entries.set(0));
            let result = match variant {
                MossImageVariant::Base => load_base(&spec),
                MossImageVariant::Sft => load_sft(&spec),
            };
            let error = result.err().expect("backbone pin is absent").to_string();
            assert!(error.contains("prepared MOSS identity missing"), "{error}");
            assert!(
                error.contains(&root.join("transformer/config.json").display().to_string()),
                "{variant:?}: {error}"
            );
            EAGER_LOAD_ENTRIES.with(|entries| assert_eq!(entries.get(), 0));
        }
    }

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
    fn resume_accepts_checkpoint_cadence_inside_accumulation_window() {
        let request = TrainingRequest {
            items: vec![gen_core::TrainingItem::captioned(
                "not-opened.png".into(),
                "caption".into(),
            )],
            config: TrainingConfig {
                resolution: 32,
                rank: 1,
                steps: 6,
                gradient_accumulation: 2,
                save_every: 1,
                resume: true,
                ..Default::default()
            },
            output_dir: "not-created".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        };
        assert!(validate_floors(&descriptor(MossImageVariant::Base), &request).is_ok(),
            "complete recovery stores pending gradients, token denominator and cursor; cadence need not align to updates");
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
            #[cfg(feature = "test-support")]
            test_probe: None,
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
        EAGER_LOAD_ENTRIES.with(|entries| entries.set(0));
        let result = trainer.train(&request, &mut |_| {
            panic!("no model/cache/progress before refusal")
        });
        assert!(
            matches!(result,Err(gen_core::Error::Unsupported(message)) if message == "training memory budget must be positive"),
            "zero override must produce the typed positive-budget refusal before missing assets"
        );
        assert_eq!(
            EAGER_LOAD_ENTRIES.with(std::cell::Cell::get),
            0,
            "refused request entered the actual eager tensor loader"
        );
        assert!(!output.exists());
    }
}
