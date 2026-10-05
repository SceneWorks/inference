//! Native Candle training for the Mage-Flow base. Adapter runs train the exact dotted projection
//! surface consumed by inference; full runs use a separate owned-parameter loader and publish a
//! complete reloadable transformer component, never an adapter-shaped artifact.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    self, BucketSchedule, LoadSpec, Modality, NetworkType, Precision, WeightsSource,
};
use candle_gen::quant::AdaptLinear;
use candle_gen::train::checkpoint::{checkpoint_filename, file_stem};
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor};
use candle_gen::train::flow_match::{
    self, check_aux_memory, combine_terms, component_bytes, device_training_budget_bytes,
    effective_weight_decay, noise_seed, prepared_subject_mask_weight, sample_noise, save_adapter,
    step_sample, step_terms, validate_flow_match_request, weighted_velocity_loss, AuxDriver,
    AuxStep, StepLosses,
};
use candle_gen::train::lora::{
    adapter_optimizer_step, build_adapt_lokr_targets, build_adapt_lora_targets, AdaptLoraHost,
    LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen::train::perceptual::{AuxModelFootprint, Parameterization, X0Decoder};
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
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets — one cached latent per (item, bucket).
        // sc-24828 (epic 2123): subject-masked loss — a per-bucket weight map, packed like that
        // bucket's latent, cached next to it.
        // sc-24830 (epic 2123): depth anchoring on the adapter surface through the shared
        // perceptual builder; Mage-VAE has no tiny decoder, so x0 decodes through the full Mage-VAE
        // decoder ([`MageX0Decoder`]). The full fine-tune surface refuses it.
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

/// The full Mage-VAE decoder as the perceptual path's x0 decoder (epic 2123 E8, sc-24830): Mage's
/// 128-channel / 16× latent space has no tiny decoder. Runs [`MageVae::decode_differentiable`]
/// (every op has a candle backward) and maps the raw `[-1, 1]` RGB to NHWC `[0, 1]`. Candle has no
/// activation-checkpoint primitive for a graph inside one `backward()`, so the decode's activations
/// stay on the tape until the step's backward; [`MageDecoderSpec::footprint`] budgets for that.
pub struct MageX0Decoder {
    vae: MageVae,
}

impl MageX0Decoder {
    /// Wrap a loaded Mage-VAE (its decoder half is used).
    pub fn new(vae: MageVae) -> Self {
        Self { vae }
    }
}

impl X0Decoder for MageX0Decoder {
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let rgb = self
            .vae
            .decode_differentiable(latents)?
            .to_dtype(DType::F32)?;
        Ok(rgb
            .affine(0.5, 0.5)?
            .clamp(0f32, 1f32)?
            .permute((0, 2, 3, 1))?
            .contiguous()?)
    }
}

/// Mage's measured decode curve (decimal GB; `mlx-gen-mage`'s `memory::vae_peak_gb`): a fixed term
/// plus a per-megapixel term.
fn mage_vae_peak_gb(h: u32, w: u32) -> f64 {
    0.267 + 2.039 * (h as f64 * w as f64 / 1e6)
}

/// [`MageX0Decoder`] for the shared aux-loss builder, loaded (f32) from the trainer's own resolved
/// VAE directory only when an enabled loss decodes pixels. Mage has no separately cataloged x0
/// decoder: `TrainingConfig::perceptual_decoder_dir` (which the shared floor requires) names the
/// base snapshot and is not read here — the split-tier mirror can stage the VAE elsewhere, and the
/// trainer already resolved where.
struct MageDecoderSpec {
    vae_dir: PathBuf,
}

impl candle_gen_perceptual::CustomDecoder for MageDecoderSpec {
    fn name(&self) -> &'static str {
        "Mage-VAE decoder"
    }

    /// Conservative pre-load figures from Mage's measured inference decode curve: f32 weights at
    /// twice the bf16 fixed term, and a working set of 4× the f32 decode peak (no activation
    /// checkpointing on candle: the decode's activations and their cotangents are both live in the
    /// backward). Not a measured training value.
    fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint {
        let gb = |v: f64| (v * 1e9) as u64;
        AuxModelFootprint {
            param_bytes: gb(2.0 * 0.267),
            working_set_bytes: gb(4.0 * 2.0 * mage_vae_peak_gb(h, w)),
            reference_bytes_per_image: 0,
        }
    }

    fn load(&self, _dir: Option<&Path>, device: &Device) -> Result<Box<dyn X0Decoder>> {
        Ok(Box::new(MageX0Decoder::new(MageVae::load_full_dtype(
            &self.vae_dir,
            device,
            DType::F32,
        )?)))
    }
}

/// Mage's latent family for the shared aux-loss builder (epic 2123 E8).
fn aux_loss_context<'a>(
    device: &'a Device,
    vae_dir: &Path,
) -> candle_gen_perceptual::AuxLossContext<'a> {
    candle_gen_perceptual::AuxLossContext {
        label: LABEL,
        decoder: candle_gen_perceptual::DecoderSpec::Custom(Box::new(MageDecoderSpec {
            vae_dir: vae_dir.to_path_buf(),
        })),
        device,
        latent_lpips: None,
    }
}

/// Extra training memory (bytes) the enabled aux losses add at the largest bucket `edge` over
/// `entries` cached (item, bucket) references (epic 2123 E7). `0` when none is enabled.
fn perceptual_footprint_bytes(
    cfg: &gen_core::train::TrainingConfig,
    edge: u32,
    entries: usize,
) -> u64 {
    candle_gen_perceptual::perceptual_footprint(
        cfg,
        &candle_gen_perceptual::DecoderSpec::Custom(Box::new(MageDecoderSpec {
            vae_dir: PathBuf::new(),
        })),
        candle_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// The perceptual aux losses train the adapter surface only; the full fine-tune has no aux seam, so
/// the combination is a typed refusal (never silently ignored).
fn refuse_aux_losses_on_full_finetune(req: &TrainingRequest) -> gen_core::Result<()> {
    if req.config.full_finetune && candle_gen_perceptual::any_aux_loss(&req.config) {
        return Err(gen_core::Error::Unsupported(format!(
            "{LABEL}: depth anchoring / perceptual aux losses train a LoRA/LoKr adapter only; \
             they cannot be combined with a full base fine-tune"
        )));
    }
    Ok(())
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
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        refuse_aux_losses_on_full_finetune(req)?;
        validate_request(req).map_err(Into::into)
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
    /// Subject-mask loss weight packed exactly like `latent` (`[1, grid², C]`); `None` unless
    /// subject-masked loss is on (sc-24828).
    mask_weight: Option<Tensor>,
}

/// Pack a `[1, C, grid, grid]` latent-grid tensor into Mage's `[1, grid², C]` token sequence — the
/// layout the cached latent (and so the velocity target) uses. Shared with the subject-mask loss
/// weight so it lines up element-for-element with the latent it multiplies.
fn pack_latent_tokens(latent: &Tensor, grid: usize, channels: usize) -> Result<Tensor> {
    Ok(latent
        .permute((0, 2, 3, 1))?
        .reshape((1, grid * grid, channels))?)
}

/// The subject-mask loss weight for one bucket's cached latent (sc-24828 × sc-2127), `None` when
/// masked loss is off. The item's mask (loaded once) takes [`decode_square`]'s centre-square crop,
/// is area-averaged onto this bucket's unpacked `[1, C, grid, grid]` latent grid, and is packed with
/// [`pack_latent_tokens`] exactly like that bucket's latent.
fn bucket_mask_weight(
    mask: Option<&PreparedSubjectMask>,
    grid: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    prepared_subject_mask_weight(
        LABEL,
        mask,
        CropBox::center_square,
        &[1, config::LATENT_CHANNELS, grid, grid],
        device,
    )?
    .map(|weight| pack_latent_tokens(&weight, grid, config::LATENT_CHANNELS))
    .transpose()
}

/// One step's flow-match loss over a cached sample: noise the packed latent at `sigma`, predict the
/// velocity through the transformer (adapter or full surface alike), regress it toward
/// `noise − latent`, weighted by the sample's subject-mask weight (`None` ⇒ exactly the unweighted
/// `velocity_loss`).
///
/// `aux` (epic 2123 E8) carries the step's perceptual plan: on an aux-only step the velocity term is
/// not computed and the loss is the weighted aux term on the x0 estimate `x_t − σ·v` (Mage regresses
/// `noise − x0`); with `aux = None` the graph is exactly the pre-epic-2123 one.
fn step_loss(
    transformer: &MageTransformer,
    sample: &CachedSample,
    noise: &Tensor,
    sigma: f64,
    compute_dtype: DType,
    mae: bool,
    aux: Option<&AuxStep<'_>>,
) -> Result<(Tensor, StepLosses)> {
    let (x_t, target) = build_training_batch(&sample.latent, noise, sigma, compute_dtype)?;
    let sigma_tensor = Tensor::new(&[sigma as f32], noise.device())?;
    let prediction = transformer.forward(
        &x_t,
        &sample.text.to_dtype(compute_dtype)?,
        &sigma_tensor,
        &sample.layout,
    )?;
    let (diffusion_on, aux_on) = step_terms(aux);
    let diffusion = if diffusion_on {
        Some(weighted_velocity_loss(
            &prediction,
            &target,
            sample.mask_weight.as_ref(),
            mae,
        )?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let x0 = Parameterization::FlowNoiseMinusX0 {
                sigma: sigma as f32,
            }
            .recover_x0(
                &x_t.to_dtype(DType::F32)?,
                &prediction.to_dtype(DType::F32)?,
            )?;
            a.aux_loss(&tokens_to_latent_grid(&x0, sample_grid(sample)?)?)?
        }
        _ => None,
    };
    combine_terms(diffusion, aux_term)
}

/// The square latent grid side of a cached sample (`[1, grid², C]` tokens).
fn sample_grid(sample: &CachedSample) -> Result<usize> {
    let tokens = sample.latent.dim(1)?;
    let grid = (tokens as f64).sqrt().round() as usize;
    if grid * grid != tokens {
        return Err(CandleError::Msg(format!(
            "{LABEL}: cached latent has {tokens} tokens, not a square grid"
        )));
    }
    Ok(grid)
}

/// Mage tokens `[1, grid², C]` → the decoder's NCHW grid `[1, C, grid, grid]` (inverse of
/// [`pack_latent_tokens`]).
fn tokens_to_latent_grid(tokens: &Tensor, grid: usize) -> Result<Tensor> {
    let channels = tokens.dim(2)?;
    Ok(tokens
        .reshape((1, grid, grid, channels))?
        .permute((0, 3, 1, 2))?
        .contiguous()?)
}

/// One micro-step on the 1-based `step`: the schedule's (item, entry), its seeded σ + noise (as
/// before epic 2123), the perceptual plan when a driver is configured (an aux-only step trains at σ
/// remapped into the loss window), then [`step_loss`]. With no driver the step is bit-identical.
#[allow(clippy::too_many_arguments)]
fn run_step(
    transformer: &MageTransformer,
    cache: &[CachedSample],
    schedule: &BucketSchedule,
    aux: Option<&mut AuxDriver>,
    cfg: &gen_core::train::TrainingConfig,
    step: u32,
    compute_dtype: DType,
    mae: bool,
    device: &Device,
) -> Result<(Tensor, StepLosses)> {
    let picked = step_sample(aux, step, schedule);
    let sample = &cache[picked.entry];
    let raw_sigma = flow_match::sample_unit_timestep(
        &cfg.timestep_type,
        &cfg.timestep_bias,
        flow_match::timestep_seed(cfg.seed, step),
    );
    let plan = picked.plan(raw_sigma)?;
    let sigma = plan.as_ref().map_or(raw_sigma, |p| p.noise_level()) as f64;
    let noise = sample_noise(sample.latent.dims(), noise_seed(cfg.seed, step), device)?;
    step_loss(
        transformer,
        sample,
        &noise,
        sigma,
        compute_dtype,
        mae,
        plan.as_ref(),
    )
}

/// The packed latent grid side and the single-image generation layout for one bucket `edge` and a
/// `text_len`-token caption (sc-2127): every bucket's cached latent carries the layout of its own
/// grid, so the MSRoPE table built per step always matches the token count it is applied to.
fn bucket_layout(edge: u32, text_len: usize) -> Result<(usize, PackLayout)> {
    let grid = edge as usize / VAE_DOWNSAMPLE;
    let layout = PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![text_len])?;
    Ok((grid, layout))
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
    // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are off);
    // each bucket packs its own latent grid.
    let edges = bucket_edges(&req.config);
    // Item-major: `cache[item * edges.len() + bucket]` (sc-2127).
    let mut cache = Vec::with_capacity(req.items.len() * edges.len());
    for (index, item) in req.items.iter().enumerate() {
        if req.cancel.is_cancelled() {
            break;
        }
        on_progress(TrainingProgress::Caching {
            current: index as u32 + 1,
            total: req.items.len() as u32,
        });
        let text = text_encoder.encode(&item.caption)?.detach();
        // sc-24828: the item's subject mask is read + checked once, then resampled per bucket onto
        // that bucket's latent grid (`None` when masked loss is off).
        let mask = PreparedSubjectMask::load_if_enabled(
            LABEL,
            item,
            req.config.subject_mask_loss.as_ref(),
        )?;
        let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
        for &edge in &edges {
            let (grid, layout) = bucket_layout(edge, text.dim(1)?)?;
            let image = square_image_tensor(&square, edge, device)?;
            let latent = pack_latent_tokens(
                &vae.encode_sample(&image, req.config.seed.wrapping_add(index as u64))?,
                grid,
                config::LATENT_CHANNELS,
            )?
            .detach();
            let mask_weight = bucket_mask_weight(mask.as_ref(), grid, device)?;
            cache.push(CachedSample {
                latent,
                text: text.clone(),
                layout,
                mask_weight,
            });
        }
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
        // Epic 2123 E7: the auxiliary perceptual models count against the device budget before any
        // caching (one reference per (item, bucket) entry, sized at the largest bucket edge).
        let edges = bucket_edges(&req.config);
        let edge = edges.iter().copied().max().unwrap_or(req.config.resolution);
        let aux_bytes =
            perceptual_footprint_bytes(&req.config, edge, req.items.len() * edges.len());
        if aux_bytes > 0 {
            check_aux_memory(
                LABEL,
                component_bytes(&self.dirs.transformer, "", LABEL)?,
                aux_bytes,
                device_training_budget_bytes(&self.device, LABEL),
            )?;
        }
        // Epic 2123 E8: the shared builder loads the decoder + enabled losses before caching (a
        // missing checkpoint fails fast); `None` — nothing loaded — when no aux loss is enabled.
        let perceptual = candle_gen_perceptual::build_perceptual_path(
            &req.config,
            &aux_loss_context(&self.device, &self.dirs.vae),
        )?;
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
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise). The
        // adapter and full fine-tune surfaces share this cache and schedule.
        let buckets = req.config.training_buckets();
        let schedule = BucketSchedule::new(cache.len() / buckets.len(), &buckets, req.config.seed);
        // Epic 2123 E8: references once per (item, bucket) entry; alternation keyed on the item.
        let mut aux = perceptual
            .map(|path| {
                AuxDriver::prepare(
                    path,
                    cache.len(),
                    |entry| {
                        let sample = &cache[entry];
                        tokens_to_latent_grid(
                            &sample.latent.to_dtype(DType::F32)?,
                            sample_grid(sample)?,
                        )
                    },
                    &schedule,
                    accum,
                    0,
                )
            })
            .transpose()?;

        for step in 1..=req.config.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (loss, losses) = run_step(
                &transformer,
                &cache,
                &schedule,
                aux.as_mut(),
                &req.config,
                step,
                compute_dtype,
                mae,
                &self.device,
            )?;
            last_loss = losses.total;
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
                match &surface {
                    // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                    TrainSurface::Adapter(set) => adapter_optimizer_step(
                        &mut optimizer,
                        &mut grads,
                        set,
                        &req.config,
                        update,
                        req.config.seed,
                    )?,
                    // A full fine-tune trains base weights: both noise techniques are refused for
                    // it by the shared `validate_training_techniques` floor, so the plain
                    // clip + step is the whole update.
                    TrainSurface::Full(_) => {
                        clip_grad_norm(&mut grads, &vars, 1.0)?;
                        optimizer.step(&grads)?;
                    }
                }
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

    /// A 2×2-grid cached sample for the tiny transformer, its weight built on the unpacked
    /// `[1, 4, 2, 2]` grid and packed like the latent.
    fn masked_sample(mask: Option<&[f32]>) -> CachedSample {
        let (grid, channels) = (2usize, 4usize);
        let unpacked = [1usize, channels, grid, grid];
        let latent = Tensor::from_vec(values(16, 0.1), &unpacked, &Device::Cpu).unwrap();
        CachedSample {
            latent: pack_latent_tokens(&latent, grid, channels).unwrap(),
            text: Tensor::from_vec(values(8, 0.1), (1, 1, 8), &Device::Cpu).unwrap(),
            layout: PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![1]).unwrap(),
            mask_weight: mask.map(|m| {
                let w = flow_match::subject_mask_weight(m, grid, grid, &unpacked, &Device::Cpu)
                    .unwrap();
                pack_latent_tokens(&w, grid, channels).unwrap()
            }),
        }
    }

    /// sc-24828: the subject-mask weight reaches the step loss on BOTH training surfaces (adapter and
    /// full fine-tune). An all-ones map is the unweighted loss; an all-zero map zeroes the loss AND
    /// every trainable gradient; a half map lands between.
    #[test]
    fn subject_mask_weight_reaches_adapter_and_full_step_loss() {
        let fixture = tiny_transformer_dir();
        let cfg = tiny_config();
        let mut adapter =
            MageTransformer::load_dtype(fixture.path(), &cfg, DType::F32, &Device::Cpu).unwrap();
        let set = build_adapt_lora_targets(
            &mut adapter,
            &["proj_out".to_string(), "to_q".to_string()],
            2,
            2.0,
            7,
            &Device::Cpu,
        )
        .unwrap();
        for var in &set.vars {
            var.set(&Tensor::randn(0f32, 0.02f32, var.as_tensor().dims(), &Device::Cpu).unwrap())
                .unwrap();
        }
        let (full, named) =
            MageTransformer::load_trainable(fixture.path(), &cfg, &Device::Cpu).unwrap();
        let full_vars = named.iter().map(|(_, var)| var.clone()).collect::<Vec<_>>();
        let noise = Tensor::from_vec(values(16, 0.07), (1, 4, 4), &Device::Cpu).unwrap();
        let half: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
        for (model, vars) in [(&adapter, &set.vars), (&full, &full_vars)] {
            let loss = |mask: Option<&[f32]>| {
                step_loss(
                    model,
                    &masked_sample(mask),
                    &noise,
                    0.5,
                    DType::F32,
                    false,
                    None,
                )
                .unwrap()
                .0
            };
            let plain = loss(None).to_scalar::<f32>().unwrap();
            let ones = loss(Some(&[1.0; 4])).to_scalar::<f32>().unwrap();
            assert!((ones - plain).abs() < 1e-6, "{ones} vs {plain}");
            let zero = loss(Some(&[0.0; 4]));
            assert_eq!(zero.to_scalar::<f32>().unwrap(), 0.0);
            let grads = zero.backward().unwrap();
            for var in vars.iter() {
                if let Some(g) = grads.get(var.as_tensor()) {
                    let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert!(g.iter().all(|x| *x == 0.0), "nonzero trainable grad");
                }
            }
            let mid = loss(Some(&half)).to_scalar::<f32>().unwrap();
            assert!(mid > 0.0 && mid < plain, "{mid} vs {plain}");
        }
    }

    /// sc-24828: the packed weight lines up with the packed latent — a weight whose value encodes its
    /// `(y, x)` grid cell lands at token `y·grid + x` on every channel.
    #[test]
    fn packed_subject_mask_weight_lines_up_with_packed_latent() {
        let grid = 3usize;
        let vals: Vec<f32> = (0..grid * grid)
            .map(|i| ((i / grid) * 10 + i % grid) as f32)
            .collect();
        let w =
            flow_match::subject_mask_weight(&vals, grid, grid, &[1, 4, grid, grid], &Device::Cpu)
                .unwrap();
        let packed = pack_latent_tokens(&w, grid, 4).unwrap();
        assert_eq!(packed.dims(), &[1, grid * grid, 4]);
        let rows = packed.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        for (token, row) in rows.iter().enumerate() {
            let expected = ((token / grid) * 10 + token % grid) as f32;
            assert!(row.iter().all(|v| *v == expected), "token {token}: {row:?}");
        }
    }

    /// sc-24828 × sc-2127: with two resolution buckets and masked loss on, the item's mask (loaded
    /// once) yields a weight per bucket packed exactly like that bucket's latent (`[1, grid², C]`),
    /// and the masked-out region (the right half of the centre crop) is zero.
    #[test]
    fn bucket_mask_weight_follows_each_bucket_latent() {
        let dir = tempfile::tempdir().unwrap();
        // A 48x32 landscape image: centre crop x in [8, 40); subject = crop's left half [8, 24).
        let img = dir.path().join("img.png");
        image::RgbImage::new(48, 32).save(&img).unwrap();
        let mask_path = dir.path().join("mask.png");
        image::GrayImage::from_fn(48, 32, |x, _| {
            image::Luma([if (8..24).contains(&x) { 255 } else { 0 }])
        })
        .save(&mask_path)
        .unwrap();
        let mut item = gen_core::TrainingItem::captioned(img, "c".into());
        item.subject_mask_path = Some(mask_path);
        let on = gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask = PreparedSubjectMask::load_if_enabled(LABEL, &item, Some(&on))
            .unwrap()
            .expect("masked loss on");
        for edge in [4 * VAE_DOWNSAMPLE as u32, 8 * VAE_DOWNSAMPLE as u32] {
            let (grid, _) = bucket_layout(edge, 1).unwrap();
            let latent_shape = [1usize, grid * grid, config::LATENT_CHANNELS];
            let w = bucket_mask_weight(Some(&mask), grid, &Device::Cpu)
                .unwrap()
                .expect("weight");
            assert_eq!(w.dims(), &latent_shape, "edge {edge}");
            let rows = w.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
            for (token, row) in rows.iter().enumerate() {
                let expected = if token % grid < grid / 2 { 1.0 } else { 0.0 };
                assert!(
                    row.iter().all(|v| *v == expected),
                    "edge {edge} token {token}: {row:?}"
                );
            }
        }
        assert!(bucket_mask_weight(None, 4, &Device::Cpu).unwrap().is_none());
    }

    #[test]
    fn descriptor_declares_subject_mask_loss() {
        assert!(trainer_descriptor().techniques.subject_mask_loss);
    }

    /// sc-2127: the trainer declares buckets, each bucket edge packs its own grid, and the real
    /// transformer accepts a cached sample of each bucket's size with that bucket's layout.
    #[test]
    fn each_bucket_packs_its_own_grid_and_forwards() {
        assert!(trainer_descriptor().techniques.resolution_buckets);
        for (edge, grid) in [(512u32, 32usize), (1024, 64)] {
            let (g, layout) = bucket_layout(edge, 7).unwrap();
            assert_eq!(g, grid, "edge {edge}");
            assert_eq!(layout.image_tokens(), grid * grid, "edge {edge}");
            assert_eq!(layout.text_tokens(), 7);
        }
        let fixture = tiny_transformer_dir();
        let model =
            MageTransformer::load_dtype(fixture.path(), &tiny_config(), DType::F32, &Device::Cpu)
                .unwrap();
        let (_, text, sigma, _) = tiny_inputs();
        for edge in [VAE_DOWNSAMPLE as u32, 2 * VAE_DOWNSAMPLE as u32] {
            let (grid, layout) = bucket_layout(edge, text.dim(1).unwrap()).unwrap();
            let latent = Tensor::from_vec(
                values(grid * grid * 4, 0.1),
                (1, grid * grid, 4),
                &Device::Cpu,
            )
            .unwrap();
            let output = model.forward(&latent, &text, &sigma, &layout).unwrap();
            assert_eq!(output.dims(), latent.dims(), "edge {edge}");
        }
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

    /// sc-24830 (epic 2123 depth anchoring) — the Candle Mage step seam on the tiny synthetic
    /// transformer (4 latent channels) with the builder's tiny random-init decoder + tiny DA2. CPU,
    /// seconds; no weights downloaded.
    mod depth_anchoring {
        use super::*;
        use candle_gen::gen_core::train::{AuxLossSchedule, DepthModelSize, TrainingConfig};

        fn schedule() -> AuxLossSchedule {
            AuxLossSchedule {
                weight: 0.1,
                t_min: 0.0,
                t_max: 1.0,
                every_n: 2,
            }
        }

        fn cfg() -> TrainingConfig {
            let mut c = TrainingConfig {
                seed: 7,
                ..Default::default()
            };
            c.depth_anchoring.schedule = schedule();
            c
        }

        fn sample(seed: u64) -> CachedSample {
            let (grid, channels) = (2usize, 4usize);
            let latent = Tensor::randn(0f32, 1f32, (1, channels, grid, grid), &Device::Cpu)
                .unwrap()
                .affine(1.0, seed as f64 * 0.01)
                .unwrap();
            CachedSample {
                latent: pack_latent_tokens(&latent, grid, channels).unwrap(),
                text: Tensor::randn(0f32, 1f32, (1, 1, 8), &Device::Cpu).unwrap(),
                layout: PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![1])
                    .unwrap(),
                mask_weight: None,
            }
        }

        fn schedule_of(n: usize) -> BucketSchedule {
            BucketSchedule::new(
                n,
                &[gen_core::train::ResolutionBucket {
                    resolution: 32,
                    repeats: 1,
                }],
                7,
            )
        }

        struct Fixture {
            _dir: tempfile::TempDir,
            model: MageTransformer,
            set: LoraSet,
        }

        fn fixture() -> Fixture {
            let dir = tiny_transformer_dir();
            let mut model =
                MageTransformer::load_dtype(dir.path(), &tiny_config(), DType::F32, &Device::Cpu)
                    .unwrap();
            let set = build_adapt_lora_targets(
                &mut model,
                &["proj_out".to_string(), "to_q".to_string()],
                2,
                2.0,
                7,
                &Device::Cpu,
            )
            .unwrap();
            Fixture {
                _dir: dir,
                model,
                set,
            }
        }

        fn driver(cache: &[CachedSample], sched: &BucketSchedule) -> AuxDriver {
            let path = candle_gen_perceptual::testing::tiny_depth_path(4, schedule(), &Device::Cpu)
                .unwrap();
            AuxDriver::prepare(
                path,
                cache.len(),
                |e| tokens_to_latent_grid(&cache[e].latent, sample_grid(&cache[e])?),
                sched,
                1,
                0,
            )
            .unwrap()
        }

        fn step(
            f: &Fixture,
            cache: &[CachedSample],
            sched: &BucketSchedule,
            aux: Option<&mut AuxDriver>,
            n: u32,
        ) -> (Tensor, StepLosses) {
            run_step(
                &f.model,
                cache,
                sched,
                aux,
                &cfg(),
                n,
                DType::F32,
                false,
                &Device::Cpu,
            )
            .unwrap()
        }

        /// AC1: a depth step has no diffusion term, total == aux, and the LoRA factors get a nonzero
        /// gradient. Mutation: ignore `plan.diffusion` (always compute the velocity term) ⇒ red.
        #[test]
        fn depth_step_trains_the_lora_through_depth_only() {
            let f = fixture();
            let cache = vec![sample(1)];
            let sched = schedule_of(1);
            let mut aux = driver(&cache, &sched);
            let (_, d) = step(&f, &cache, &sched, Some(&mut aux), 1);
            assert_eq!(d.aux, None);
            assert_eq!(Some(d.total), d.diffusion);
            let (loss, depth) = step(&f, &cache, &sched, Some(&mut aux), 2);
            assert_eq!(
                depth.diffusion, None,
                "depth step computes no diffusion loss"
            );
            let a = depth.aux.expect("depth term");
            assert!(a > 0.0 && a.is_finite(), "{a}");
            assert_eq!(depth.total, a);
            let grads = loss.backward().unwrap();
            let mag: f32 = f
                .set
                .vars
                .iter()
                .filter_map(|v| grads.get(v.as_tensor()))
                .map(|g| {
                    g.abs()
                        .unwrap()
                        .sum_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                })
                .sum();
            assert!(mag > 0.0 && mag.is_finite(), "LoRA grad |Σ| {mag}");
        }

        /// Alternation on the real item + references once per entry. Mutation: build the driver with
        /// a per-entry alternation keyed on the global step ⇒ an item locks to one kind ⇒ red.
        #[test]
        fn every_item_alternates_and_references_are_built_once() {
            let f = fixture();
            let cache = vec![sample(1), sample(2)];
            let sched = schedule_of(2);
            let mut aux = driver(&cache, &sched);
            let mut kinds = vec![Vec::new(), Vec::new()];
            for n in 1..=8u32 {
                let item = sched.sample((n - 1) as usize).0;
                let (_, l) = step(&f, &cache, &sched, Some(&mut aux), n);
                kinds[item].push(l.aux.is_some());
            }
            for k in &kinds {
                assert_eq!(k, &vec![false, true, false, true], "{kinds:?}");
            }
            assert_eq!(aux.path().reference_computations(), cache.len());
        }

        /// E1: no driver ⇒ the step equals the pre-epic-2123 step bit for bit (loss value and every
        /// LoRA gradient). Mutation: perturb the off-path loss (scale the prediction by 1.0001) ⇒ red.
        #[test]
        fn everything_off_is_bit_identical_to_the_legacy_step() {
            let f = fixture();
            let cache = vec![sample(1)];
            let sched = schedule_of(1);
            let c = TrainingConfig {
                seed: 7,
                ..Default::default()
            };
            assert!(candle_gen_perceptual::build_perceptual_path(
                &c,
                &aux_loss_context(&Device::Cpu, Path::new("/nonexistent"))
            )
            .unwrap()
            .is_none());
            assert_eq!(perceptual_footprint_bytes(&c, 1024, 4), 0);
            let (loss, off) = run_step(
                &f.model,
                &cache,
                &sched,
                None,
                &c,
                1,
                DType::F32,
                false,
                &Device::Cpu,
            )
            .unwrap();
            assert_eq!(off.aux, None);
            let g_off = loss.backward().unwrap();
            // The pre-sc-24830 step body.
            let s = &cache[0];
            let sigma = flow_match::sample_unit_timestep(
                &c.timestep_type,
                &c.timestep_bias,
                flow_match::timestep_seed(c.seed, 1),
            ) as f64;
            let noise = sample_noise(s.latent.dims(), noise_seed(c.seed, 1), &Device::Cpu).unwrap();
            let (x_t, target) = build_training_batch(&s.latent, &noise, sigma, DType::F32).unwrap();
            let pred = f
                .model
                .forward(
                    &x_t,
                    &s.text,
                    &Tensor::new(&[sigma as f32], &Device::Cpu).unwrap(),
                    &s.layout,
                )
                .unwrap();
            let legacy = weighted_velocity_loss(&pred, &target, None, false).unwrap();
            assert_eq!(off.total, legacy.to_scalar::<f32>().unwrap());
            let g_legacy = legacy.backward().unwrap();
            let bits = |t: &Tensor| -> Vec<u32> {
                t.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            };
            for v in &f.set.vars {
                let (a, b) = (g_off.get(v.as_tensor()), g_legacy.get(v.as_tensor()));
                assert_eq!(a.map(bits), b.map(bits));
            }
        }

        /// E7: depth grows the estimate by decoder + DA2 (more for Large); the shared check refuses at
        /// a synthetic budget between base and base + aux. Mutation: compute the footprint with
        /// `DecoderSpec::None` ⇒ the decoder term vanishes ⇒ red.
        #[test]
        fn memory_estimate_includes_the_aux_models() {
            let mut on = TrainingConfig::default();
            on.depth_anchoring.schedule = schedule();
            let small = perceptual_footprint_bytes(&on, 1024, 10);
            let da2 =
                candle_gen_depth::anchor::depth_anchor_footprint(DepthModelSize::Small, 1024, 1024);
            let decoder = MageDecoderSpec {
                vae_dir: PathBuf::new(),
            }
            .footprint_for_test(1024);
            assert_eq!(
                small,
                candle_gen::train::perceptual::perceptual_footprint_bytes(
                    Some(decoder),
                    &[da2],
                    10
                )
            );
            on.depth_anchoring.model_size = DepthModelSize::Large;
            let large = perceptual_footprint_bytes(&on, 1024, 10);
            assert!(large > small + 1_000_000_000, "{small} {large}");
            let base = 10_000_000_000u64;
            assert!(check_aux_memory(LABEL, base, 0, base + large / 2).is_ok());
            assert!(check_aux_memory(LABEL, base, large, base + large / 2).is_err());
        }

        /// E3: declared; full fine-tune refuses it (typed); a missing decoder dir is named.
        #[test]
        fn descriptor_and_refusals() {
            assert!(trainer_descriptor().techniques.depth_anchoring);
            let mut req = full_request();
            req.config.depth_anchoring.schedule = schedule();
            assert!(matches!(
                refuse_aux_losses_on_full_finetune(&req),
                Err(gen_core::Error::Unsupported(_))
            ));
            req.config.full_finetune = false;
            assert!(refuse_aux_losses_on_full_finetune(&req).is_ok());
            let tmp = tempfile::tempdir().unwrap();
            let mut c = cfg();
            c.perceptual_decoder_dir = Some(tmp.path().join("no-vae"));
            c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
            let e = candle_gen_perceptual::build_perceptual_path(
                &c,
                &aux_loss_context(&Device::Cpu, &tmp.path().join("no-vae")),
            )
            .err()
            .unwrap()
            .to_string();
            assert!(e.contains("Mage-VAE decoder"), "{e}");
        }

        impl MageDecoderSpec {
            fn footprint_for_test(&self, edge: u32) -> AuxModelFootprint {
                candle_gen_perceptual::CustomDecoder::footprint(self, edge, edge)
            }
        }
    }
}
