//! Rectified-flow **LoRA/LoKr training** on the Mage-Flow NR-MMDiT, Mac-first on MLX (sc-14055).
//!
//! The [`MageFlowTrainer`] realizes the backend-neutral [`Trainer`] contract on the **Base**
//! checkpoint (the confirmed training target, epic sc-14034): a frozen DiT + VAE encoder + Qwen3-VL
//! LM text encoder that caches a captioned image dataset to Mage-VAE latents / prompt embeddings,
//! then runs the functional-autograd LoRA loop and writes a PEFT adapter that reloads through the
//! inference path ([`crate::adapters::apply_mage_adapters`]).
//!
//! ## Reuse — the shared training core, with the Mage forward
//!
//! The host-generic factor machinery lives in [`mlx_gen::train::lora`] (build LoRA/LoKr targets,
//! inject them as forward-time residuals via
//! [`AdaptableLinear::set_adapters`](mlx_gen::adapters::AdaptableLinear::set_adapters), write the
//! PEFT safetensors), hoisted so every family trainer shares it. This module keeps only what is
//! Mage-specific: the native-resolution *packing* of a single training sample, the Mage-VAE encode
//! for latent prep, the Qwen3-VL gen-path conditioning, and the rectified flow-match noising.
//!
//! ## The rectified flow-matching objective and its sign — this is load-bearing
//!
//! Mage's sampler ([`crate::pipeline`]) starts from noise at scheduler sigma `σ = 1`, integrates
//! `x += (σ_next − σ_cur)·v` with `σ` decreasing to 0 (data), and the DiT `forward` returns the raw
//! velocity **without a negation**. For that Euler step to follow the flow-match ODE on the
//! interpolant `z_σ = (1−σ)·z + σ·ε` (data `z` at `σ = 0`, noise `ε` at `σ = 1`), the velocity the
//! model must predict is
//!
//! ```text
//! v = dz_σ/dσ = ε − z = noise − data.
//! ```
//!
//! So the training regression target is **`noise − data`**, and the loss is
//! `L = ‖ v_θ(z_σ, σ, τ) − (ε − z) ‖²`. The epic/story write the objective's velocity target as
//! `(z − ε)`; that is the opposite sign convention (the *forward*-time ODE velocity `−dz_σ/dσ`).
//! The value that matches Mage's **actual, parity-verified** sampler — `pipeline::flow_euler_step`
//! is `x + v·(σ_next − σ_cur)` and `MageTransformer::forward` does not negate — is `noise − data`,
//! which is exactly the regression target the z-image sibling trainer already uses (`noise − x0`).
//! A first-step check confirms it: at `σ = 1`, `x = ε`; one step to `σ' ≈ 0.947` gives
//! `ε + (σ' − 1)·(ε − z) = 0.947·ε + 0.053·z`, exactly `z_{σ'}`. Training toward `(z − ε)` would fit
//! the model to the negated velocity and corrupt generation.
//!
//! ## Timestep sampling distribution — the documented gap resolution
//!
//! Mage's **main-training** timestep sampling distribution is not published: the reference repo is
//! inference-only (no training code), and the paper states only that the VAE stage-I uses `U(0,1)`
//! (epic sc-14034 GAP 6, re-confirmed against the vendored source). Per the epic decision, this
//! trainer **defaults to the z-image trainer's schedule** — `sigmoid(randn)` with the same
//! `timestep_type` / `timestep_bias` knobs (the private `sample_sigma`) — and this choice is
//! recorded here, in
//! the PR, and on the story. The sampled `σ` is fed to the DiT directly (it is the scheduler sigma
//! where `σ = 1` is noise); the static schedule shift is a *sampling-time* schedule warp and is
//! **not** applied during training, matching the sibling.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{FAMILY, LATENT_CHANNELS, VAE_DOWNSAMPLE_FACTOR};
use crate::latent::GsKey;
use crate::pipeline::{denoise, encode_noise_tokens, generation_layout, mage_flow_sigmas};
use crate::rope_embedder::{ImgShape, PackLayout};
use crate::text_encoder::{MageTextEncoder, PromptKind};
use crate::transformer::MageTransformer;
use crate::vae::{MageVae, VaePart};

/// The registered trainer id — the Base checkpoint is the training target, so the trainer shares the
/// `mage_flow_base` generator id (the [`TrainerDescriptor::id`] convention).
pub const MODEL_ID: &str = "mage_flow_base";

/// Mage reconstructs a LoKr delta at **bf16** for inference (the bf16-residual path); training must
/// reconstruct at the same dtype so the adapter round-trips bit-for-bit.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence.
const SAMPLE_PROMPT_CAP: usize = 4;

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma`:
/// `x_t = (1−σ)·x0 + σ·noise`, `target = noise − x0`, `timestep = σ`.
///
/// See the module docs for why the target is `noise − x0` (`ε − z`) and not `(z − ε)`: it is the
/// velocity Mage's own sampler integrates, and `forward` does not negate. `timestep` is the
/// scheduler sigma fed straight to the DiT (no `1 − σ` reparameterization, unlike z-image's DiT,
/// and no static-shift warp — that is a sampling-time schedule, not a training one). `x0` is cast to
/// f32 so the target/interpolant stay f32 regardless of the cached latent dtype (master-weights).
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let x0 = x0.as_dtype(Dtype::Float32)?;
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(&x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, &x0)?;
    Ok((x_t, target, sigma))
}

/// A cached training sample: the clean VAE-latent tokens `[1, gh·gw, 128]`, the encoded conditioning
/// `[1, txt_tokens, hidden]`, and its post-drop token count (native-resolution packing is per-sample,
/// so each caption keeps its own length).
struct CachedSample {
    latent_tokens: Array,
    txt: Array,
    txt_tokens: i32,
}

/// The frozen base components a Mage-Flow LoRA run trains against.
pub struct MageFlowTrainer {
    descriptor: TrainerDescriptor,
    /// The Qwen3-VL LM text encoder, in an `Option` so it can be **dropped after the caching loop**
    /// (32 GB-Mac support): every prompt is already encoded into the cache and the preview prompts
    /// pre-encoded, so it is idle during the train loop yet a multi-GB resident. Freeing it before
    /// the loop reclaims that budget for the DiT working set.
    text_encoder: Option<MageTextEncoder>,
    vae: MageVae,
    transformer: MageTransformer,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // LoRA/LoKr only — no control-branch training path.
        supports_control: false,
    }
}

/// Construct the trainer from a diffusers snapshot directory (`text_encoder/ transformer/ vae/`). No
/// quantization — training needs the dense base. The VAE is loaded with its **encoder** (latent prep)
/// and decoder (preview samples). Registered via [`mlx_gen::register_trainer`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(
                "mage_flow_base trainer expects a diffusers snapshot directory (text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    // sc-14980/sc-14979: training reads the DENSE base, which under the split mirror layout means
    // the `bf16/` tier's DiT plus the SHARED text encoder + VAE staged as caller-provisioned
    // co-requisite dirs. A flat snapshot (upstream, or an existing install) stages nothing and every
    // component resolves under `root` exactly as before — the trainer is unchanged on that path.
    let dirs = crate::model::resolve_component_dirs(root, spec)?;
    // Training must never run against packed weights: the gradient path needs dense projections and
    // `quantize` is a no-op over an already-packed base, so a q4/q8 tier would train silently wrong
    // rather than fail. This is the engine-side twin of the app's `TrainingTierMissing` pre-flight.
    for (label, dir) in [
        ("transformer", &dirs.transformer),
        ("text_encoder", &dirs.text_encoder),
    ] {
        let (Some(parent), Some(name)) = (dir.parent(), dir.file_name().and_then(|n| n.to_str()))
        else {
            continue;
        };
        if let Some(bits) = mlx_gen::quant::packed_quant_bits(parent, name)? {
            return Err(mlx_gen::Error::Msg(format!(
                "mage_flow_base trainer requires the dense bf16 base, but {label} at {} is a \
                 pre-quantized Q{bits} artifact; install the bf16 tier to train",
                dir.display()
            )));
        }
    }
    Ok(Box::new(MageFlowTrainer {
        descriptor: trainer_descriptor(),
        text_encoder: Some(crate::text_encoder::load_dir(&dirs.text_encoder)?),
        vae: crate::vae::load(&dirs.vae, VaePart::Both, Dtype::Bfloat16)?,
        transformer: MageTransformer::load(&dirs.transformer)?,
    }))
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const REGISTRATION = trainer_descriptor => load_trainer
}

/// Recognized `timestep_type` values [`sample_sigma`] branches on plus the `sigmoid` default it
/// falls back to. Any other string would silently sample sigmoid — rejected in [`validate_request`].
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values plus the neutral default.
const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];
/// Recognized `loss_type` values — `mae`/`l1` select MAE, `mse`/`l2` the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// Normalize a free-form config string the way the trainer's parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects an empty dataset, zero rank, **zero steps** (a 0-step run would fall through to
/// the save and write a no-op `B = 0` identity adapter), an unsupported optimizer, and an
/// unrecognized `timestep_type` / `timestep_bias` / `loss_type` (rather than silently falling back to
/// a default sampler/loss). The non-empty target-module resolution is checked in [`Trainer::validate`],
/// which has the loaded DiT to match suffixes against.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("mage_flow_base trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("mage_flow_base trainer: rank must be > 0".into());
    }
    if req.config.steps == 0 {
        return Err("mage_flow_base trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "mage_flow_base trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&req.config.timestep_type).as_str()) {
        return Err(format!(
            "mage_flow_base trainer: timestep_type '{}' is not recognized (supported: {})",
            req.config.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&req.config.timestep_bias).as_str()) {
        return Err(format!(
            "mage_flow_base trainer: timestep_bias '{}' is not recognized (supported: {})",
            req.config.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&req.config.loss_type).as_str()) {
        return Err(format!(
            "mage_flow_base trainer: loss_type '{}' is not recognized (supported: {})",
            req.config.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for MageFlowTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared control-training floor: a LoRA-only trainer must reject a control-branch request
        // (typed `Unsupported`) rather than silently training a plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        validate_request(req)?;
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would resolve
        // to an empty target set — a full-length run that trains zero parameters yet "succeeds".
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "mage_flow_base trainer: lora_target_modules {:?} matched no adaptable module on \
                 the DiT",
                req.config.lora_target_modules
            )
            .into());
        }
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl MageFlowTrainer {
    /// The rich-`Result` body behind [`Trainer::train`].
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);

        // Training resolution → square latent grid. `bucket_resolution` floors to a multiple of 32
        // (a subset of Mage's 16× stride, so the latent tiles cleanly). `patch_size == 1`, so the
        // token count is exactly the latent grid.
        let edge = bucket_resolution(cfg.resolution);
        let grid = (edge / VAE_DOWNSAMPLE_FACTOR) as i32;

        // The published checkpoint is bf16; bf16 is the trained/native configuration and the
        // ecosystem-standard mixed precision (the trainable factors / loss / grads / optimizer stay
        // f32 — master weights). `f32` widens the frozen base losslessly. Casting is idempotent for
        // the frozen base (bf16→f32→bf16 round-trips), so no "already cast" guard is needed.
        let compute_dtype = if cfg.train_dtype.trim().eq_ignore_ascii_case("bf16")
            || cfg.train_dtype.trim().eq_ignore_ascii_case("bfloat16")
        {
            Dtype::Bfloat16
        } else if cfg.train_dtype.trim().eq_ignore_ascii_case("f32")
            || cfg.train_dtype.trim().eq_ignore_ascii_case("float32")
            || cfg.train_dtype.trim().is_empty()
        {
            Dtype::Float32
        } else {
            // Unrecognized values mean f32 (the contract), but the frozen base is bf16 on disk, so
            // f32 is the widening default rather than a silent narrowing.
            Dtype::Float32
        };
        self.transformer.cast_weights(compute_dtype)?;

        // --- prepare → cache: VAE-latents + prompt-embeds into memory before the loop ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        let mut cache: Vec<CachedSample> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let nchw = preprocess_to_nchw(&img, edge)?;
            // Mage-VAE encode at t = 0 → posterior **mean** (`sample_posterior = false`), no latent
            // scale/shift. `[1, 128, gh, gw]` → token layout `[1, gh·gw, 128]`.
            let latent_tokens = self
                .vae
                .encode_mean(&nchw)?
                .transpose_axes(&[0, 2, 3, 1])?
                .reshape(&[1, grid * grid, LATENT_CHANNELS])?;
            let text_encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "mage_flow_base trainer: text encoder already freed (caching after loop)"
                        .into(),
                )
            })?;
            let (txt, txt_tokens) = encode_caption(text_encoder, &item.caption)?;
            eval([&latent_tokens, &txt])?;
            cache.push(CachedSample {
                latent_tokens,
                txt,
                txt_tokens,
            });
        }
        if cache.is_empty() {
            // Disambiguate cancel-during-caching (typed `Canceled`) from a genuinely unusable dataset.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("mage_flow_base trainer: no usable dataset items".into());
        }

        // Pre-encode the preview-sample prompts while the encoder is still resident (freed just
        // below). Each `sample_every` cadence reuses these to render previews from the in-progress
        // adapter. Skipped when sampling is off (the default) or the run is already cancelled.
        let sample_caps: Vec<(String, Array, i32)> = if cfg.sample_every > 0
            && !cfg.sample_prompts.is_empty()
            && !req.cancel.is_cancelled()
        {
            let text_encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "mage_flow_base trainer: text encoder already freed (sample pre-encode)".into(),
                )
            })?;
            let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let (txt, txt_tokens) = encode_caption(text_encoder, prompt)?;
                eval([&txt])?;
                caps.push((prompt.clone(), txt, txt_tokens));
            }
            caps
        } else {
            Vec::new()
        };

        // Every prompt is now encoded; drop the Qwen encoder and evict its buffers before the loop.
        self.text_encoder = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(&self.transformer, cfg);
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(
                    &mut self.transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.seed,
                )?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut self.transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.decompose_factor,
                    cfg.seed,
                )?;
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        let alpha = cfg.alpha;
        let mae = {
            let lt = normalize_cfg(&cfg.loss_type);
            lt == "mae" || lt == "l1"
        };

        // AdamW with wd=0 is identical to Adam, so one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;

        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        // --- resume: continue from the latest snapshot of THIS adapter in output_dir, if any ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!("[sc-14055] resuming from step {start_step} (update {update_idx})");
            }
        }

        // --- train loop ---
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let sample = &cache[((step - 1) as usize) % cache.len()];
            let sigma = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            )?;
            let noise = random::normal::<f32>(
                sample.latent_tokens.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut self.transformer,
                &params,
                &adapter,
                alpha,
                rank,
                sample,
                grid,
                sigma,
                &noise,
                mae,
                compute_dtype,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // The final update can fire with fewer than `accum` grads when `steps` isn't a
                // multiple of the accumulation; divide by the actual in-window count.
                let window = if step % accum == 0 {
                    accum
                } else {
                    step % accum
                };
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    window,
                )?;
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                let clipped: LoraParams = clipped
                    .into_iter()
                    .map(|(k, v)| (k, v.into_owned()))
                    .collect();
                opt.step(&mut params, &clipped)?;
                eval(params.values())?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                adapter.save(&params, alpha, rank, cfg.decompose_factor, "", &ckpt)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // Periodic preview samples from the in-progress adapter (installed exactly as a train
            // step installs it; the next step's traced loss re-installs, so no teardown is needed).
            if cfg.sample_every > 0 && !sample_caps.is_empty() && step % cfg.sample_every == 0 {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                adapter.install_as(
                    &mut self.transformer,
                    &params,
                    alpha,
                    rank,
                    lora_dtype,
                    LOKR_DTYPE,
                )?;
                let total = sample_caps.len() as u32;
                for (i, (prompt, txt, txt_tokens)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    // Previews are best-effort: a render failure must NOT abort the long-running
                    // training run — log it and keep training.
                    match render_sample(
                        &self.transformer,
                        &self.vae,
                        txt,
                        *txt_tokens,
                        edge,
                        cfg.sample_steps.max(1) as usize,
                        cfg.sample_guidance_scale,
                        sample_seed as i64,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(mlx_gen::Error::Canceled) => break,
                        Err(e) => eprintln!(
                            "[sc-14055] {MODEL_ID} preview sample failed at step {step} (prompt \
                             {}): {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
            }
        }

        // Cancelled before completing a single step: the LoRA factors are still `B = 0`, a no-op
        // adapter. Surface the cancellation rather than writing a valid-looking identity adapter.
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter (PEFT keys + alpha/rank into __metadata__) ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save(
            &params,
            alpha,
            rank,
            cfg.decompose_factor,
            "",
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted paths by matching them against every adapter-routable module on the DiT — the same
/// suffix-match PEFT's `LoraConfig(target_modules=…)` does. The default trains the image-stream
/// attention projections in every block; a config can name any block leaf (the text-stream
/// `add_*_proj`/`to_add_out`, the FFN `net.0.proj`/`net.2`, the modulation `img_mod.1`/`txt_mod.1`,
/// or the global `img_in`/`txt_in`).
fn resolve_target_paths(transformer: &MageTransformer, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    AdaptableHost::adaptable_paths(transformer)
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = image::open(path)
        .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Resize a center-cropped square core [`Image`] to `edge × edge` and normalise to the reference's
/// `[-1, 1]` NCHW input (`pixel / 127.5 − 1`), matching `pipeline::reference_nchw`.
fn preprocess_to_nchw(image: &Image, edge: u32) -> Result<Array> {
    let rgb = image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .ok_or_else(|| {
            mlx_gen::Error::Msg("mage_flow_base trainer: dataset image is not valid RGB8".into())
        })?;
    let resized = if rgb.dimensions() == (edge, edge) {
        rgb
    } else {
        image::imageops::resize(&rgb, edge, edge, image::imageops::FilterType::CatmullRom)
    };
    let mut values = vec![0f32; 3 * edge as usize * edge as usize];
    for channel in 0..3usize {
        for y in 0..edge as usize {
            for x in 0..edge as usize {
                let pixel = resized.get_pixel(x as u32, y as u32)[channel] as f32;
                values[(channel * edge as usize + y) * edge as usize + x] = pixel / 127.5 - 1.0;
            }
        }
    }
    Ok(Array::from_slice(
        &values,
        &[1, 3, edge as i32, edge as i32],
    ))
}

/// Encode one caption through the Qwen3-VL gen (LM) path, returning `([1, txt_tokens, hidden], txt_tokens)`.
fn encode_caption(text_encoder: &MageTextEncoder, caption: &str) -> Result<(Array, i32)> {
    let conditioning = text_encoder.encode(&[caption], PromptKind::Gen)?;
    let txt_tokens = conditioning.seq_lens[0] as i32;
    let hidden = conditioning.txt.shape()[1];
    let txt = conditioning.txt.reshape(&[1, txt_tokens, hidden])?;
    Ok((txt, txt_tokens))
}

/// Sample a normalised flow-match `σ ∈ [1e-3, 1−1e-3]` — a faithful port of the z-image trainer's
/// `sample_sigma` (the epic's documented default for Mage's unpublished main-training distribution):
/// `sigmoid(randn)` by default, `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted;
/// bias `high` → `√σ`, `low` → `σ²`. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = normalize_cfg(timestep_type);
    let t = match ttype.as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed.wrapping_add(0x9E37_79B9))?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    let t = match normalize_cfg(timestep_bias).as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), pack the
/// single training sample, run the DiT training forward, regress the velocity toward `noise − x0`,
/// return `(loss, grads)`.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut MageTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    sample: &CachedSample,
    grid: i32,
    sigma: f32,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (x_t, target, _timestep) = build_batch(&sample.latent_tokens, noise, sigma)?;
    // One packed segment: this image's latent grid + its caption tokens.
    let layout =
        PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![sample.txt_tokens])?;
    let ctx = transformer.pack_context(layout)?;
    let sigma_arr = Array::from_slice(&[sigma], &[1]); // one entry per packed segment
    let txt = sample.txt.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // NEVER check the cancel flag inside this traced grad closure — it returns `MlxResult`, so an
        // early-out would be stringified through `Exception::custom` and lose the typed `Canceled`.
        // Cancellation is the caller's job at the step boundary; the grad graph is one atomic unit.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let v = transformer
            .forward_train(&x_t, &txt, &sigma_arr, &ctx)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let diff = subtract(&v, &target)?;
        // `mean(None)` reduces to a 0-d scalar (grad requires a scalar cotangent). The v(bf16 on the
        // bf16 path) − target(f32) subtract promotes to f32, so the loss/grads are f32 (master weights).
        let loss = if mae {
            diff.abs()?.mean(None)?
        } else {
            diff.square()?.mean(None)?
        };
        Ok(vec![loss])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    Ok((val[0].item::<f32>(), grads))
}

/// Render one preview image from the in-progress adapter (already installed on `transformer`): a
/// short native-resolution txt2img denoise + VAE decode into a core [`Image`]. Runs the CFG-off path
/// when `guidance <= 1` (the reference builds no negative branch there); above 1 it packs a blank
/// negative branch, matching the generate path.
#[allow(clippy::too_many_arguments)]
fn render_sample(
    transformer: &MageTransformer,
    vae: &MageVae,
    cond_txt: &Array,
    cond_tokens: i32,
    edge: u32,
    steps: usize,
    guidance: f32,
    seed: i64,
) -> Result<Image> {
    let grid = (edge / VAE_DOWNSAMPLE_FACTOR) as i32;
    let key = GsKey::default();
    let tokens = encode_noise_tokens(grid, grid, seed, &key, Dtype::Bfloat16)?;
    let cond = cond_txt.as_dtype(Dtype::Bfloat16)?;
    let sigmas = mage_flow_sigmas(steps)?;
    // CFG-off preview: one segment, positive conditioning only (matching the generate path at cfg 1).
    let uses_cfg = guidance > 1.0;
    let negative = if uses_cfg {
        Some((cond.clone(), vec![cond_tokens]))
    } else {
        None
    };
    let layout = generation_layout(&[(grid, grid)], vec![cond_tokens])?;
    let cfg = if uses_cfg { guidance } else { 1.0 };
    let final_tokens = denoise(
        transformer,
        tokens,
        &cond,
        layout,
        negative.as_ref().map(|(txt, lens)| (txt, lens.clone())),
        cfg,
        false,
        &sigmas,
    )?;
    let image_u8 = crate::pipeline::decode(vae, &final_tokens, grid, grid)?;
    eval([&image_u8])?;
    let pixels = image_u8
        .try_as_slice::<u8>()
        .map_err(|e| mlx_gen::Error::Msg(format!("mage_flow_base trainer preview: {e}")))?
        .to_vec();
    Ok(Image {
        width: (grid * VAE_DOWNSAMPLE_FACTOR as i32) as u32,
        height: (grid * VAE_DOWNSAMPLE_FACTOR as i32) as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{CancelFlag, TrainingItem};

    fn base_request(items: Vec<TrainingItem>, config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items,
            config,
            output_dir: std::env::temp_dir().join("mage_trainer_unit"),
            file_name: "lora.safetensors".to_string(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    fn one_item() -> Vec<TrainingItem> {
        vec![TrainingItem::captioned("a.png".into(), "a caption".into())]
    }

    #[test]
    fn build_batch_is_the_rectified_flow_interpolant_and_sign() {
        // x0 = data, noise = ε. At σ, x_t = (1−σ)·x0 + σ·noise and target = noise − x0.
        let x0 = Array::from_slice(&[1.0f32, 3.0], &[1, 1, 2]);
        let noise = Array::from_slice(&[-1.0f32, 5.0], &[1, 1, 2]);
        let (x_t, target, timestep) = build_batch(&x0, &noise, 0.25).unwrap();
        // x_t = 0.75·[1,3] + 0.25·[-1,5] = [0.5, 3.5].
        assert_eq!(x_t.as_slice::<f32>(), &[0.5, 3.5]);
        // target = noise − x0 = [-2, 2]  (ε − z), NOT z − ε = [2, -2].
        assert_eq!(target.as_slice::<f32>(), &[-2.0, 2.0]);
        // timestep is the scheduler sigma itself (no 1 − σ, no static-shift warp).
        assert_eq!(timestep, 0.25);
    }

    #[test]
    fn build_batch_endpoints_match_the_sampler_convention() {
        let x0 = Array::from_slice(&[2.0f32], &[1, 1, 1]);
        let noise = Array::from_slice(&[7.0f32], &[1, 1, 1]);
        // σ = 0 → pure data; σ = 1 → pure noise (the interpolant's endpoints).
        assert_eq!(
            build_batch(&x0, &noise, 0.0).unwrap().0.as_slice::<f32>(),
            &[2.0]
        );
        assert_eq!(
            build_batch(&x0, &noise, 1.0).unwrap().0.as_slice::<f32>(),
            &[7.0]
        );
    }

    #[test]
    fn sample_sigma_is_clamped_and_deterministic() {
        for (ty, bias) in [
            ("sigmoid", "balanced"),
            ("uniform", "high_noise"),
            ("weighted", "low_noise"),
        ] {
            let a = sample_sigma(ty, bias, 42).unwrap();
            let b = sample_sigma(ty, bias, 42).unwrap();
            assert_eq!(a, b, "same seed must be deterministic");
            assert!(
                (1e-3..=1.0 - 1e-3).contains(&a),
                "σ in the open unit range: {a}"
            );
        }
        // The bias tilts the distribution: high → √σ (larger), low → σ² (smaller), from the same draw.
        let hi = sample_sigma("uniform", "high_noise", 7).unwrap();
        let lo = sample_sigma("uniform", "low_noise", 7).unwrap();
        let base = sample_sigma("uniform", "balanced", 7).unwrap();
        assert!(
            hi >= base - 1e-6 && lo <= base + 1e-6,
            "hi {hi} base {base} lo {lo}"
        );
    }

    #[test]
    fn validate_rejects_empty_dataset_zero_rank_zero_steps() {
        assert!(validate_request(&base_request(vec![], TrainingConfig::default())).is_err());
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                rank: 0,
                ..Default::default()
            }
        ))
        .is_err());
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                steps: 0,
                ..Default::default()
            }
        ))
        .is_err());
    }

    #[test]
    fn validate_rejects_unknown_optimizer_and_sampler_and_loss() {
        let bad = |f: fn(&mut TrainingConfig)| {
            let mut c = TrainingConfig::default();
            f(&mut c);
            validate_request(&base_request(one_item(), c)).is_err()
        };
        assert!(bad(|c| c.optimizer = "nope".into()));
        assert!(bad(|c| c.timestep_type = "gaussian".into()));
        assert!(bad(|c| c.timestep_bias = "sideways".into()));
        assert!(bad(|c| c.loss_type = "huber".into()));
        // The defaults (adamw / sigmoid / balanced / mse) pass.
        assert!(validate_request(&base_request(one_item(), TrainingConfig::default())).is_ok());
    }

    /// sc-14055 convergence smoke test — a real short training on the fixed weights. Gradient
    /// descent on a **stationary** objective (one fixed latent/caption/sigma/noise) must drive the
    /// rectified flow-match loss down monotonically-ish and substantially; this isolates the trainer
    /// loop's mechanics (forward → autograd → optimizer → re-inject) from the per-step sigma/noise
    /// variance that makes a real random-schedule loss curve too noisy to gate on over a short run.
    /// Random latent/caption stand in for the VAE/TE encode — the loop is what is under test.
    ///
    /// Run (this Mac has MLX GPU):
    ///   MAGE_BASE_SNAPSHOT=/path/to/Mage-Flow-Base \
    ///     cargo test -p mlx-gen-mage --lib overfits_a_fixed_batch -- --ignored --nocapture
    #[test]
    #[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT)"]
    fn overfits_a_fixed_batch_loss_decreases() {
        let Ok(root) = std::env::var("MAGE_BASE_SNAPSHOT") else {
            return;
        };
        let root = std::path::Path::new(&root);
        let mut transformer = MageTransformer::load(root.join("transformer")).unwrap();
        let vae = crate::vae::load(root.join("vae"), VaePart::Both, Dtype::Bfloat16).unwrap();
        let text_encoder = crate::text_encoder::load(root).unwrap();

        // A real, low-entropy solid-colour swatch: the base predicts its velocity well (an
        // in-distribution latent), so the fixed-batch overfit shows a real, substantial drop rather
        // than the plateau a random (unlearnable) target imposes.
        let edge = 256u32;
        let grid = (edge / VAE_DOWNSAMPLE_FACTOR) as i32;
        let mut swatch = image::RgbImage::new(edge, edge);
        for px in swatch.pixels_mut() {
            *px = image::Rgb([200u8, 40, 40]);
        }
        let core = Image {
            width: edge,
            height: edge,
            pixels: swatch.into_raw(),
        };
        let nchw = preprocess_to_nchw(&core, edge).unwrap();
        let latent_tokens = vae
            .encode_mean(&nchw)
            .unwrap()
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
            .reshape(&[1, grid * grid, LATENT_CHANNELS])
            .unwrap();
        let (txt, txt_tokens) = encode_caption(&text_encoder, "a solid red colour swatch").unwrap();
        eval([&latent_tokens, &txt]).unwrap();
        drop(text_encoder);
        let sample = CachedSample {
            latent_tokens,
            txt,
            txt_tokens,
        };
        let noise = random::normal::<f32>(
            &[1, grid * grid, LATENT_CHANNELS],
            None,
            None,
            Some(&random::key(12).unwrap()),
        )
        .unwrap();
        let sigma = 0.5f32;

        let cfg = TrainingConfig {
            rank: 8,
            alpha: 8.0,
            ..Default::default()
        };
        let paths = resolve_target_paths(&transformer, &cfg);
        assert_eq!(
            paths.len(),
            48,
            "default targets: to_q/k/v/out.0 over 12 blocks"
        );
        let (targets, mut params) = build_lora_targets(&mut transformer, &paths, 8, 7).unwrap();
        let adapter = TrainAdapter::Lora { targets };
        // The default 1e-4 LR: high-LR AdamW overshoots this landscape (the loss climbs before it
        // recovers), which is exactly why a short real-schedule run at 1e-3 looks flat.
        let mut opt = TrainOptimizer::from_config("adamw", 1e-4, 0.0).unwrap();

        let mut losses = Vec::new();
        for _ in 0..100 {
            let (loss, grads) = compute_loss_grads(
                &mut transformer,
                &params,
                &adapter,
                8.0,
                8.0,
                &sample,
                grid,
                sigma,
                &noise,
                false,
                Dtype::Bfloat16,
            )
            .unwrap();
            losses.push(loss);
            opt.step(&mut params, &grads).unwrap();
            eval(params.values()).unwrap();
        }
        println!(
            "[overfit] fixed-batch loss {:.5} -> {:.5}\n[overfit] curve: {losses:?}",
            losses[0],
            losses.last().unwrap()
        );
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "no NaN/Inf during the overfit"
        );
        assert!(
            *losses.last().unwrap() < losses[0] * 0.5,
            "a stationary-objective LoRA overfit must halve the loss: {} -> {}",
            losses[0],
            losses.last().unwrap()
        );
    }

    #[test]
    fn descriptor_advertises_lora_lokr_no_control_on_the_base_target() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "mage_flow_base");
        assert_eq!(d.family, FAMILY);
        assert_eq!(d.backend, "mlx");
        assert!(d.supports_lora && d.supports_lokr);
        assert!(!d.supports_control);
    }
}
