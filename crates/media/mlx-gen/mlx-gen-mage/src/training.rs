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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

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
use mlx_gen::weights::Weights;
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{MageFlowConfig, FAMILY, LATENT_CHANNELS, VAE_DOWNSAMPLE_FACTOR};
use crate::latent::GsKey;
use crate::pipeline::{denoise, encode_noise_tokens, generation_layout, mage_flow_sigmas};
use crate::rope_embedder::{ImgShape, MsRope, PackContext, PackLayout};
use crate::text_encoder::{MageTextEncoder, PromptKind};
use crate::transformer::{MageTransformer, TRANSFORMER_CONFIG_FILE, TRANSFORMER_WEIGHTS_FILE};
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

/// One pre-encoded preview-sample prompt — `(prompt, txt embedding [1, tokens, hidden], token count)`.
/// Encoded during the dataset-caching pass, while the Qwen encoder is still resident, and reused at
/// every [`TrainingConfig::sample_every`] cadence by both train paths.
type SamplePrompt = (String, Array, i32);

/// The base components a Mage-Flow training run trains against — the LoRA/LoKr adapter path (frozen
/// DiT + injected factors) and the full base fine-tune path (every DiT weight trainable, sc-14056).
pub struct MageFlowTrainer {
    descriptor: TrainerDescriptor,
    /// The Qwen3-VL LM text encoder, in an `Option` so it can be **dropped after the caching loop**
    /// (32 GB-Mac support): every prompt is already encoded into the cache and the preview prompts
    /// pre-encoded, so it is idle during the train loop yet a multi-GB resident. Freeing it before
    /// the loop reclaims that budget for the DiT working set.
    text_encoder: Option<MageTextEncoder>,
    vae: MageVae,
    /// The pre-loaded frozen DiT the LoRA/LoKr path adapts, in an `Option` so the **full base
    /// fine-tune** path (sc-14056) can drop it: that path trains its own f32 master-weight map seeded
    /// from the raw checkpoint and never touches this bf16 copy, so freeing it before the loop reclaims
    /// its multi-GB residency for the (much larger) full-tune working set. Always `Some` on the LoRA
    /// path.
    transformer: Option<MageTransformer>,
    /// The diffusers snapshot root (`text_encoder/ transformer/ vae/`) this trainer loaded from. The
    /// full base fine-tune path (sc-14056) re-reads the raw `transformer/` checkpoint from here to seed
    /// its f32 master-weight map and copies `transformer/config.json` beside the saved checkpoint so it
    /// reloads through [`MageTransformer::load`]. The LoRA/LoKr path never touches it.
    root: PathBuf,
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
        // The one trainer with a full base fine-tune path today (sc-14056 / epic 14034): it trains
        // every DiT weight and writes a full checkpoint rather than an adapter.
        supports_full_finetune: true,
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
    Ok(Box::new(MageFlowTrainer {
        descriptor: trainer_descriptor(),
        text_encoder: Some(crate::text_encoder::load(root)?),
        vae: crate::vae::load(root.join("vae"), VaePart::Both, Dtype::Bfloat16)?,
        transformer: Some(MageTransformer::load(root.join("transformer"))?),
        root: root.clone(),
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
    // `rank` parameterizes the LoRA/LoKr adapter; a full base fine-tune (sc-14056) trains the dense
    // weights directly, so rank is irrelevant there and a `0` is not an error.
    if !req.config.full_finetune && req.config.rank == 0 {
        return Err("mage_flow_base trainer: rank must be > 0".into());
    }
    // Gradient (activation) checkpointing is NOT ported for Mage yet (sc-14989). On the LoRA path the
    // flag has always been a silent no-op (sc-14055 shipped that way, and the SceneWorks Mage target
    // sets it by default), so leave that behavior alone. On the FULL base fine-tune path silence is
    // dangerous rather than merely untidy: this flag is the advertised mitigation for exactly the
    // memory wall this path hits, and an MLX overcommit is an uncatchable SIGKILL — so a caller who
    // asks for it and is silently ignored gets a hard process kill instead of the help they requested.
    // Say so plainly, and name the levers that do work today.
    if req.config.full_finetune && req.config.gradient_checkpointing {
        return Err(
            "mage_flow_base trainer: gradient (activation) checkpointing is not yet \
                    available for Mage-Flow (sc-14989), so it cannot be honored for a full base \
                    fine-tune. Lower the training resolution, or train a LoRA/LoKr adapter instead \
                    — do not rely on this flag to fit a production-resolution full fine-tune."
                .into(),
        );
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
        // Shared full-base-fine-tune floor (sc-14056). This trainer advertises
        // `supports_full_finetune`, so the floor is a pass-through here — it is routed through
        // anyway so the capability claim and the acceptance stay one fact (and the conformance
        // suite's validate-honesty check exercises the same seam for every family).
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        validate_request(req)?;
        // `lora_target_modules` only scopes the LoRA/LoKr adapter; a full base fine-tune trains every
        // DiT weight, so the target-resolution guard below does not apply to it.
        if !req.config.full_finetune {
            // Non-default `lora_target_modules` that match no adaptable module on the DiT would resolve
            // to an empty target set — a full-length run that trains zero parameters yet "succeeds".
            if resolve_target_paths(self.transformer_ref()?, &req.config).is_empty() {
                return Err(format!(
                    "mage_flow_base trainer: lora_target_modules {:?} matched no adaptable module on \
                     the DiT",
                    req.config.lora_target_modules
                )
                .into());
            }
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
    /// The frozen DiT the LoRA/LoKr path adapts. `Some` on that path by construction (only the full
    /// base fine-tune path frees it); a typed error rather than a panic if that invariant is ever
    /// violated (F-008).
    fn transformer_ref(&self) -> Result<&MageTransformer> {
        self.transformer.as_ref().ok_or_else(|| {
            mlx_gen::Error::Msg(
                "mage_flow_base trainer: DiT was freed (only the full fine-tune path frees it)"
                    .into(),
            )
        })
    }

    /// `&mut` sibling of [`transformer_ref`](Self::transformer_ref).
    fn transformer_mut(&mut self) -> Result<&mut MageTransformer> {
        self.transformer.as_mut().ok_or_else(|| {
            mlx_gen::Error::Msg(
                "mage_flow_base trainer: DiT was freed (only the full fine-tune path frees it)"
                    .into(),
            )
        })
    }

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

        let compute_dtype = resolve_compute_dtype(&cfg.train_dtype);

        // Full base fine-tune (sc-14056) trains every DiT weight; dispatch before the LoRA-only setup
        // below (which casts the frozen base and builds adapter factors). The full path seeds its own
        // f32 master weights from the raw checkpoint instead of freezing `self.transformer`.
        if cfg.full_finetune {
            return self.train_full_impl(req, edge, grid, compute_dtype, on_progress);
        }

        self.transformer_mut()?.cast_weights(compute_dtype)?;

        // --- prepare → cache: VAE-latents + prompt-embeds into memory before the loop ---
        let (cache, sample_caps) = self.prepare_caches(req, edge, grid, on_progress)?;

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(self.transformer_ref()?, cfg);
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(
                    self.transformer_mut()?,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.seed,
                )?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    self.transformer_mut()?,
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
                self.transformer.as_mut().ok_or_else(|| {
                    mlx_gen::Error::Msg("mage_flow_base trainer: DiT was freed".into())
                })?,
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
                    self.transformer_mut()?,
                    &params,
                    alpha,
                    rank,
                    lora_dtype,
                    LOKR_DTYPE,
                )?;
                let transformer = self.transformer_ref()?;
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
                        transformer,
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

    /// Cache the whole dataset (Mage-VAE latents + Qwen3-VL prompt embeddings) into memory, pre-encode
    /// the preview-sample prompts, then **drop the Qwen encoder** before the memory-heavy train loop —
    /// the prepare→cache lifecycle shared by the LoRA/LoKr and full base fine-tune paths (sc-14056).
    /// Returns the per-sample cache and the pre-encoded preview prompts (empty when sampling is off).
    fn prepare_caches(
        &mut self,
        req: &TrainingRequest,
        edge: u32,
        grid: i32,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<CachedSample>, Vec<SamplePrompt>)> {
        let cfg = &req.config;
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
        // model. Skipped when sampling is off (the default) or the run is already cancelled.
        let sample_caps: Vec<SamplePrompt> = if cfg.sample_every > 0
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
        Ok((cache, sample_caps))
    }

    /// The **full base fine-tune** (sc-14056): train *every* DiT weight against the same rectified
    /// flow-matching objective as the LoRA path (`L = ‖v_θ(z_σ,σ,τ) − (ε − z)‖²`), producing a full
    /// fine-tuned transformer checkpoint that reloads through [`MageTransformer::load`] and infers.
    ///
    /// The trainable state is an f32 **master-weight** map ([`load_master_weights`]) seeded from the
    /// raw `transformer/` checkpoint, keyed exactly as the diffusers layout. The model crates forward
    /// over raw `Array`s (not mlx-rs `Module`s), so — mirroring the adapter path's functional-autograd
    /// injection — the differentiable step reconstructs a [`MageTransformer`] from the (compute-dtype
    /// cast) master map *inside* the autograd trace ([`compute_full_loss_grads`]); the whole state dict
    /// is the `keyed_value_and_grad` argument, so the gradient reaches every weight. Master weights stay
    /// f32; the forward runs at `compute_dtype` (bf16 mixed precision or f32) with the loss/grads/
    /// optimizer in f32 (the master-weights pattern the LoRA factors already use).
    ///
    /// The dense retained-graph forward holds the whole model, so this is only affordable at small
    /// resolution / tiny dataset until gradient (activation) checkpointing lands (sc-14989); a
    /// production-resolution full-tune is refused up front by the SceneWorks platform/tier memory gate.
    fn train_full_impl(
        &mut self,
        req: &TrainingRequest,
        edge: u32,
        grid: i32,
        compute_dtype: Dtype,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        let cfg = &req.config;
        let transformer_dir = self.root.join("transformer");

        // Free the pre-loaded bf16 DiT (LoRA-only) before allocating the much larger full-tune working
        // set: the full path never adapts it — it trains its own f32 master map — so holding the ~8 GB
        // bf16 copy resident alongside the master weights + optimizer state + retained backward graph is
        // pure waste that, on a multi-billion-parameter model, is the difference between fitting unified
        // memory and spilling to swap.
        self.transformer = None;
        mlx_rs::memory::clear_cache();

        // Seed the f32 master weights from the raw checkpoint (keys = diffusers layout). No separate
        // "probe" reconstruction here: [`load_trainer`] already built a [`MageTransformer`] from this
        // exact checkpoint (which requires every consumed key and cross-checks the geometry against the
        // config), so a corrupt or partially-remapped checkpoint has already failed loudly before this
        // point — and reconstructing a second full model here purely to re-validate would transiently
        // double the resident weights, the single largest term in this path's peak.
        let (mut params, dit_cfg) = load_master_weights(&transformer_dir)?;
        // The msrope table is a weight-independent constant (built from `dit_cfg`); build it once and
        // derive each step's pack context from it, so the per-step in-trace reconstruction never
        // recomputes it.
        let rope = MsRope::from_config(&dit_cfg)?;

        // Cache the dataset + drop the text encoder (shared with the LoRA path).
        let (cache, sample_caps) = self.prepare_caches(req, edge, grid, on_progress)?;

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
            .unwrap_or("finetune")
            .to_string();

        // --- resume: continue from the latest full-state snapshot of THIS run (heavy — the snapshot is
        // the whole state dict + optimizer state, via the shared param-map-generic resume machinery). ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[sc-14056] full fine-tune resuming from step {start_step} (update {update_idx})"
                );
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
            // One packed segment: this image's latent grid + its caption tokens (native-res per-sample).
            let layout = PackLayout::generation(
                vec![ImgShape::latent(grid, grid)],
                vec![sample.txt_tokens],
            )?;
            let ctx = PackContext::new(layout, &rope)?;
            let (loss, grads) = compute_full_loss_grads(
                &dit_cfg,
                &params,
                sample,
                &ctx,
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
                // A full-tune intermediate checkpoint is a whole reloadable transformer dir (heavy).
                let ckpt_dir = req.output_dir.join(format!("{stem}-step{step:06}"));
                save_full_checkpoint(&params, &transformer_dir, &ckpt_dir)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // Periodic preview samples from the in-progress weights (reconstruct a bf16 model from the
            // current masters). Best-effort: a render failure must not abort the run.
            if cfg.sample_every > 0 && !sample_caps.is_empty() && step % cfg.sample_every == 0 {
                match params_to_weights(&params, Dtype::Bfloat16)
                    .map_err(mlx_gen::Error::from)
                    .and_then(|w| MageTransformer::from_weights(&w, dit_cfg.clone()))
                {
                    Ok(model) => {
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
                            match render_sample(
                                &model,
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
                                    "[sc-14056] {MODEL_ID} full-tune preview failed at step {step} \
                                     (prompt {}): {e} — skipping this preview, training continues",
                                    i + 1
                                ),
                            }
                        }
                    }
                    Err(e) => eprintln!(
                        "[sc-14056] {MODEL_ID} full-tune preview reconstruct failed at step {step}: \
                         {e} — skipping this cadence, training continues"
                    ),
                }
            }

            // Release the step's transient buffers to the OS before the next iteration. The full-tune
            // step allocates a whole dense retained graph plus (on the bf16 path) a fresh reconstructed
            // model each step; MLX pools freed buffers rather than returning them, so without this the
            // pool grows unbounded across steps and a multi-billion-parameter run spills to swap. This
            // is unique to the full path — the LoRA path's frozen base and tiny adapter don't churn.
            mlx_rs::memory::clear_cache();
        }

        // Cancelled before completing a single step: nothing was trained — surface the cancellation
        // rather than writing a byte-for-byte copy of the base checkpoint as a "fine-tune".
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save the final full checkpoint (bf16) as a reloadable transformer dir ---
        on_progress(TrainingProgress::Saving);
        let weights_path = save_full_checkpoint(&params, &transformer_dir, &req.output_dir)?;
        Ok(TrainingOutput {
            adapter_path: weights_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Resolve the training compute dtype from the config's free-form `train_dtype`: `bf16` (the published
/// checkpoint's dtype and the ecosystem-standard mixed precision — the trainable factors / loss /
/// grads / optimizer stay f32, the master-weights pattern) or `f32`. Unrecognized / empty means f32,
/// which widens the bf16 base losslessly rather than silently narrowing. Shared by both train paths.
fn resolve_compute_dtype(train_dtype: &str) -> Dtype {
    let t = train_dtype.trim();
    if t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16") {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    }
}

/// Seed the f32 **master-weight** map for a full base fine-tune (sc-14056) from the raw `transformer/`
/// checkpoint, plus the DiT config. Keys are the diffusers layout (`img_in.weight`,
/// `transformer_blocks.{i}.…`, `proj_out.weight`, …) so the map round-trips through
/// [`MageTransformer::from_weights`] (reconstruct-in-trace) and [`save_full_checkpoint`] (save). Every
/// tensor is widened to f32 (master weights); the published checkpoint is bf16.
fn load_master_weights(transformer_dir: &Path) -> Result<(LoraParams, MageFlowConfig)> {
    let json = std::fs::read_to_string(transformer_dir.join(TRANSFORMER_CONFIG_FILE))?;
    let cfg = MageFlowConfig::from_transformer_config_json(&json)?;
    let weights = Weights::from_file(transformer_dir.join(TRANSFORMER_WEIGHTS_FILE))?;
    let keys: Vec<String> = weights.keys().map(str::to_string).collect();
    let mut params: LoraParams = HashMap::with_capacity(keys.len());
    for key in keys {
        let w = weights.require(&key)?.as_dtype(Dtype::Float32)?;
        params.insert(Rc::from(key.as_str()), w);
    }
    eval(params.values())?;
    Ok((params, cfg))
}

/// Build a diffusers-keyed [`Weights`] from the master params for in-trace reconstruction, casting each
/// master weight to the forward `compute_dtype` (bf16 mixed precision or f32). Differentiable: the cast
/// is a traced op, so the gradient flows back to the f32 masters (master-weights pattern). Called every
/// step, so it stays allocation-light — a shallow `Array` handle clone when the dtype already matches.
fn params_to_weights(params: &LoraParams, compute_dtype: Dtype) -> MlxResult<Weights> {
    let mut map: HashMap<String, Array> = HashMap::with_capacity(params.len());
    for (k, v) in params {
        let w = if v.dtype() == compute_dtype {
            v.clone()
        } else {
            v.as_dtype(compute_dtype)?
        };
        map.insert(k.to_string(), w);
    }
    Ok(Weights::from_map(map))
}

/// One forward+backward over the **entire** DiT for the full base fine-tune: reconstruct the model from
/// the (compute-dtype cast) master params inside the autograd trace, pack this sample, run the training
/// forward, regress the velocity toward `noise − x0` (`ε − z`; see [`build_batch`]), return
/// `(loss, grads)` where `grads` is keyed exactly as `params`.
#[allow(clippy::too_many_arguments)]
fn compute_full_loss_grads(
    cfg: &MageFlowConfig,
    params: &LoraParams,
    sample: &CachedSample,
    ctx: &PackContext,
    sigma: f32,
    noise: &Array,
    mae: bool,
    compute_dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (x_t, target, _timestep) = build_batch(&sample.latent_tokens, noise, sigma)?;
    let sigma_arr = Array::from_slice(&[sigma], &[1]); // one entry per packed segment
    let txt = sample.txt.clone();
    let ctx = ctx.clone();
    let cfg = cfg.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // NEVER check the cancel flag inside this traced grad closure (F-008 / sc-14055): it returns
        // `MlxResult`, so an early-out would lose the typed `Canceled`. The grad graph is one atomic
        // unit; cancellation is the caller's job at the step boundary.
        let weights =
            params_to_weights(&p, compute_dtype).map_err(|e| Exception::custom(e.to_string()))?;
        let model = MageTransformer::from_weights(&weights, cfg.clone())
            .map_err(|e| Exception::custom(e.to_string()))?;
        let v = model
            .forward_train(&x_t, &txt, &sigma_arr, &ctx)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let diff = subtract(&v, &target)?;
        // `mean(None)` → 0-d scalar (grad needs a scalar cotangent). v(bf16 on the bf16 path) − target
        // (f32) promotes to f32, so the loss/grads are f32 (master weights).
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

/// Persist the full fine-tuned DiT as a reloadable `transformer/`-shaped directory: the master weights
/// cast back to the published **bf16** into `diffusion_pytorch_model.safetensors`, plus a copy of the
/// source `transformer/config.json`, so [`MageTransformer::load`] reads `out_dir` directly. Returns the
/// weights path. A `networkType=full` / `family` metadata stamp records the artifact's provenance.
fn save_full_checkpoint(
    params: &LoraParams,
    src_transformer_dir: &Path,
    out_dir: &Path,
) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;
    let mut casted: Vec<(String, Array)> = Vec::with_capacity(params.len());
    for (k, v) in params {
        casted.push((k.to_string(), v.as_dtype(Dtype::Bfloat16)?));
    }
    eval(casted.iter().map(|(_, v)| v))?;
    let entries: Vec<(String, &Array)> = casted.iter().map(|(k, v)| (k.clone(), v)).collect();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "full".to_string());
    meta.insert("family".to_string(), FAMILY.to_string());
    let weights_path = out_dir.join(TRANSFORMER_WEIGHTS_FILE);
    Array::save_safetensors(entries, Some(&meta), &weights_path)?;
    // Copy config.json so the saved dir reloads as a transformer component tree.
    std::fs::copy(
        src_transformer_dir.join(TRANSFORMER_CONFIG_FILE),
        out_dir.join(TRANSFORMER_CONFIG_FILE),
    )?;
    Ok(weights_path)
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
    fn descriptor_advertises_lora_lokr_full_finetune_no_control_on_the_base_target() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "mage_flow_base");
        assert_eq!(d.family, FAMILY);
        assert_eq!(d.backend, "mlx");
        assert!(d.supports_lora && d.supports_lokr);
        assert!(!d.supports_control);
        // sc-14056: this is the one trainer with a full base fine-tune path, so it must ADVERTISE it —
        // the shared `validate_full_finetune_request` floor turns a `false` here into a typed reject of
        // every full-tune request, which would strand the capability behind its own capability flag.
        assert!(d.supports_full_finetune);
    }

    /// The capability claim and the acceptance must be one fact: because this trainer advertises
    /// `supports_full_finetune`, the shared floor must let a full request through `validate` — and it
    /// must reject one on a descriptor that does not advertise it. Asserting only the boolean above
    /// would not prove `validate` routes through the floor at all.
    #[test]
    fn validate_routes_through_the_full_finetune_floor() {
        let desc = trainer_descriptor();
        let req = base_request(
            one_item(),
            TrainingConfig {
                full_finetune: true,
                ..Default::default()
            },
        );
        assert!(
            gen_core::train::validate_full_finetune_request(&desc, &req).is_ok(),
            "the Mage trainer advertises the full path, so the floor must admit it"
        );

        let mut adapter_only = desc;
        adapter_only.supports_full_finetune = false;
        let err = gen_core::train::validate_full_finetune_request(&adapter_only, &req).unwrap_err();
        assert!(
            matches!(err, gen_core::Error::Unsupported(_)),
            "a trainer without the full path must reject, not silently train an adapter, got {err:?}"
        );
    }

    #[test]
    fn resolve_compute_dtype_maps_bf16_and_defaults_to_f32() {
        assert_eq!(resolve_compute_dtype("bf16"), Dtype::Bfloat16);
        assert_eq!(resolve_compute_dtype(" BFloat16 "), Dtype::Bfloat16);
        assert_eq!(resolve_compute_dtype("f32"), Dtype::Float32);
        assert_eq!(resolve_compute_dtype("float32"), Dtype::Float32);
        // Empty / unrecognized widen the bf16 base to f32 (never a silent narrowing).
        assert_eq!(resolve_compute_dtype(""), Dtype::Float32);
        assert_eq!(resolve_compute_dtype("fp8"), Dtype::Float32);
    }

    #[test]
    fn validate_full_finetune_ignores_rank_but_keeps_the_other_guards() {
        // A full base fine-tune trains dense weights, so `rank` is irrelevant — a `0` is not an error
        // (it IS for the adapter path). The dataset/steps/optimizer/sampler/loss guards still apply.
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                full_finetune: true,
                rank: 0,
                ..Default::default()
            }
        ))
        .is_ok());
        // rank 0 is still rejected on the (default) adapter path.
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                rank: 0,
                ..Default::default()
            }
        ))
        .is_err());
        // The shared guards still bite in full mode.
        assert!(validate_request(&base_request(
            vec![],
            TrainingConfig {
                full_finetune: true,
                ..Default::default()
            }
        ))
        .is_err());
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                full_finetune: true,
                steps: 0,
                ..Default::default()
            }
        ))
        .is_err());
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                full_finetune: true,
                optimizer: "nope".into(),
                ..Default::default()
            }
        ))
        .is_err());
    }

    #[test]
    fn full_finetune_rejects_unported_gradient_checkpointing_but_lora_is_unaffected() {
        // sc-14989 is not ported. On the FULL path this flag is the advertised mitigation for the
        // exact memory wall the path hits, and an MLX overcommit is an uncatchable SIGKILL — so
        // silently ignoring it would hand the caller a hard process kill instead of the help they
        // asked for. Reject it with a message that names the story and the levers that DO work.
        let err = validate_request(&base_request(
            one_item(),
            TrainingConfig {
                full_finetune: true,
                gradient_checkpointing: true,
                ..Default::default()
            },
        ))
        .expect_err("a full fine-tune must not silently ignore gradient checkpointing");
        let msg = err.to_string();
        assert!(
            msg.contains("sc-14989"),
            "must name the tracking story: {msg}"
        );
        assert!(
            msg.contains("resolution") && msg.contains("LoRA"),
            "must name the levers that work today: {msg}"
        );

        // The LoRA/LoKr path is deliberately UNCHANGED — the flag has always been a no-op there
        // (sc-14055), and the SceneWorks Mage target sets it by default, so erroring would break the
        // shipped adapter path.
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                gradient_checkpointing: true,
                ..Default::default()
            }
        ))
        .is_ok());

        // A full fine-tune WITHOUT the flag is fine.
        assert!(validate_request(&base_request(
            one_item(),
            TrainingConfig {
                full_finetune: true,
                gradient_checkpointing: false,
                ..Default::default()
            }
        ))
        .is_ok());
    }

    /// sc-14056 full base fine-tune **convergence** — a real run on this Mac's MLX GPU that trains
    /// EVERY DiT weight against a FIXED batch (one latent/caption/sigma/noise). The stationary objective
    /// isolates the full-parameter loop mechanics (reconstruct-in-trace → autograd over the whole state
    /// dict → optimizer → re-inject) from the per-step schedule variance, so the loss must fall. Runs at
    /// f32 (so `params_to_weights` is a zero-copy shallow reuse of the master weights) with a per-step
    /// `clear_cache` — the multi-billion-parameter state cannot afford the pooled-buffer growth.
    ///
    /// Run (this Mac has MLX GPU):
    ///   MAGE_BASE_SNAPSHOT=/path/to/Mage-Flow-Base \
    ///     cargo test -p mlx-gen-mage --lib full_finetune_overfits_a_fixed_batch -- --ignored --nocapture
    #[test]
    #[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT)"]
    fn full_finetune_overfits_a_fixed_batch() {
        let Ok(root) = std::env::var("MAGE_BASE_SNAPSHOT") else {
            return;
        };
        let root = PathBuf::from(&root);

        // Seed the f32 master weights the full path trains, plus VAE/TE to encode one fixed sample.
        let (mut params, dit_cfg) = load_master_weights(&root.join("transformer")).unwrap();
        let vae = crate::vae::load(root.join("vae"), VaePart::Both, Dtype::Bfloat16).unwrap();
        let text_encoder = crate::text_encoder::load(&root).unwrap();

        let edge = 64u32;
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
        mlx_rs::memory::clear_cache();
        let sample = CachedSample {
            latent_tokens,
            txt,
            txt_tokens,
        };

        // A FIXED sigma + noise ⇒ a stationary objective. Build the pack context once.
        let rope = MsRope::from_config(&dit_cfg).unwrap();
        let layout =
            PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![sample.txt_tokens])
                .unwrap();
        let ctx = PackContext::new(layout, &rope).unwrap();
        let noise = random::normal::<f32>(
            &[1, grid * grid, LATENT_CHANNELS],
            None,
            None,
            Some(&random::key(12).unwrap()),
        )
        .unwrap();
        let sigma = 0.5f32;

        // A full fine-tune moves EVERY weight, so it is far more sensitive than a LoRA adapter: 1e-4
        // overshoots and diverges (observed), 1e-5 descends cleanly.
        let mut opt = TrainOptimizer::from_config("adamw", 1e-5, 0.0).unwrap();
        let mut losses = Vec::new();
        for _ in 0..40 {
            let (loss, grads) = compute_full_loss_grads(
                &dit_cfg,
                &params,
                &sample,
                &ctx,
                sigma,
                &noise,
                false,
                Dtype::Float32,
            )
            .unwrap();
            losses.push(loss);
            opt.step(&mut params, &grads).unwrap();
            eval(params.values()).unwrap();
            mlx_rs::memory::clear_cache();
        }
        println!(
            "[full-overfit] fixed-batch loss {:.5} -> {:.5}\n[full-overfit] curve: {losses:?}",
            losses[0],
            losses.last().unwrap()
        );
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "no NaN/Inf during the full-parameter overfit"
        );
        assert!(
            *losses.last().unwrap() < losses[0],
            "a stationary full base fine-tune must reduce the loss: {} -> {}",
            losses[0],
            losses.last().unwrap()
        );
    }

    /// sc-14056 full base fine-tune **e2e** — the whole [`Trainer::train`] path (dataset caching → the
    /// full-parameter loop → save) on this Mac's MLX GPU, then reloads the produced checkpoint and
    /// proves it (a) is a loadable `transformer/` dir and (b) infers a velocity that DIFFERS from the
    /// base (every weight actually moved). The random per-step schedule makes the loss noisy, so the
    /// gate is the reload + weights-moved proof, not a monotone curve (the fixed-batch test above
    /// gates convergence). f32 + lr 1e-5 keeps the multi-billion-parameter run stable and in memory.
    ///
    /// Run (this Mac has MLX GPU):
    ///   MAGE_BASE_SNAPSHOT=/path/to/Mage-Flow-Base \
    ///     cargo test -p mlx-gen-mage --lib full_finetune_runs_e2e_and_reloads -- --ignored --nocapture
    #[test]
    #[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT)"]
    fn full_finetune_runs_e2e_and_reloads() {
        let Ok(root) = std::env::var("MAGE_BASE_SNAPSHOT") else {
            return;
        };
        let root = PathBuf::from(&root);
        let tmp = std::env::temp_dir().join(format!("mage_full_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // A tiny dataset: two solid-colour swatches (a real, low-entropy in-distribution target).
        let mut items = Vec::new();
        for (i, colour) in [[200u8, 40, 40], [40, 80, 200]].into_iter().enumerate() {
            let path = tmp.join(format!("swatch_{i}.png"));
            let mut im = image::RgbImage::new(96, 96);
            for px in im.pixels_mut() {
                *px = image::Rgb(colour);
            }
            im.save(&path).unwrap();
            items.push(TrainingItem::captioned(
                path,
                format!("a solid colour swatch {i}"),
            ));
        }

        let out_dir = tmp.join("out");
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let mut trainer = load_trainer(&spec).unwrap();
        let cfg = TrainingConfig {
            full_finetune: true,
            steps: 10,
            // 64 buckets to a multiple of 32 → grid 64/16 = 4 (16 tokens): keeps the dense
            // retained-graph forward affordable pre-grad-checkpointing (sc-14989).
            resolution: 64,
            // f32 master + f32 forward (no per-step bf16 reconstruct alloc); lr 1e-5 for stability.
            learning_rate: 1e-5,
            train_dtype: "f32".into(),
            save_every: 0,
            sample_every: 0,
            seed: 7,
            ..Default::default()
        };
        let req = TrainingRequest {
            items,
            config: cfg,
            output_dir: out_dir.clone(),
            file_name: "finetune.safetensors".into(),
            trigger_words: vec![],
            cancel: mlx_gen::CancelFlag::new(),
        };

        let mut losses: Vec<f32> = Vec::new();
        let out = trainer
            .train(&req, &mut |p| {
                if let TrainingProgress::Training { step, loss, .. } = p {
                    losses.push(loss);
                    println!("[full-e2e] step {step} loss {loss:.5}");
                }
            })
            .unwrap();
        println!(
            "[full-e2e] {} steps; losses {:?}; checkpoint {}",
            out.steps,
            losses,
            out.adapter_path.display()
        );
        assert_eq!(out.steps, 10);
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "the run must stay numerically stable (no NaN/Inf): {losses:?}"
        );

        // (a) The saved dir is a reloadable transformer component tree.
        assert!(out.adapter_path.ends_with(TRANSFORMER_WEIGHTS_FILE));
        assert!(out_dir.join(TRANSFORMER_CONFIG_FILE).is_file());
        let reloaded = MageTransformer::load(&out_dir).unwrap();
        let base = MageTransformer::load(root.join("transformer")).unwrap();

        // (b) The reloaded DiT infers, and its velocity differs from the base on the same input.
        let grid = 4i32;
        let img_tokens = grid * grid;
        let img = random::normal::<f32>(
            &[1, img_tokens, LATENT_CHANNELS],
            None,
            None,
            Some(&random::key(3).unwrap()),
        )
        .unwrap();
        let txt_tokens = 8i32;
        let txt = random::normal::<f32>(
            &[1, txt_tokens, base.config().context_in_dim],
            None,
            None,
            Some(&random::key(4).unwrap()),
        )
        .unwrap();
        let sigma = Array::from_slice(&[0.5f32], &[1]);
        let layout = generation_layout(&[(grid, grid)], vec![txt_tokens]).unwrap();
        let ctx = reloaded.pack_context(layout).unwrap();
        let v_base = base.forward(&img, &txt, &sigma, &ctx).unwrap();
        let v_tuned = reloaded.forward(&img, &txt, &sigma, &ctx).unwrap();
        let diff = subtract(&v_tuned, &v_base)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap();
        eval([&v_base, &v_tuned, &diff]).unwrap();
        let max_abs = diff.item::<f32>();
        println!("[full-e2e] reloaded-vs-base velocity max_abs = {max_abs}");
        assert!(
            max_abs > 0.0 && max_abs.is_finite(),
            "a full fine-tune must move the DiT: reloaded velocity == base (max_abs {max_abs})"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }
}
