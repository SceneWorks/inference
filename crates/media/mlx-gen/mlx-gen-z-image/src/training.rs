//! LoRA/LoKr *training* on the Z-Image DiT, in pure Rust on mlx-rs (epic 3039).
//!
//! The production [`ZImageTurboTrainer`] (sc-3044) realizes the core [`Trainer`] contract on the
//! REAL 30-block Z-Image transformer. The mechanism (the sc-3042 spike that first proved it has been
//! retired — F-043):
//!
//!   * **Trainable LoRA injection** — the model crates do NOT use mlx-rs's `Module`/`ModuleParameters`
//!     system (hand-rolled `&self` forwards over raw `Array`s, `src/adapters.rs:6`), so training uses
//!     the *functional* autograd: the trainable factors live OUTSIDE the model in a [`LoraParams`],
//!     and each step they are re-injected into the target [`mlx_gen::adapters::AdaptableLinear`]s as a single
//!     `Adapter::Lora` via [`mlx_gen::adapters::AdaptableLinear::set_adapters`]. The injection mirrors the inference
//!     reload (`adapters::loader::install_lora_groups`) op-for-op, so the trained adapter round-trips
//!     through the normal inference path bit-for-bit. The host-generic factor machinery lives in core
//!     [`mlx_gen::train::lora`] (hoisted in sc-3045 so every family trainer shares it); this module
//!     keeps only the Z-Image-specific forward, flow-match noising, and dual-encoder caching.
//!   * **Autograd + optimizer** — `keyed_value_and_grad` over the factor map + `AdamW::update_single`
//!     per parameter + `clip_grad_norm` (proven in `tests/lora_train_probe.rs`).
//!   * **Flow-match velocity target** — the Z-Image `forward()` already negates its raw output
//!     (`transformer.rs:246`) and the denoise loop integrates it as `latents += dσ·v` with
//!     `timestep = 1-σ`, so the regression target for `forward()` is `noise - latents` (the *raw*
//!     diffusers output trains toward `latents - noise`; the negation flips the sign — see
//!     `SceneWorks training_adapters.py:485` `flow_matching_velocity_target`).
//!   * **safetensors out** — PEFT keys `{path}.lora_A.weight` `[r,in]`, `{path}.lora_B.weight`
//!     `[out,r]`, `{path}.alpha`, reloadable by `apply_z_image_adapters`.
//!
//! sc-3043 generalizes this into a reusable `Trainer` surface (dataset/VAE-cache/bucket, checkpoint,
//! LR schedule, the `lora_train` job); sc-3044 hardens it for Z-Image + adds LoKr.

use std::path::{Path, PathBuf};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::{self, BucketSchedule};
use mlx_gen::media::Image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, LoraParams, TrainAdapter,
};
use mlx_gen::train::perceptual::{
    combine_step_loss, step_sample, AuxDriver, Parameterization, PerceptualPath, StepPlan,
};
use mlx_gen::train::tae::TinyDecoderConfig;
// Re-export the `LoraTarget` that `build_lora_targets` returns so the crate's public surface is
// unchanged (the host-generic factor machinery moved to `mlx_gen::train::lora` in sc-3045).
pub use mlx_gen::train::lora::LoraTarget;
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::{
    FlowMatchEuler, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::model::{MODEL_ID, SCHEDULE_SHIFT};
use crate::pipeline::encode_init_latents;
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Z-Image reconstructs its LoKr delta at **bf16** (the bf16-residual inference path); training must
/// match so the adapter round-trips bit-for-bit.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637); the
/// SceneWorks UI sends four (`samplePromptsFromTrigger`). A hard cap bounds the per-cadence cost.
const SAMPLE_PROMPT_CAP: usize = 4;

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma`:
/// `x_t = (1-σ)·x0 + σ·noise`, `target = noise - x0`, `timestep = 1-σ`.
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?; // velocity for the already-negated forward output
    Ok((x_t, target, 1.0 - sigma))
}

// ===========================================================================================
// sc-3044: the production `Trainer` impl — realizes the sc-3043 contract on Z-Image, end to end.
// ===========================================================================================

/// LoRA trainer for Z-Image-Turbo, implementing the core [`Trainer`] surface: a frozen base model
/// (transformer + VAE + text encoder + tokenizer) that caches a captioned image dataset to
/// VAE-latents/prompt-embeds, then runs the functional-autograd LoRA loop with the sc-3043
/// runtime glue (LR schedule, gradient accumulation, checkpoint cadence, cancel, progress bands),
/// and writes a PEFT adapter that round-trips through the inference loader.
pub struct ZImageTurboTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: TextTokenizer,
    /// The Qwen text encoder, in an `Option` so it can be **dropped after the caching loop** (sc-4952,
    /// 32 GB-Mac support): it is idle during training — every prompt is already encoded to the cached
    /// `cap` embedding — yet it is a multi-GB resident. Freeing it before the train loop reclaims that
    /// budget for the DiT working set. Kept at its loaded precision (the caption embedding is the
    /// outlier-sensitive conditioning stream that needs the f32 carve-out under bf16 training, sc-4887,
    /// so we do NOT downcast the encoder here — just free it once its embeddings are cached).
    text_encoder: Option<TextEncoder>,
    vae: Vae,
    transformer: ZImageTransformer,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "z_image",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // LoRA/LoKr only — no control-branch training path (F-006).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123: weight + gradient noise at the shared adapter optimizer update
        // (`adapter_optimizer_update`, sc-24826/sc-24827), multi-resolution buckets (sc-2127),
        // depth anchoring (sc-2125) — the shared decoded-x0 perceptual path (TAEF1 decode →
        // Depth-Anything-V2 → cached round-trip reference) — and subject-masked loss (sc-24828,
        // a per-(item, bucket) weight map cached next to each latent, applied by `reduce_loss`),
        // and the body losses (sc-24832: ViTPose proportion, HybrIK shape, Sapiens normals) on the
        // same perceptual path through the `mlx-gen-perceptual` builder's arms.
        // sc-24833: the VAE anchor (TAEF1 decode → FLUX.2 encoder taps) and E-LatentLPIPS (the
        // FLUX.1 16-channel family weights) through the shared aux-loss builder.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            depth_anchoring: true,
            subject_mask_loss: true,
            // sc-24831: the ArcFace identity + FaceMesh landmark losses, picked up through the
            // same shared perceptual builder arms (no trainer-loop change).
            identity_loss: true,
            face_landmark_loss: true,
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            vae_anchor_loss: true,
            latent_lpips_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// Construct the trainer from a snapshot directory (the diffusers multi-component tree). No
/// quantization — training needs the dense base. Registered via [`mlx_gen::TrainerRegistration`].
///
/// The weights load lazily (sc-2124): construction only checks the spec, so `validate` and `train`'s
/// refusal floors never read weights; see [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    snapshot_root(spec)?;
    Ok(Box::new(
        mlx_gen::train::lazy::LazyTrainer::new(trainer_descriptor(), validate_floors, {
            let spec = spec.clone();
            move || load_weights(&spec)
        })
        .validating_on_base_when(mlx_gen::train::lazy::custom_targets),
    ))
}

/// The snapshot directory a trainer spec names — a single `.safetensors` file is refused.
fn snapshot_root(spec: &LoadSpec) -> Result<&PathBuf> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(mlx_gen::Error::Msg(
            "z_image_turbo trainer expects a snapshot directory (tokenizer/ text_encoder/ \
             transformer/ vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// The weight load behind [`load_trainer`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights(spec: &LoadSpec) -> Result<ZImageTurboTrainer> {
    let root = snapshot_root(spec)?;
    Ok(ZImageTurboTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: crate::loader::load_tokenizer(root)?,
        text_encoder: Some(crate::loader::load_text_encoder(root)?),
        vae: crate::loader::load_vae(root)?,
        transformer: crate::loader::load_transformer(root)?,
    })
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const REGISTRATION = trainer_descriptor => load_trainer
}

/// Recognized `timestep_type` values — the noise-schedule samplers [`sample_sigma`] branches on
/// (`linear`/`uniform`/`weighted`) plus the `sigmoid` default it falls back to. Any other string
/// would silently sample sigmoid (F-041).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values — the high/low-noise tilts [`sample_sigma`] branches on plus
/// the neutral default (`balanced`/`none`/`neutral`) it falls back to.
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
/// Recognized `loss_type` values — `mae`/`l1` select MAE, `mse`/`l2` the MSE default; any other
/// string would silently train MSE (F-041).
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer (mirrors the inference-side `validate_request`). Rejects an
/// empty dataset, zero rank, **zero steps** (F-040 — a 0-step run would otherwise fall straight
/// through to the save and write a no-op `B = 0` identity adapter), an unsupported optimizer, and —
/// rather than letting a typo silently fall back to a default sampler/loss (F-041) — an unrecognized
/// `timestep_type` / `timestep_bias` / `loss_type`. The non-empty target-module resolution (also
/// F-041) is checked in [`Trainer::validate`], which has the loaded DiT to match suffixes against.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("z_image_turbo trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("z_image_turbo trainer: rank must be > 0".into());
    }
    if req.config.steps == 0 {
        return Err("z_image_turbo trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "z_image_turbo trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&req.config.timestep_type).as_str()) {
        return Err(format!(
            "z_image_turbo trainer: timestep_type '{}' is not recognized (supported: {})",
            req.config.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&req.config.timestep_bias).as_str()) {
        return Err(format!(
            "z_image_turbo trainer: timestep_bias '{}' is not recognized (supported: {})",
            req.config.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&req.config.loss_type).as_str()) {
        return Err(format!(
            "z_image_turbo trainer: loss_type '{}' is not recognized (supported: {})",
            req.config.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

/// Every weights-free [`Trainer::validate`] floor — all of it but the target-module match.
fn validate_floors(descriptor: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    // Shared control-training floor (F-006): a LoRA-only trainer must reject a control-branch
    // request (typed `Unsupported`) rather than silently training a plain adapter.
    gen_core::train::validate_control_request(descriptor, req)?;
    // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
    // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
    gen_core::train::validate_full_finetune_request(descriptor, req)?;
    // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
    // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
    gen_core::train::validate_training_techniques(descriptor, req)?;
    gen_core::train::validate_edit_request(descriptor, req)?;
    validate_request(req)?;
    Ok(())
}

impl Trainer for ZImageTurboTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_floors(self.descriptor(), req)?;
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would resolve
        // to an empty target set — a full-length run that trains zero parameters yet "succeeds"
        // (F-041). Catch it here, where the loaded DiT is available to match suffixes against.
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "z_image_turbo trainer: lora_target_modules {:?} matched no adaptable module on the \
                 DiT",
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
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl ZImageTurboTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are
        // off). Memory guards and previews size for the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let edge = preflight_edge(&edges);

        // sc-4887 — training compute dtype. bf16 halves the activation working set (and the resident
        // base) and is the ecosystem-standard mixed precision; the trainable factors / loss / grads /
        // optimizer stay f32 (master-weights). The cast is destructive (f32→bf16), so a trainer that
        // was already cast cannot honor a later f32 request — reload instead of silently training at
        // the wrong precision.
        let use_bf16 = cfg.train_dtype.trim().eq_ignore_ascii_case("bf16")
            || cfg.train_dtype.trim().eq_ignore_ascii_case("bfloat16");
        let compute_dtype = if use_bf16 {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        if !use_bf16 && self.transformer.compute_dtype() == Some(Dtype::Bfloat16) {
            return Err(
                "z_image_turbo trainer: this trainer instance was already cast to bf16 by a \
                 previous run; reload the trainer for f32 training"
                    .into(),
            );
        }

        // sc-4874 — fail-fast pre-flight memory guard. The dense (non-block-checkpointed) first step
        // materializes the whole forward graph in one MLX `eval`; at high resolution that working set
        // can exceed unified memory and the OS hard-kills the worker with an UNCATCHABLE SIGKILL (no
        // in-process error, the run just appears to hang at the last cached latent). We cannot catch
        // that kill, so we predict it and refuse up front with an actionable, catchable error —
        // BEFORE the (~minutes-long) latent caching — when gradient checkpointing is not enabled.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        // Epic 2123 E7: the training-time auxiliary models (TAEF1 + Depth-Anything-V2) count
        // against the budget on BOTH paths — the default Z-Image preset trains checkpointed, so a
        // dense-only check would let a depth job skip admission entirely.
        // One cached depth reference per (item, bucket) entry, sized here at the largest edge.
        let aux_gb = perceptual_footprint_gb(cfg, edge, req.items.len() * edges.len());
        if !will_checkpoint || aux_gb > 0.0 {
            preflight_memory_guard(cfg, edge, use_bf16, aux_gb, will_checkpoint)?;
        }

        if use_bf16 {
            self.transformer.cast_weights(Dtype::Bfloat16)?;
        }

        // Epic 2123 depth anchoring (sc-2125): load the frozen TAEF1 decoder + Depth-Anything-V2
        // before the (minutes-long) caching pass, so a missing/corrupt aux checkpoint fails fast.
        let perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: VAE-latents + prompt-embeds into memory before the loop ---
        on_progress(TrainingProgress::LoadingModel); // base model is already resident from load_trainer
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). Each entry: clean latent,
        // caption embeds, and (subject-masked loss, sc-24828) that bucket's latent loss-weight map
        // — `None` when the technique is off.
        let mut cache: Vec<CacheEntry> = Vec::with_capacity(req.items.len() * edges.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            // sc-24828: the item's subject mask is read + checked once, then resampled per bucket.
            let mask = PreparedSubjectMask::load_if_enabled(
                "z_image_turbo trainer",
                item,
                cfg.subject_mask_loss.as_ref(),
            )?;
            let text_encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "z_image_turbo trainer: text encoder already freed (caching after train loop)"
                        .into(),
                )
            })?;
            let cap = crate::pipeline::encode_prompt(
                &self.tokenizer,
                text_encoder,
                &item.caption,
                "z_image_turbo trainer",
                // Trainers never select a memory rung: the resident encoder is the training path.
                None,
            )?;
            eval([&cap])?;
            // clean latent [16,1,h,w] per bucket edge
            cache_item_buckets(
                &edges,
                mask.as_ref(),
                &cap,
                |edge| encode_init_latents(&self.vae, &img, edge, edge),
                &mut cache,
            )?;
        }
        if cache.is_empty() {
            // sc-4895 — disambiguate the two ways the cache ends up empty. A cancel tripped during
            // caching is a *genuine* cancellation → typed `Error::Canceled` (bridged 1:1 to
            // `gen_core::Error::Canceled`, so the conformance suite distinguishes it from failure);
            // an empty cache with no cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("z_image_turbo trainer: no usable dataset items".into());
        }

        // Epic 2123 E8: each image's perceptual reference (TAEF1 decode of its cached clean latent →
        // DA2 depth) is computed exactly once per job, by the `AuxDriver` before the loop.
        // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
        // cropped like the image and resampled onto its decoded size.
        let aux_masks = match perceptual {
            Some(_) => mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
                "z_image_turbo trainer",
                &req.items,
                cfg,
                edges.len(),
                CropBox::center_square,
            )?,
            None => None,
        };

        // sc-5637 — pre-encode the preview-sample prompts while the Qwen encoder is still resident
        // (it is freed just below, sc-4952). Each `sample_every` cadence reuses these cached cap
        // embeddings to render previews from the in-progress adapter, so the encoder need not stay
        // loaded. Skipped entirely when sampling is off (the default) or the run is already cancelled.
        let sample_caps: Vec<(String, Array)> = if cfg.sample_every > 0
            && !cfg.sample_prompts.is_empty()
            && !req.cancel.is_cancelled()
        {
            let text_encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg(
                    "z_image_turbo trainer: text encoder already freed (sample pre-encode)".into(),
                )
            })?;
            let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                // Match the txt2img generate path: round caption embeddings to the DiT compute
                // dtype (bf16) when training bf16 so the preview denoise runs the parity-proven
                // single-dtype stream (f32 stays f32).
                let cap = crate::pipeline::encode_prompt(
                    &self.tokenizer,
                    text_encoder,
                    prompt,
                    "z_image_turbo trainer (sample)",
                    // Trainers never select a memory rung: the resident encoder is the training path.
                    None,
                )?;
                let cap = if compute_dtype == Dtype::Float32 {
                    cap
                } else {
                    cap.as_dtype(compute_dtype)?
                };
                eval([&cap])?;
                caps.push((prompt.clone(), cap));
            }
            caps
        } else {
            Vec::new()
        };

        // sc-4952 (32 GB-Mac support) — every prompt is now encoded into `cache`, so the Qwen text
        // encoder is dead weight for the rest of the run. Drop it and evict its buffers before the
        // train loop, reclaiming that resident for the DiT working set.
        self.text_encoder = None;
        mlx_rs::memory::clear_cache();

        // sc-5637 — the preview-sample scheduler (constant; z-image is guidance-distilled so the
        // configured sample guidance scale is inert, matching the txt2img generate path). Built once,
        // only when sampling is enabled.
        let sample_scheduler = (!sample_caps.is_empty()).then(|| {
            FlowMatchEuler::for_static_shift(cfg.sample_steps.max(1) as usize, SCHEDULE_SHIFT)
        });

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
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };

        // sc-4874 — gradient checkpointing. Collect, per MAIN block, the adapter-routable LOCAL paths
        // trained on it (e.g. `"attention.to_q"`); the long unified main stack is where the
        // first-step activation memory concentrates, so that is what we checkpoint. Refiner/embedder
        // targets are not collected here — they train through ordinary autograd in the
        // (non-checkpointed) pre-main forward.
        let n_layers = self.transformer.cfg.n_layers;
        let mut main_block_local_targets: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("layers.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        main_block_local_targets[i].push(local.to_string());
                    }
                }
            }
        }
        // Gradient checkpointing is an OPT-IN OPTION (the SceneWorks "Gradient Checkpointing"
        // toggle), never auto-forced — a run that would OOM is caught instead by the fail-fast
        // pre-flight guard below, which surfaces a catchable error and *recommends* this flag rather
        // than silently changing the user's training dynamics. Only the LoRA path is checkpointed
        // today — LoKr (a distinct Kronecker reconstruction) falls back to the dense path (follow-up).
        let is_lora = matches!(adapter, TrainAdapter::Lora { .. });
        let use_checkpoint = is_lora && cfg.gradient_checkpointing;
        let checkpoint_main: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&main_block_local_targets)
        } else {
            None
        };
        // sc-4886 — attention-segment checkpointing is ALWAYS on in training (LoRA and LoKr): it is
        // numerically identical to the retained backward (same decomposed attention math, recomputed)
        // and removes the dominant seq² per-block retention — the flash-backward surrogate every
        // torch trainer gets from its fused SDPA kernel. When whole-block checkpointing is on, the
        // main stack's flag goes OFF (the block recompute already covers attention; nesting would
        // recompute it twice for no memory win) — the refiners are never block-checkpointed, so
        // theirs stays on.
        self.transformer.set_sdpa_checkpoint(!use_checkpoint, true);
        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
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

        // --- resume (F-125): continue from the latest snapshot of THIS adapter in output_dir, if any ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[F-125] resuming from step {start_step} (optimizer update {update_idx})"
                );
            }
        }

        // --- train loop ---
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: the perceptual alternation interleaves optimizer windows over the
        // schedule's epochs (sc-2124) — a pure function of the step, so a resumed run needs no
        // replay.
        let mut aux_driver = match perceptual {
            Some(path) => Some(aux_driver(
                path,
                &cache,
                aux_masks,
                &schedule,
                accum,
                &req.cancel,
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (losses, grads) = run_train_step(
                &mut self.transformer,
                &params,
                &adapter,
                cfg,
                &cache,
                &schedule,
                aux_driver.as_mut(),
                step,
                mae,
                checkpoint_main,
                compute_dtype,
            )?;
            last_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // The final update can fire with fewer than `accum` grads when `steps` isn't a
                // multiple of the accumulation; dividing by `accum` would down-scale that step. Divide
                // by the actual in-window count instead (F-069).
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
                optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx)?;
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

            // sc-5637 — periodic preview samples from the in-progress adapter so the user can watch
            // the LoRA learn (the Python trainer did this; the native port dropped it). Install the
            // current factors as concrete adapters for the forward-only render; the next step's traced
            // `loss_fn` re-installs them, so no teardown is needed. Inference is far cheaper than the
            // training backward that already fits this machine, so no extra memory guard is warranted.
            if let Some(scheduler) = &sample_scheduler {
                if step % cfg.sample_every == 0 {
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
                    for (i, (prompt, cap)) in sample_caps.iter().enumerate() {
                        if req.cancel.is_cancelled() {
                            break;
                        }
                        let sample_seed = cfg
                            .seed
                            .wrapping_add(step as u64)
                            .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                            .wrapping_add(i as u64);
                        // Previews are best-effort: a render failure must NOT abort the (expensive,
                        // long-running) training run — log it and keep training (sc-5637).
                        match crate::pipeline::render_sample(
                            &self.transformer,
                            &self.vae,
                            scheduler,
                            cap,
                            sample_seed,
                            edge,
                            compute_dtype,
                            &req.cancel,
                        ) {
                            Ok(image) => on_progress(TrainingProgress::Sample {
                                step,
                                index: i as u32 + 1,
                                total,
                                prompt: prompt.clone(),
                                image,
                            }),
                            // F-117: a cancelled preview exits the preview loop (outer cancel check
                            // unwinds the run); other failures skip a single preview.
                            Err(mlx_gen::Error::Canceled) => break,
                            Err(e) => eprintln!(
                                "[sc-5637] {MODEL_ID} preview sample failed at step {step} \
                                 (prompt {}): {e} — skipping this preview, training continues",
                                i + 1
                            ),
                        }
                    }
                }
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the LoRA factors are still freshly initialized with `B = 0`, a mathematically
        // no-op adapter. Surface the cancellation as the typed `Error::Canceled` (sc-4895, bridged
        // 1:1 to `gen_core::Error::Canceled`) rather than writing a valid-looking `.safetensors` and
        // returning `Ok` — downstream tooling would otherwise ship an identity LoRA as a trained
        // artifact (F-040).
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
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

/// One real optimizer update over the (already window-averaged) adapter gradients — the shared
/// [`adapter_optimizer_update`]: clip to unit global norm, epic 2123 gradient noise (sc-24827),
/// step the optimizer, materialize the factors, then weight noising (sc-24826), both seeded from
/// `(cfg.seed, update_idx)`. Only ever called once per *update* (never per gradient-accumulation
/// micro-step), and only `params` (adapter factors) is touched — the frozen base weights live in
/// the transformer and are not reachable from here. With both techniques off this is exactly the
/// pre-epic-2123 update (clip → step → eval).
fn optimizer_update(
    opt: &mut TrainOptimizer,
    params: &mut LoraParams,
    avg_grads: &LoraParams,
    cfg: &TrainingConfig,
    update_idx: u32,
) -> Result<()> {
    adapter_optimizer_update(opt, params, avg_grads, cfg, update_idx, cfg.seed)
}

/// The per-step loss breakdown [`compute_loss_grads`] returns.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StepLosses {
    /// The differentiated step loss.
    total: f32,
    /// The diffusion (velocity-regression) term, `None` on an aux-only step (it contributed zero).
    diffusion: Option<f32>,
    /// The weighted aux-loss term, `None` when no aux loss contributed this step.
    aux: Option<f32>,
}

/// One aux-loss step's view of the trainer's [`PerceptualPath`].
struct AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: &'a StepPlan,
    /// Dataset index of this step's image (selects its cached reference).
    image: usize,
}

/// Build the epic-2123 perceptual path from the config: `None` when no aux loss is enabled (the
/// default — nothing is loaded and every step is the plain diffusion step). The floor
/// ([`gen_core::train::validate_training_techniques`]) has already required both directories.
fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(cfg, &aux_loss_context())
}

/// Z-Image's latent family for the shared aux-loss builder (epic 2123 E8): the FLUX.1 16-channel
/// VAE latent, decoded by TAEF1.
fn aux_loss_context() -> mlx_gen_perceptual::AuxLossContext<'static> {
    mlx_gen_perceptual::AuxLossContext {
        label: "z_image_turbo trainer",
        decoder: taef1_decoder(),
        latent_lpips: Some(gen_core::train::LatentLpipsFamily::Flux),
    }
}

fn taef1_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Tiny {
        name: "TAEF1",
        config: TinyDecoderConfig::taef1().into(),
    }
}

/// One `train_impl` cache entry (item-major, `cache[item * n_buckets + bucket]`, sc-2127): the
/// clean latent, the caption embeds, and — subject-masked loss on (sc-24828) — that bucket's latent
/// loss-weight map, the same shape as the latent.
type CacheEntry = (Array, Array, Option<Array>);

/// Append one item's cache entries, one per bucket edge (item-major, sc-2127): `encode(edge)` is
/// the bucket's clean latent, and — sc-24828 — its loss-weight map is the item's already-loaded
/// mask cropped with the center square `center_crop_square` cut, area-averaged onto THIS bucket's
/// latent grid and broadcast to that latent's shape (`None` when masked loss is off).
fn cache_item_buckets(
    edges: &[u32],
    mask: Option<&PreparedSubjectMask>,
    cap: &Array,
    mut encode: impl FnMut(u32) -> Result<Array>,
    cache: &mut Vec<CacheEntry>,
) -> Result<()> {
    for &edge in edges {
        let x0 = encode(edge)?;
        let mask_weight = prepared_subject_mask_weight(
            "z_image_turbo trainer",
            mask,
            CropBox::center_square,
            x0.shape(),
        )?;
        eval([&x0])?;
        cache.push((x0, cap.clone(), mask_weight));
    }
    Ok(())
}

/// The loop's [`AuxDriver`] (epic 2123 E8): every cached entry's perceptual reference computed once
/// (its clean `[C, 1, h, w]` latent, unpacked to the decoder's NCHW layout), the alternation over
/// the schedule's epochs with `accum` micro-steps per update. Hands the path the job's subject
/// masks first (sc-24832), so a mask-reading loss gets each entry's item mask on its decoded grid.
fn aux_driver(
    mut path: PerceptualPath,
    cache: &[CacheEntry],
    masks: Option<mlx_gen::train::subject_mask::PerceptualSubjectMasks>,
    schedule: &BucketSchedule,
    accum: u32,
    cancel: &mlx_gen::gen_core::runtime::CancelFlag,
) -> Result<AuxDriver> {
    path.attach_subject_masks(masks);
    AuxDriver::prepare(
        path,
        cache.len(),
        |i| crate::pipeline::unpack_latents(&cache[i].0),
        schedule,
        accum,
        cancel,
    )
}

/// Extra training memory (GB) the enabled perceptual losses add at the bucketed `edge` — the
/// TAEF1 decoder + the selected Depth-Anything-V2 (resident f32 weights and one differentiable
/// forward/backward each) plus the cached per-image depth references (epic 2123 E7). `0` when
/// depth anchoring is off.
fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, images: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &taef1_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, images),
    )
}

/// One training micro-step on the 1-based `step`: pick the step's cached item, sample its σ and
/// noise (seeded, exactly as before epic 2123), plan the step's loss terms through the perceptual
/// path (when one is configured: the alternation key comes from the step's optimizer window, and
/// an aux-only step trains at σ remapped into the loss window), and run [`compute_loss_grads`].
/// With no perceptual path every step is the plain diffusion step, bit-identical to the
/// pre-epic-2123 loop.
#[allow(clippy::too_many_arguments)]
fn run_train_step(
    transformer: &mut ZImageTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    cfg: &TrainingConfig,
    cache: &[CacheEntry],
    schedule: &BucketSchedule,
    perceptual: Option<&mut AuxDriver>,
    step: u32,
    mae: bool,
    checkpoint_main: Option<&[Vec<String>]>,
    compute_dtype: Dtype,
) -> Result<(StepLosses, LoraParams)> {
    // sc-2127: the step's (item, bucket) from the bucket schedule. The perceptual references are
    // keyed per cache entry (item, bucket) — each bucket's clean latent decodes to its own size —
    // while the alternation is keyed on the step's window.
    let sample = step_sample(perceptual, step, schedule);
    let entry = sample.entry;
    let (x0, cap, mask_weight) = &cache[entry];
    let mut sigma = sample_sigma(
        &cfg.timestep_type,
        &cfg.timestep_bias,
        cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
    )?;
    let noise = random::normal::<f32>(
        x0.shape(),
        None,
        None,
        Some(&random::key(
            cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
        )?),
    )?;
    let planned = sample.plan(sigma)?;
    if let Some(p) = &planned {
        sigma = p.plan.noise_level;
    }
    let aux = planned.as_ref().map(|p| AuxStep {
        path: p.path,
        plan: &p.plan,
        image: p.entry,
    });
    compute_loss_grads(
        transformer,
        params,
        adapter,
        cfg.alpha,
        cfg.rank as f32,
        x0,
        cap,
        sigma,
        &noise,
        mae,
        // sc-24828: the weight cached in the SAME (item, bucket) slot as `x0`.
        mask_weight.as_ref(),
        checkpoint_main,
        compute_dtype,
        aux,
    )
}

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the
/// unified token count `s` — an empirical fit to peaks measured on the 128 GB target.
///
/// The structure follows the sc-4874 root-cause decomposition: `weights + linear·s + quad·s²`,
/// where the constant is the resident base, the linear term is the per-token retained hidden-state
/// activations across the 30 blocks, and the quadratic term is seq² attention. Since sc-4886 the
/// training dense path always runs attention-segment checkpointing, which demotes the seq² term
/// from "one retained `[30-heads,s,s]` probability matrix per block" (3.92e-6·s² ≈ 66 GB at 1024,
/// the original sc-4874 explosion) to a single layer's backward transient (~1/30 of that).
/// sc-4887 bf16 halves the weights + activation terms.
///
/// Measured (`first_step_attn_ckpt_sweep`, 128 GB Mac17,6, rank 16 / 136 targets / batch 1) AFTER the
/// Qwen text encoder is freed post-caching (sc-4952 — the resident base now excludes the ~6.6 GB
/// encoder): f32 edge 768/1024 → 49.6/71.9 GB; bf16 → 25.6/36.7 GB. The activation/seq² shape is
/// unchanged from the pre-sc-4952 fit (same forward); only the resident base shifts down (f32 −7.3,
/// bf16 −6.6 GB). bf16 768 (~25.6 GB) now fits a 32 GB Mac at the default (attention-segment)
/// checkpointing; 1024 (~36.7 GB) needs `gradient_checkpointing` (block-ckpt ≈ 23 GB). Assumes
/// micro-batch 1; refit if the encoder lifecycle or activation shape changes.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        12.5 + 0.00522 * s + 1.55e-7 * s * s
    } else {
        23.46 + 0.01045 * s + 3.09e-7 * s * s
    }
}

/// The checkpointed DiT baseline the auxiliary-model guard stacks on: the resident-base term of
/// [`projected_dense_peak_gb`] (block checkpointing removes most of the activation terms; the base
/// weights stay). A lower bound — measured block-checkpointed bf16 at 1024 is ~23 GB.
fn checkpointed_baseline_gb(bf16: bool) -> f64 {
    projected_dense_peak_gb(0.0, bf16)
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. `edge` is the bucketed training
/// edge; the unified token count is ≈ `(edge/16)²` (latent /8, patch 2) plus the small padded
/// caption block. The budget is MLX's own reported memory limit (≈ the device's recommended working
/// set), scaled by 0.85 to leave headroom for the worker/host — exceeding it is the regime where the
/// dense run was observed to die. Consulted when gradient checkpointing is OFF, and — whenever the
/// training-time auxiliary models add memory — when it is on too. `extra_gb` is those models'
/// footprint ([`perceptual_footprint_gb`], epic 2123 E7), added on top of the DiT projection. With
/// `checkpointed`, the DiT projection is [`checkpointed_baseline_gb`] (no fitted checkpointed curve
/// exists, so the resident base is the lower bound the auxiliary models stack on).
fn preflight_memory_guard(
    cfg: &TrainingConfig,
    edge: u32,
    bf16: bool,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    preflight_memory_guard_with_budget(cfg, edge, bf16, extra_gb, checkpointed, get_memory_limit())
}

/// The edge the memory pre-flight (and the preview render) sizes for: the largest bucket edge
/// (epic 2123 E7, sc-2127) — a bucketed run's peak is its largest latent.
fn preflight_edge(edges: &[u32]) -> u32 {
    edges.iter().copied().max().unwrap_or(0)
}

/// [`preflight_memory_guard`] against an explicit memory budget (`budget_bytes`, the live MLX limit
/// in production) — so the guard's arithmetic is testable on any host. A checkpointed refusal goes
/// through the shared [`mlx_gen_perceptual::check_aux_memory`], naming `cfg`'s enabled aux losses.
fn preflight_memory_guard_with_budget(
    cfg: &TrainingConfig,
    edge: u32,
    bf16: bool,
    extra_gb: f64,
    checkpointed: bool,
    budget_bytes: usize,
) -> Result<()> {
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    // The padded caption block can reach the model's max prompt length (~512 tokens), not just the
    // 32-token padding granularity; under-counting it lets a near-threshold long-prompt dense run slip
    // past this guard into a SIGKILL instead of the catchable error. Use a conservative cap (F-057).
    let s = tokens_per_side * tokens_per_side + (512.0 + 32.0);
    let projected = if checkpointed {
        checkpointed_baseline_gb(bf16)
    } else {
        projected_dense_peak_gb(s, bf16)
    } + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if checkpointed {
        return mlx_gen_perceptual::check_aux_memory(
            "z_image_turbo trainer",
            cfg,
            &format!("a checkpointed training step at resolution {edge}"),
            extra_gb,
            projected,
            safe,
            &format!("{budget_gb:.0} GB MLX limit × 0.85"),
        );
    }
    if projected > safe {
        return Err(mlx_gen_perceptual::name_aux_losses(
            format!(
                "z_image_turbo trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
                 (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
                 safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS would hard-kill the \
                 worker (SIGKILL) at the first step with no recoverable error (sc-4874). Enable Gradient \
                 Checkpointing (recomputes block activations in the backward) or reduce the training resolution."
            )
            .into(),
            cfg,
            extra_gb,
        ));
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted paths by matching them against every adapter-routable module on the DiT — the same
/// suffix-match PEFT's `LoraConfig(target_modules=…)` does. This trains the attention projections in
/// the main `layers` AND the noise/context refiner stacks (matching the torch trainer), and handles
/// non-attention suffixes (FFN `w1`/`w2`/`w3`, `adaLN_modulation.0`) when configured — not just a
/// hardcoded `layers.{i}.attention.{suffix}`.
fn resolve_target_paths(transformer: &ZImageTransformer, cfg: &TrainingConfig) -> Vec<String> {
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

/// Sample a normalised flow-match timestep (interpolation coefficient) `σ ∈ [1e-3, 1-1e-3]` — a
/// faithful port of the SceneWorks `sample_training_timestep`: `sigmoid(randn)` by default,
/// `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√σ`,
/// `low` → `σ²`. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t = match ttype.as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            // wrapping_add (Fibonacci-hash mixing) rather than an XOR-sibling seed, whose two draws
            // can correlate depending on `key()`'s splat (F-058).
            let k2 = random::key(seed.wrapping_add(0x9E37_79B9))?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    let bias = timestep_bias
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_");
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), run the
/// DiT, regress the (already-negated) `forward()` output toward the velocity `noise - x0`, return
/// the step's loss breakdown and the grads of its total.
/// `aux` (epic 2123 E8) carries the step's [`StepPlan`]: on an aux-only step the diffusion term is
/// not computed (it contributes zero) and the loss is the weighted perceptual term on the model's
/// x0 estimate `x_t − σ·v`; with `aux = None` (or a diffusion-only plan with no aux loss) the
/// traced graph is exactly the pre-epic-2123 one.
/// `checkpoint_main`, when `Some`, lists per-main-block LOCAL LoRA target paths and switches the
/// forward to the gradient-checkpointed path (sc-4874) — each main block recomputes its activations
/// in the backward instead of retaining them. `None` runs the dense (activation-retaining) forward.
/// `dtype` is the training compute dtype (sc-4887): for bf16 the latent / caption / RoPE inputs are
/// cast at entry (the weights were cast once in `train_impl`) and the LoRA factors are cast inside
/// the traced install, so the whole DiT graph runs bf16; the noising math, loss, and grads stay f32.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut ZImageTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cap: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_main: Option<&[Vec<String>]>,
    dtype: Dtype,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (x_t_f32, target, timestep) = build_batch(x0, noise, sigma)?;
    let mask_weight = mask_weight.cloned();
    let x_t = x_t_f32.as_dtype(dtype)?; // no-op in f32 mode
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    let capf = cap.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // F-149 invariant: NEVER check the cancel flag inside this traced grad closure. It returns
        // `MlxResult`, so any early-out here would be stringified through `Exception::custom` and lose
        // the typed `Error::Canceled` variant (the errors below are already forced through that lossy
        // bridge). Cancellation is the caller's job at the step boundary (before/after this runs), where
        // it stays a typed `Error::Canceled` — the whole grad graph is one atomic, uncancellable unit.
        //
        // Install ALL targets: refiners/embedders train through this on the (non-checkpointed)
        // pre-main forward; the main-block adapters installed here are simply replaced inside each
        // checkpoint segment by the explicit-input factors, so they cost nothing on the ckpt path.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let sh = x_t.shape();
        let prep = transformer
            .prepare((sh[0], sh[1], sh[2], sh[3]), &capf)
            .and_then(|prep| {
                if dtype == Dtype::Float32 {
                    Ok(prep)
                } else {
                    prep.cast_floats(dtype)
                }
            })
            .map_err(|e| Exception::custom(e.to_string()))?;
        let v = match checkpoint_main {
            Some(locals) => transformer
                .forward_with_main_checkpointed(&prep, &x_t, timestep, &p, locals, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward_with(&prep, &x_t, timestep)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diffusion = if diffusion_on {
            let diff = subtract(&v, &target)?;
            // MSE / MAE, subject-mask weighted when on (sc-24828) — reduces to a 0-d scalar (grad
            // requires a scalar cotangent).
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // x0 estimate in f32 from the (already-negated) velocity: x0 = x_t − σ·v.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .and_then(|x| crate::pipeline::unpack_latents(&x))
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0_hat)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        // Only the first output is differentiated; the other two are reported terms.
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: val[0].item::<f32>(),
        diffusion: diffusion_on.then(|| val[1].item::<f32>()),
        aux: aux_on.then(|| val[2].item::<f32>()),
    };
    Ok((losses, grads))
}

// ===========================================================================================
// sc-4874 — first-step signal-9 repro/instrumentation (weight-gated, run as its own process).
//
// The production run dies with SIGKILL the instant the first real training step begins, at the
// default resolution 1024 (the e2e `trainer_e2e.rs` runs at 64 and never exercises this regime).
// This harness drives the exact inner step (`compute_loss_grads` + the step-1 grad `eval` the real
// loop forces at training.rs' optimizer step) directly, sweeping resolution with MLX peak-memory
// probes around it, to pinpoint the working set at the death point and test whether a memory
// ceiling converts the silent kill into a catchable error.
//
//   cargo test -p mlx-gen-z-image --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::media::Image;
    use mlx_gen::train::lora::build_lora_targets;
    use mlx_rs::memory::{
        clear_cache, get_active_memory, get_peak_memory, reset_peak_memory, set_memory_limit,
    };
    use std::path::PathBuf;

    /// The Z-Image-Turbo dense source root (a `SceneWorks/z-image-turbo-mlx` bf16 tier dir — the
    /// re-host the MLX product path trains from, sc-18213) from the required
    /// `MLX_GEN_ZIMAGE_SNAPSHOT` env var. sc-13668: there
    /// is no implicit default — the source snapshot path must be passed in explicitly.
    fn snapshot() -> Option<PathBuf> {
        std::env::var("MLX_GEN_ZIMAGE_SNAPSHOT")
            .ok()
            .map(PathBuf::from)
    }

    #[test]
    fn source_root_requires_explicit_env_no_default() {
        let key = "MLX_GEN_ZIMAGE_SNAPSHOT";
        let saved = std::env::var(key).ok();
        std::env::remove_var(key);
        assert!(
            snapshot().is_none(),
            "the source snapshot root must come from {key}: sc-13668 removed the implicit default"
        );
        std::env::set_var(key, "/sentinel/zimage");
        assert_eq!(snapshot(), Some(PathBuf::from("/sentinel/zimage")));
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// A solid-colour `edge`×`edge` RGB source image (the latent magnitude is irrelevant; the graph
    /// size — driven by resolution — is the variable under test).
    fn swatch(edge: u32) -> Image {
        let mut img = image::RgbImage::new(edge, edge);
        for px in img.pixels_mut() {
            *px = image::Rgb([180u8, 60, 90]);
        }
        Image {
            width: edge,
            height: edge,
            pixels: img.into_raw(),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Run a single first training step at `edge` and report peak GPU memory across the
    /// forward+backward. Forces the backward (grad eval) — the real step-1 kill point.
    /// `dtype` is the compute dtype handed to `compute_loss_grads` (the caller is responsible for
    /// having `cast_weights` the transformer to match); the SDPA-checkpoint flag is likewise set by
    /// the caller on `trainer.transformer` (sc-4886 arms A/B against the retained backward).
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut ZImageTurboTrainer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        cap: &Array,
        edge: u32,
        checkpoint_main: Option<&[Vec<String>]>,
        dtype: Dtype,
        tag: &str,
    ) -> Result<(f32, f64, [i32; 4])> {
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge)?;
        eval([&x0])?;
        let shape = {
            let s = x0.shape();
            [s[0], s[1], s[2], s[3]]
        };
        let noise = random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1)?))?;
        eval([&noise])?;

        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            &mut trainer.transformer,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            cap,
            0.5,
            &noise,
            false,
            None,
            checkpoint_main,
            dtype,
            None,
        )?;
        let loss = loss.total;
        // `compute_loss_grads` only forces the loss (forward). The real trainer forces the backward
        // at the step-1 optimizer `eval`; do the same here so the peak reflects the true working set.
        eval(grads.values())?;
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "  [edge {edge:>4} {tag}] latent {shape:?}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak), shape))
    }

    /// Per-main-block LOCAL LoRA target paths (mirrors `train_impl`), for driving the checkpointed
    /// path from the harness.
    fn main_block_local_targets(trainer: &ZImageTurboTrainer) -> Vec<Vec<String>> {
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.transformer, &cfg);
        let n_layers = trainer.transformer.cfg.n_layers;
        let mut out: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("layers.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        out[i].push(local.to_string());
                    }
                }
            }
        }
        out
    }

    fn build_trainer_and_adapter() -> (ZImageTurboTrainer, TrainAdapter, LoraParams, Array) {
        let root =
            snapshot().expect("set MLX_GEN_ZIMAGE_SNAPSHOT to the Z-Image-Turbo snapshot root");
        let mut trainer = ZImageTurboTrainer {
            descriptor: trainer_descriptor(),
            tokenizer: crate::loader::load_tokenizer(&root).unwrap(),
            text_encoder: Some(crate::loader::load_text_encoder(&root).unwrap()),
            vae: crate::loader::load_vae(&root).unwrap(),
            transformer: crate::loader::load_transformer(&root).unwrap(),
        };
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.transformer, &cfg);
        let (targets, params) =
            build_lora_targets(&mut trainer.transformer, &target_paths, 16, 7).unwrap();
        let cap = crate::pipeline::encode_prompt(
            &trainer.tokenizer,
            trainer.text_encoder.as_ref().unwrap(),
            "a solid colour swatch",
            "sc-4874 repro",
            None,
        )
        .unwrap();
        eval([&cap]).unwrap();
        // Drop the encoder exactly as `train_impl` does after caching, so the measured peaks reflect
        // the post-free training working set (sc-4952).
        trainer.text_encoder = None;
        mlx_rs::memory::clear_cache();
        eprintln!(
            "[sc-4874] loaded trainer (encoder freed); {} LoRA targets; cap {:?}",
            targets.len(),
            cap.shape()
        );
        (trainer, TrainAdapter::Lora { targets }, params, cap)
    }

    /// Attribute the first-step peak to the FORWARD eval vs the BACKWARD eval (sc-4874 root-cause:
    /// is the "blip" the forward materializing, or the autograd backward retaining/recomputing?).
    /// `compute_loss_grads` forces the forward (via `loss.item()`) and returns lazy grads; we then
    /// force the backward. Run at SAFE resolutions (512/768) so it characterizes the split without
    /// risking the OOM kill, and extrapolate.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_memory_attribution() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        for edge in [512u32, 768] {
            let img = center_crop_square(&swatch(edge));
            let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
            let noise =
                random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap()))
                    .unwrap();
            eval([&x0, &noise]).unwrap();

            clear_cache();
            reset_peak_memory();
            let build_active = get_active_memory();
            let (_loss, grads) = compute_loss_grads(
                &mut trainer.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                None,
                Dtype::Float32,
                None,
            )
            .unwrap();
            // `loss.item()` inside `compute_loss_grads` already forced the FORWARD graph.
            let fwd_peak = get_peak_memory();
            eval(grads.values()).unwrap(); // force the BACKWARD
            let full_peak = get_peak_memory();
            eprintln!(
                "[sc-4874] edge {edge}: lazy-build active {:.2} GB | forward-eval peak {:.2} GB | +backward peak {:.2} GB (backward added {:.2} GB)",
                gb(build_active),
                gb(fwd_peak),
                gb(full_peak),
                gb(full_peak.saturating_sub(fwd_peak)),
            );
        }
    }

    /// Sweep resolution from tiny → production, printing the peak-memory curve. If the process dies
    /// (SIGKILL) at some edge, the prints from the survived edges are already flushed (eprintln +
    /// per-step completion), so the curve shows exactly where it falls over.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process (may SIGKILL)"]
    fn first_step_memory_sweep() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        eprintln!("[sc-4874] sweeping first-step peak memory by resolution:");
        for edge in [64u32, 256, 512, 768, 1024] {
            eprintln!("[sc-4874] === edge {edge} ===");
            match one_step(
                &mut trainer,
                &adapter,
                &params,
                &cap,
                edge,
                None,
                Dtype::Float32,
                "dense",
            ) {
                Ok((_, peak, shape)) => {
                    eprintln!("[sc-4874] edge {edge} SURVIVED  latent {shape:?}  peak {peak:.2} GB")
                }
                Err(e) => eprintln!("[sc-4874] edge {edge} returned CATCHABLE error: {e}"),
            }
        }
        eprintln!("[sc-4874] sweep complete (reached the end without SIGKILL)");
    }

    /// Production resolution (1024) with a low MLX memory ceiling: tests whether forcing MLX to
    /// stay under a limit converts the silent SIGKILL into a catchable Rust error (story step 2).
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_1024_with_memory_limit() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let limit_gb = std::env::var("SC4874_LIMIT_GB")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(24.0);
        let prev = set_memory_limit((limit_gb * 1024.0 * 1024.0 * 1024.0) as usize);
        eprintln!(
            "[sc-4874] set memory limit to {limit_gb:.1} GB (prev {:.2} GB); running edge 1024…",
            gb(prev)
        );
        match one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            None,
            Dtype::Float32,
            "dense",
        ) {
            Ok((loss, peak, shape)) => eprintln!(
                "[sc-4874] edge 1024 SURVIVED under {limit_gb:.1} GB limit  latent {shape:?}  loss {loss:.5}  peak {peak:.2} GB"
            ),
            Err(e) => eprintln!("[sc-4874] edge 1024 returned CATCHABLE error under limit: {e}"),
        }
    }

    /// The fix: at production resolution 1024, gradient checkpointing must drop the first-step peak
    /// well under the dense path's ~135 GB (which exceeds the 128 GB unified memory). Runs the dense
    /// step first (baseline), then the checkpointed step, and asserts a substantial reduction.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_1024_checkpointed_vs_dense() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let locals = main_block_local_targets(&trainer);
        let n_main: usize = locals.iter().map(|v| v.len()).sum();
        eprintln!("[sc-4874] checkpointing {n_main} LoRA targets across the main stack");

        let (_, dense_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            None,
            Dtype::Float32,
            "dense",
        )
        .expect("dense step");
        let (_, ckpt_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            Some(&locals),
            Dtype::Float32,
            "blk-ckpt",
        )
        .expect("checkpointed step");
        eprintln!(
            "[sc-4874] edge 1024  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_peak / dense_peak)
        );
        assert!(
            ckpt_peak < dense_peak,
            "checkpointing must reduce the first-step peak: dense {dense_peak:.2} GB vs ckpt {ckpt_peak:.2} GB"
        );
        // The whole point is fitting under 128 GB with production headroom — expect a large drop.
        assert!(
            ckpt_peak < 128.0,
            "checkpointed peak must fit unified memory: {ckpt_peak:.2} GB"
        );
    }

    /// Gradient checkpointing must not change the math: the checkpointed forward+grads must match the
    /// dense path within fp tolerance (it reuses the same install + block forward, recompute-only).
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let locals = main_block_local_targets(&trainer);
        let edge = 256u32; // small enough that the dense path is cheap; math is resolution-agnostic
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        eval([&x0, &noise]).unwrap();

        let grads_of = |t: &mut ZImageTurboTrainer, ck: Option<&[Vec<String>]>| -> LoraParams {
            let (_l, g) = compute_loss_grads(
                &mut t.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                ck,
                Dtype::Float32,
                None,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_dense = grads_of(&mut trainer, None);
        let g_ckpt = grads_of(&mut trainer, Some(&locals));

        let mut max_rel = 0f32;
        for (k, gd) in &g_dense {
            let gc = g_ckpt.get(k).expect("same keys");
            let num = gd.subtract(gc).unwrap().abs().unwrap().max(None).unwrap();
            let den = gd.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        eprintln!("[sc-4874] checkpointed-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense within tolerance: max rel {max_rel:.2e}"
        );
    }

    /// Max relative grad diff between two param maps (shared by the sc-4886/4887 parity tests).
    fn max_rel_diff(ga: &LoraParams, gb_: &LoraParams) -> f32 {
        let mut max_rel = 0f32;
        for (k, a) in ga {
            let b = gb_.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        max_rel
    }

    /// sc-4886 — the always-on attention-segment checkpointing must not change the math: grads with
    /// the SDPA checkpoint on must match the retained backward (flag off). Same decomposed
    /// attention, recomputed instead of retained → expect (near-)bit-identical.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        eval([&x0, &noise]).unwrap();

        let grads_of = |t: &mut ZImageTurboTrainer, on: bool| -> LoraParams {
            t.transformer.set_sdpa_checkpoint(on, on);
            let (_l, g) = compute_loss_grads(
                &mut t.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                None,
                Dtype::Float32,
                None,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_retained = grads_of(&mut trainer, false);
        let g_ckpt = grads_of(&mut trainer, true);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-4886] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4886/4887 — first-step peak sweep on the NEW training dense path (attention-segment
    /// checkpointing always on), f32 then bf16. These measured points are the basis of the
    /// `projected_dense_peak_gb` guard fit — refit the constants if this prints materially
    /// different numbers.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_attn_ckpt_sweep() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        trainer.transformer.set_sdpa_checkpoint(true, true);
        eprintln!("[sc-4886] attn-ckpt dense sweep, f32:");
        for edge in [512u32, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cap,
                edge,
                None,
                Dtype::Float32,
                "attn-ckpt f32",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4887] casting weights to bf16…");
        trainer
            .transformer
            .cast_weights(Dtype::Bfloat16)
            .expect("cast");
        clear_cache();
        eprintln!("[sc-4887] attn-ckpt dense sweep, bf16:");
        for edge in [512u32, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cap,
                edge,
                None,
                Dtype::Bfloat16,
                "attn-ckpt bf16",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4887] block-ckpt + bf16 at 1024:");
        let locals = main_block_local_targets(&trainer);
        trainer.transformer.set_sdpa_checkpoint(false, true);
        let _ = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            Some(&locals),
            Dtype::Bfloat16,
            "blk-ckpt bf16",
        )
        .map_err(|e| eprintln!("  blk-ckpt bf16 CATCHABLE error: {e}"));
    }

    /// sc-4887 — bf16 is mixed precision, NOT bit parity: assert the grads point the same way as
    /// the f32 path (per-param cosine) and the loss is finite. Runs f32 first (the weight cast is
    /// destructive), then casts the same trainer to bf16. Also asserts the bf16 working set is
    /// genuinely smaller — a silent f32 re-promotion anywhere in the forward would pass the cosine
    /// check while saving nothing, so the memory ratio IS the dtype assertion.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn bf16_grads_direction_and_memory_vs_f32() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        trainer.transformer.set_sdpa_checkpoint(true, true);

        // Memory A/B at 768 (big enough that activations dominate the peak).
        let (_, f32_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            768,
            None,
            Dtype::Float32,
            "attn-ckpt f32",
        )
        .expect("f32 step");

        // Grad reference at 256 in f32.
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        eval([&x0, &noise]).unwrap();
        let grads_of = |t: &mut ZImageTurboTrainer, dt: Dtype| -> (f32, LoraParams) {
            let (l, g) = compute_loss_grads(
                &mut t.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                None,
                dt,
                None,
            )
            .unwrap();
            eval(g.values()).unwrap();
            (l.total, g)
        };
        let (f32_loss, g_f32) = grads_of(&mut trainer, Dtype::Float32);

        trainer
            .transformer
            .cast_weights(Dtype::Bfloat16)
            .expect("cast");
        clear_cache();
        let (bf16_loss, g_bf16) = grads_of(&mut trainer, Dtype::Bfloat16);
        assert!(
            bf16_loss.is_finite(),
            "bf16 loss must be finite: {bf16_loss}"
        );
        eprintln!("[sc-4887] loss f32 {f32_loss:.5} vs bf16 {bf16_loss:.5}");

        // Cosine between bf16 and f32 grads (both arrive f32 through the astype VJP). Gate on the
        // GLOBAL cosine (the concatenated gradient — what the optimizer step actually follows) and
        // the norm-weighted view; per-param minima are dominated by tiny-norm params whose direction
        // bf16 rounding legitimately scrambles while contributing nothing to the update. Print the
        // worst offenders with their norms so a real systematic bug (large-norm divergence) is
        // distinguishable from precision noise on negligible grads.
        let mut per: Vec<(String, f32, f32, f32)> = Vec::new(); // (key, cos, na, nb)
        let (mut gdot, mut gna2, mut gnb2) = (0f64, 0f64, 0f64);
        for (k, a) in &g_f32 {
            let b = g_bf16.get(k).expect("same keys");
            let dot = a.multiply(b).unwrap().sum(None).unwrap().item::<f32>();
            let na2 = a.square().unwrap().sum(None).unwrap().item::<f32>();
            let nb2 = b.square().unwrap().sum(None).unwrap().item::<f32>();
            gdot += dot as f64;
            gna2 += na2 as f64;
            gnb2 += nb2 as f64;
            let (na, nb) = (na2.sqrt(), nb2.sqrt());
            if na > 1e-12 && nb > 1e-12 {
                per.push((k.to_string(), dot / (na * nb), na, nb));
            }
        }
        let global_cos = (gdot / (gna2.sqrt() * gnb2.sqrt())) as f32;
        per.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let max_norm = per.iter().map(|p| p.2).fold(0f32, f32::max);
        eprintln!(
            "[sc-4887] bf16-vs-f32 grads: global cosine {global_cos:.5}; worst per-param (cos, |f32|, |bf16|, |f32|/max):"
        );
        for (k, c, na, nb) in per.iter().take(5) {
            eprintln!(
                "    {k}: cos {c:.4}  |g| {na:.3e} vs {nb:.3e}  rel-norm {:.2e}",
                na / max_norm
            );
        }
        // Any LARGE-norm param (≥1% of the biggest grad) must also agree directionally — that is
        // the systematic-bug detector; small-norm direction noise is expected mixed-precision.
        let min_large = per
            .iter()
            .filter(|p| p.2 >= 0.01 * max_norm)
            .map(|p| p.1)
            .fold(1f32, f32::min);
        eprintln!("[sc-4887] min cosine among params with |g| >= 1% of max: {min_large:.4}");
        assert!(
            global_cos > 0.995,
            "bf16 global grad must point the same way as f32: {global_cos:.5}"
        );
        // Measured on real weights: 0.966 (worst = main-stack to_k.lora_b — k-grads flow through
        // the bf16 softmax backward, the most precision-sensitive chain; no norm shrink). The
        // structural failure this gate exists for looked very different: a CLUSTER at cos 0.43-0.81
        // with systematically smaller bf16 norms (the caption-entry sensitivity, fixed by keeping
        // it f32 — see `train_unify_dtype`).
        assert!(
            min_large > 0.95,
            "a large-norm param's bf16 grad diverged from f32 (systematic bug, not precision): {min_large:.4}"
        );

        let (_, bf16_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            768,
            None,
            Dtype::Bfloat16,
            "attn-ckpt bf16",
        )
        .expect("bf16 step");
        eprintln!(
            "[sc-4887] 768 peak f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB ({:.0}%)",
            100.0 * bf16_peak / f32_peak
        );
        assert!(
            bf16_peak < 0.70 * f32_peak,
            "bf16 must materially shrink the working set (silent f32 re-promotion?): \
             f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB"
        );
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::projected_dense_peak_gb;

    // The empirical fit must reproduce the measured first-step peaks within a few GB and stay
    // monotonic — it is the basis of the pre-flight OOM guard, so a regression here silently
    // mis-sizes the guard. s ≈ (edge/16)²: edge 512→1024, 768→2304, 1024→4096 tokens.
    // The measured points come from `first_step_attn_ckpt_sweep` (the training dense path always
    // runs attention-segment checkpointing since sc-4886; the original retained-attention 135 GB
    // curve from sc-4874 no longer describes any reachable training path).
    #[test]
    fn projection_matches_measured_curve() {
        // Measured AFTER the encoder is freed post-caching (sc-4952), `first_step_attn_ckpt_sweep`,
        // 128 GB Mac17,6. s = (edge/16)² + 32 → edge 768/1024. (The 512 samples are load/cast-transient
        // polluted — the encoder-free leaves the DiT lazy until the first eval — so the fit is anchored
        // on 768/1024, with the activation/seq² shape carried over from the pre-sc-4952 fit.)
        for (s, measured) in [(2336.0, 49.6), (4128.0, 71.9)] {
            let p = projected_dense_peak_gb(s, false);
            assert!(
                (p - measured).abs() < 3.0,
                "f32 projection at s={s} = {p:.1} GB, expected ≈{measured} GB"
            );
        }
        for (s, measured) in [(2336.0, 25.6), (4128.0, 36.7)] {
            let p = projected_dense_peak_gb(s, true);
            assert!(
                (p - measured).abs() < 3.0,
                "bf16 projection at s={s} = {p:.1} GB, expected ≈{measured} GB"
            );
        }
        for bf16 in [false, true] {
            // Monotonic increasing in token count; bf16 strictly below f32.
            assert!(projected_dense_peak_gb(1056.0, bf16) < projected_dense_peak_gb(2336.0, bf16));
            assert!(projected_dense_peak_gb(2336.0, bf16) < projected_dense_peak_gb(4128.0, bf16));
            assert!(projected_dense_peak_gb(2336.0, true) < projected_dense_peak_gb(2336.0, false));
        }
        // Still fits the 128 GB target at 1024 in both dtypes (budget ≈ 103 GB); the encoder-free just
        // lowers the resident base, extending the headroom (and bringing bf16 768 under a 32 GB box).
        assert!(projected_dense_peak_gb(4128.0, false) < 103.0);
        assert!(projected_dense_peak_gb(4128.0, true) < 103.0);
    }
}

#[cfg(test)]
mod validate_request_tests {
    use super::validate_request;
    use mlx_gen::{TrainingConfig, TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn request(items: usize, steps: u32, rank: u32) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "a cat".into(),
                    control_image_path: None,
                    model_options: Default::default(),
                    reference_image_paths: Vec::new(),
                    subject_mask_path: None,
                })
                .collect(),
            config: TrainingConfig {
                steps,
                rank,
                ..Default::default()
            },
            output_dir: PathBuf::from("/tmp/z-image-trainer-test"),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        }
    }

    #[test]
    fn rejects_zero_steps() {
        // F-040: a 0-step run must fail validation — otherwise the train loop runs no iterations and
        // falls straight through to the save, writing a no-op `B = 0` identity adapter.
        let err = validate_request(&request(1, 0, 16)).unwrap_err();
        assert!(format!("{err}").contains("steps must be > 0"), "got: {err}");
    }

    #[test]
    fn accepts_valid_request_and_keeps_existing_guards() {
        assert!(validate_request(&request(1, 100, 16)).is_ok());
        assert!(validate_request(&request(0, 100, 16)).is_err()); // empty dataset
        assert!(validate_request(&request(1, 100, 0)).is_err()); // zero rank
    }

    #[test]
    fn rejects_unrecognized_schedule_and_loss_strings() {
        // F-041: a typo in these strings used to silently fall back to a default sampler/loss.
        let with = |f: fn(&mut TrainingConfig)| {
            let mut r = request(1, 100, 16);
            f(&mut r.config);
            validate_request(&r)
        };
        assert!(with(|c| c.timestep_type = "sgmoid".into()).is_err());
        assert!(with(|c| c.timestep_bias = "hihg_noise".into()).is_err());
        assert!(with(|c| c.loss_type = "huber".into()).is_err());

        // The defaults and documented spellings still pass, case- and separator-insensitively.
        assert!(with(|c| c.timestep_type = "Linear".into()).is_ok());
        assert!(with(|c| c.timestep_bias = "High-Noise".into()).is_ok());
        assert!(with(|c| c.loss_type = "L1".into()).is_ok());
        // A default request (sigmoid / balanced / mse) is accepted.
        assert!(validate_request(&request(1, 100, 16)).is_ok());
    }
}

/// sc-24826 (epic 2123 weight noising) — the Z-Image optimizer-update seam on a tiny synthetic
/// adapter host: real `build_lora_targets`/`build_lokr_targets` factors, a real `TrainOptimizer`, and
/// the same [`optimizer_update`] `train_impl` calls. Seconds, < 1 MB.
#[cfg(test)]
mod weight_noise_update_tests {
    use super::*;
    use mlx_gen::adapters::AdaptableLinear;
    use mlx_rs::optimizers::clip_grad_norm;

    struct OneLin(AdaptableLinear);
    impl AdaptableHost for OneLin {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            (path == ["to_q"]).then_some(&mut self.0)
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["to_q".to_string()]
        }
    }

    fn host() -> OneLin {
        let w =
            random::normal::<f32>(&[64, 64], None, None, Some(&random::key(3).unwrap())).unwrap();
        OneLin(AdaptableLinear::dense(w, None))
    }

    fn base_weight(h: &OneLin) -> Vec<f32> {
        let (w, _) = h.0.dense_weight().expect("dense host");
        eval([w]).unwrap();
        w.as_slice::<f32>().to_vec()
    }

    fn vals(a: &Array) -> Vec<f32> {
        eval([a]).unwrap();
        a.as_slice::<f32>().to_vec()
    }

    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    }

    /// Fixed synthetic gradients for every factor (sorted keys ⇒ deterministic).
    fn grads(params: &LoraParams, update: u32) -> LoraParams {
        let mut keys: Vec<_> = params.keys().cloned().collect();
        keys.sort();
        keys.into_iter()
            .enumerate()
            .map(|(i, k)| {
                let key = random::key(1000 + 31 * update as u64 + i as u64).unwrap();
                let g = random::normal::<f32>(params[&k].shape(), None, None, Some(&key)).unwrap();
                (k, g)
            })
            .collect()
    }

    fn setup(kind: NetworkType) -> (OneLin, TrainAdapter, LoraParams) {
        let mut h = host();
        let paths = vec!["to_q".to_string()];
        let (adapter, params) = match kind {
            NetworkType::Lora => {
                let (t, p) = build_lora_targets(&mut h, &paths, 8, 7).unwrap();
                (TrainAdapter::Lora { targets: t }, p)
            }
            NetworkType::Lokr => {
                let (t, p) = build_lokr_targets(&mut h, &paths, 8, -1, 7).unwrap();
                (TrainAdapter::Lokr { targets: t }, p)
            }
        };
        (h, adapter, params)
    }

    fn run(kind: NetworkType, sigma: f32, updates: u32) -> (LoraParams, Vec<f32>, Vec<f32>) {
        let (mut h, adapter, mut params) = setup(kind);
        let base_before = base_weight(&h);
        let cfg = TrainingConfig {
            seed: 7,
            weight_noise_sigma: sigma,
            ..Default::default()
        };
        let mut opt = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        for u in 0..updates {
            let g = grads(&params, u);
            optimizer_update(&mut opt, &mut params, &g, &cfg, u).unwrap();
            // Install exactly as a train step does — the base must survive it untouched.
            adapter
                .install(&mut h, &params, 8.0, 8.0, LOKR_DTYPE)
                .unwrap();
        }
        (params, base_before, base_weight(&h))
    }

    /// AC1: a sigma-0.0125 update differs from the sigma-0 update by noise whose RMS is
    /// `sigma · rms(w)` per adapter tensor (LoRA A/B and every LoKr factor), and the base weight is
    /// bit-identical before and after training.
    #[test]
    fn sigma_noise_is_rms_relative_per_adapter_tensor_and_base_is_untouched() {
        let sigma = 0.0125f32;
        for kind in [NetworkType::Lora, NetworkType::Lokr] {
            let (clean, _, _) = run(kind, 0.0, 1);
            let (noisy, base_before, base_after) = run(kind, sigma, 1);
            assert_eq!(base_before, base_after, "{kind:?}: base weight changed");
            assert_eq!(clean.len(), noisy.len());
            for (k, c) in &clean {
                let c = vals(c);
                let n = vals(&noisy[k]);
                let delta: Vec<f32> = c.iter().zip(&n).map(|(a, b)| b - a).collect();
                let ratio = rms(&delta) / (sigma as f64 * rms(&c));
                assert!(
                    (0.8..1.2).contains(&ratio),
                    "{kind:?} {k}: delta rms / (sigma·rms(w)) = {ratio} (n={})",
                    c.len()
                );
            }
        }
    }

    /// Two SEPARATE Metal executions of the same seeded update graph agree only to a few ulps
    /// (Metal reductions are not bit-deterministic across runs), so they are compared within
    /// `1e-6 + 1e-5·max|want|`; the guarded mutations (noise in an "off" update, a drifting noise
    /// seed) move values by orders of magnitude more.
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let tol = 1e-6 + 1e-5 * scale;
        let diff = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff <= tol, "{what}: max |diff| {diff} > tolerance {tol}");
    }

    /// AC3 (E1): with sigma 0 the update is the pre-sc-24826 update (clip → step → eval, no noise
    /// call), over several updates — within [`assert_close`], as the two runs are separate Metal
    /// executions.
    #[test]
    fn sigma_zero_update_matches_the_legacy_update() {
        let (new, _, _) = run(NetworkType::Lora, 0.0, 3);
        let (_, _, mut legacy) = setup(NetworkType::Lora);
        let mut opt = TrainOptimizer::from_config("adamw", 1e-2, 0.0).unwrap();
        for u in 0..3 {
            let g = grads(&legacy, u);
            let (clipped, _) = clip_grad_norm(&g, 1.0).unwrap();
            let clipped: LoraParams = clipped
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect();
            opt.step(&mut legacy, &clipped).unwrap();
            eval(legacy.values()).unwrap();
        }
        for (k, v) in &legacy {
            assert_close(&vals(&new[k]), &vals(v), &format!("{k} at sigma 0"));
        }
    }

    /// E4: two seeded noisy runs produce the same adapter (within [`assert_close`]).
    #[test]
    fn seeded_noisy_runs_are_reproducible() {
        let (a, _, _) = run(NetworkType::Lora, 0.0125, 3);
        let (b, _, _) = run(NetworkType::Lora, 0.0125, 3);
        for (k, v) in &a {
            assert_close(&vals(&b[k]), &vals(v), &format!("{k} reproducible"));
        }
    }

    /// E3: the Z-Image MLX descriptor declares weight noise, so the shared floor accepts it.
    #[test]
    fn descriptor_declares_weight_noise() {
        assert!(trainer_descriptor().techniques.weight_noise);
    }

    /// sc-24828: Z-Image MLX honours subject-masked loss (cache + `reduce_loss` in the closure).
    #[test]
    fn descriptor_declares_subject_mask_loss() {
        assert!(trainer_descriptor().techniques.subject_mask_loss);
    }

    /// sc-24828 × sc-2127: with mask loss on and two buckets, each cache entry's weight map has
    /// THAT bucket's latent shape, and the background (right half of the center-square crop, with
    /// `background_weight` 0) is zero at both grids. The 48×32 image's center square is x ∈ [8, 40);
    /// the subject is x < 24 — the crop's left half.
    #[test]
    fn subject_mask_weight_is_computed_per_bucket() {
        let tmp = tempfile::tempdir().unwrap();
        let (iw, ih) = (48u32, 32u32);
        let image_path = tmp.path().join("a.png");
        image::RgbImage::new(iw, ih).save(&image_path).unwrap();
        let mask_path = tmp.path().join("a_mask.png");
        image::GrayImage::from_fn(iw, ih, |x, _| image::Luma([if x < 24 { 255 } else { 0 }]))
            .save(&mask_path)
            .unwrap();
        let item = gen_core::TrainingItem {
            image_path,
            caption: "a".into(),
            subject_mask_path: Some(mask_path),
            ..Default::default()
        };
        let cfg = gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask = PreparedSubjectMask::load("t", &item, &cfg).unwrap();
        let cap = Array::zeros::<f32>(&[8, 32]).unwrap();
        let edges = [32u32, 48];
        let mut cache = Vec::new();
        cache_item_buckets(
            &edges,
            Some(&mask),
            &cap,
            |edge| {
                let g = (edge / 8) as i32;
                Ok(Array::zeros::<f32>(&[16, 1, g, g])?)
            },
            &mut cache,
        )
        .unwrap();
        assert_eq!(cache.len(), 2);
        for (x0, _, w) in &cache {
            let w = w.as_ref().expect("mask loss on ⇒ a weight map");
            assert_eq!(
                w.shape(),
                x0.shape(),
                "weight must match its own bucket's latent"
            );
            let g = x0.shape()[3] as usize;
            let dense = multiply(w, Array::ones::<f32>(w.shape()).unwrap()).unwrap();
            let v = dense.as_slice::<f32>();
            // One [g, g] plane (all channels are the same broadcast plane).
            for y in 0..g {
                for x in 0..g {
                    let val = v[y * g + x];
                    if x < g / 2 {
                        assert!(val > 0.99, "subject cell ({y},{x}) of {g}x{g} = {val}");
                    } else {
                        assert_eq!(val, 0.0, "background cell ({y},{x}) of {g}x{g}");
                    }
                }
            }
        }
    }

    /// sc-2127: the descriptor declares resolution buckets, and the memory pre-flight sizes a
    /// bucketed run for its largest edge (the legacy single edge when buckets are off).
    #[test]
    fn buckets_are_declared_and_preflight_sizes_for_the_largest_edge() {
        assert!(trainer_descriptor().techniques.resolution_buckets);
        let mut cfg = TrainingConfig {
            resolution: 768,
            ..Default::default()
        };
        assert_eq!(preflight_edge(&bucket_edges(&cfg)), 768);
        cfg.resolution_buckets = vec![
            gen_core::ResolutionBucket {
                resolution: 1024,
                repeats: 1,
            },
            gen_core::ResolutionBucket {
                resolution: 512,
                repeats: 16,
            },
        ];
        assert_eq!(bucket_edges(&cfg), vec![1024, 512]);
        assert_eq!(preflight_edge(&bucket_edges(&cfg)), 1024);
    }
}

/// sc-2125 (epic 2123 depth anchoring) — the Z-Image step seam on the tiny synthetic DiT fixture
/// (`tests/fixtures/z_transformer.safetensors`: dim 96, 2 main + 1 refiner layers, 4 latent
/// channels) with a random-init tiny TAESD decoder (4 latent channels) and a random-init tiny
/// Depth-Anything-V2. Drives the same [`run_train_step`] / [`aux_driver`] /
/// [`compute_loss_grads`] `train_impl` runs. Seconds, a few MB; no weights downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use mlx_gen::train::perceptual::AuxLossSchedule;
    use mlx_gen::weights::Weights;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/z_transformer.safetensors"
    );

    fn tiny_dit() -> ZImageTransformer {
        let cfg = crate::ZImageTransformerConfig {
            patch_size: 2,
            f_patch_size: 1,
            in_channels: 4,
            dim: 96,
            n_layers: 2,
            n_refiner_layers: 1,
            n_heads: 4,
            norm_eps: 1e-5,
            cap_feat_dim: 32,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![8, 8, 8],
            axes_lens: vec![64, 64, 64],
            frequency_embedding_size: 256,
        };
        ZImageTransformer::from_weights(&Weights::from_file(FIXTURE).unwrap(), "w", cfg).unwrap()
    }

    fn schedule() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    fn path() -> PerceptualPath {
        mlx_gen_perceptual::testing::tiny_depth_path(4, schedule()).unwrap()
    }

    fn cfg() -> TrainingConfig {
        let mut cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            train_dtype: "f32".into(),
            ..Default::default()
        };
        cfg.depth_anchoring.schedule = schedule();
        cfg
    }

    /// `n` cached items: clean `[4, 1, 4, 4]` latents (32×32 decoded) + `[8, 32]` caption feats.
    fn cache_n(n: u64) -> Vec<CacheEntry> {
        (0..n)
            .map(|i| {
                let x0 = random::normal::<f32>(
                    &[4, 1, 4, 4],
                    None,
                    None,
                    Some(&random::key(100 + i).unwrap()),
                )
                .unwrap();
                let cap = random::normal::<f32>(
                    &[8, 32],
                    None,
                    None,
                    Some(&random::key(200 + i).unwrap()),
                )
                .unwrap();
                eval([&x0, &cap]).unwrap();
                (x0, cap, None)
            })
            .collect()
    }

    fn cache() -> Vec<CacheEntry> {
        cache_n(3)
    }

    fn adapter(dit: &mut ZImageTransformer, cfg: &TrainingConfig) -> (TrainAdapter, LoraParams) {
        let paths = resolve_target_paths(dit, cfg);
        assert!(!paths.is_empty());
        let (targets, params) = build_lora_targets(dit, &paths, cfg.rank as i32, cfg.seed).unwrap();
        (TrainAdapter::Lora { targets }, params)
    }

    /// A prepared driver (references built, alternation over `schedule`'s items, as `train_impl`
    /// does).
    fn prepared_with(cache: &[CacheEntry], schedule: &BucketSchedule, accum: u32) -> AuxDriver {
        aux_driver(path(), cache, None, schedule, accum, &Default::default()).unwrap()
    }

    /// [`prepared_with`] for a single-bucket cache (one entry per item).
    fn prepared(cache: &[CacheEntry], accum: u32) -> AuxDriver {
        prepared_with(cache, &single_bucket(cache), accum)
    }

    /// The single-bucket schedule `train_impl` builds when buckets are off (round-robin).
    fn single_bucket(cache: &[CacheEntry]) -> BucketSchedule {
        BucketSchedule::new(
            cache.len(),
            &[gen_core::train::ResolutionBucket {
                resolution: 64,
                repeats: 1,
            }],
            7,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn step_with(
        dit: &mut ZImageTransformer,
        params: &LoraParams,
        adapter: &TrainAdapter,
        cfg: &TrainingConfig,
        cache: &[CacheEntry],
        schedule: &BucketSchedule,
        path: Option<&mut AuxDriver>,
        n: u32,
    ) -> (StepLosses, LoraParams) {
        let (l, g) = run_train_step(
            dit,
            params,
            adapter,
            cfg,
            cache,
            schedule,
            path,
            n,
            false,
            None,
            Dtype::Float32,
        )
        .unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    fn step(
        dit: &mut ZImageTransformer,
        params: &LoraParams,
        adapter: &TrainAdapter,
        cfg: &TrainingConfig,
        cache: &[CacheEntry],
        path: Option<&mut AuxDriver>,
        n: u32,
    ) -> (StepLosses, LoraParams) {
        let schedule = single_bucket(cache);
        step_with(dit, params, adapter, cfg, cache, &schedule, path, n)
    }

    /// Run micro-steps `1..=steps` through the real step seam over `schedule`; per step
    /// `(dataset item, is_depth_step)`. Returns the driver too, for its reference counter.
    fn run_kinds_with(
        cache: &[CacheEntry],
        schedule: &BucketSchedule,
        accum: u32,
        steps: u32,
    ) -> (Vec<(usize, bool)>, AuxDriver) {
        let mut dit = tiny_dit();
        let cfg = cfg();
        let (adapter, params) = adapter(&mut dit, &cfg);
        let mut d = prepared_with(cache, schedule, accum);
        let kinds = (1..=steps)
            .map(|n| {
                let (l, _) = step_with(
                    &mut dit,
                    &params,
                    &adapter,
                    &cfg,
                    cache,
                    schedule,
                    Some(&mut d),
                    n,
                );
                assert_eq!(
                    l.diffusion.is_none(),
                    l.aux.is_some(),
                    "step {n}: a step is diffusion XOR depth here: {l:?}"
                );
                (schedule.sample((n - 1) as usize).0, l.aux.is_some())
            })
            .collect();
        (kinds, d)
    }

    fn run_kinds(cache: &[CacheEntry], accum: u32, steps: u32) -> Vec<(usize, bool)> {
        run_kinds_with(cache, &single_bucket(cache), accum, steps).0
    }

    fn abs_sum(g: &LoraParams, filter: &str) -> f32 {
        let picked: Vec<f32> = g
            .iter()
            .filter(|(k, _)| k.ends_with(filter))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .collect();
        assert!(
            !picked.is_empty(),
            "no '{filter}' grads in {:?}",
            g.keys().collect::<Vec<_>>()
        );
        picked.iter().sum()
    }

    /// AC1: a depth step (an image's every 2nd update) produces a non-zero gradient on the LoRA
    /// factors while the diffusion term contributes nothing (the differentiated total IS the
    /// weighted depth term); a diffusion step carries no depth term. Mutation: force
    /// `diffusion_on = true` in `compute_loss_grads` ⇒ the depth step reports a diffusion term and
    /// total ≠ aux ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only() {
        let mut dit = tiny_dit();
        let cfg = cfg();
        let (adapter, params) = adapter(&mut dit, &cfg);
        let cache = cache_n(1);
        let mut d = prepared(&cache, 1);

        let (diff, _) = step(&mut dit, &params, &adapter, &cfg, &cache, Some(&mut d), 1);
        assert_eq!(diff.aux, None, "diffusion step carries no depth term");
        assert_eq!(Some(diff.total), diff.diffusion);

        let (depth, g) = step(&mut dit, &params, &adapter, &cfg, &cache, Some(&mut d), 2);
        assert_eq!(
            depth.diffusion, None,
            "depth step must not compute the diffusion loss"
        );
        let aux = depth.aux.expect("depth step carries the depth term");
        assert!(aux > 0.0 && aux.is_finite(), "depth term {aux}");
        assert_eq!(
            depth.total, aux,
            "the differentiated loss is the depth term alone"
        );
        // LoRA B starts at zero, so dL/dB is where a depth gradient first appears.
        let gb = abs_sum(&g, ".lora_b");
        assert!(
            gb > 0.0 && gb.is_finite(),
            "depth step LoRA-B grad |Σ| = {gb}"
        );
    }

    /// Review blocker, through the real step seam: with N = 2 and N = 4 images and `every_n = 2`,
    /// every image gets at least one diffusion step and one depth step within 2·N steps (a
    /// global-step key would lock even images to diffusion and odd images to depth). Mutation: key
    /// the plan on the global step (`path.plan(step, …)`) ⇒ red.
    #[test]
    fn every_image_gets_diffusion_and_depth_steps() {
        for n in [2u64, 4] {
            let cache = cache_n(n);
            let kinds = run_kinds(&cache, 1, 2 * n as u32);
            for image in 0..n as usize {
                let mine: Vec<bool> = kinds
                    .iter()
                    .filter(|(i, _)| *i == image)
                    .map(|(_, d)| *d)
                    .collect();
                assert!(
                    mine.contains(&true) && mine.contains(&false),
                    "N={n}: image {image} got {mine:?} ({kinds:?})"
                );
            }
        }
    }

    /// Review major, through the real step seam: with gradient accumulation 2 both micro-steps of
    /// each optimizer window share one step kind (no averaged diffusion+depth update), and depth
    /// updates still happen. Mutation: build the tracker with accumulation 1 in `prepared` ⇒ red.
    #[test]
    fn accumulation_windows_share_one_step_kind() {
        let cache = cache();
        let kinds = run_kinds(&cache, 2, 8);
        for w in kinds.chunks(2) {
            assert_eq!(w[0].1, w[1].1, "window {w:?} mixes step kinds ({kinds:?})");
        }
        assert!(kinds.iter().any(|(_, d)| *d), "{kinds:?}");
    }

    /// sc-2127 integration: with two resolution buckets (item-major cache, seeded shuffle) and
    /// depth on, the optimizer windows interleave (never two depth steps in a row, sc-2124) and
    /// every image gets both kinds across its buckets — and the depth reference is built once per
    /// (image, bucket) cache entry (each bucket's clean latent decodes to its own size), so the
    /// counter equals the entry count after two epochs; a depth step trains on the scheduled
    /// entry's latent against that entry's reference. Mutations: key the alternation per epoch
    /// (`AuxAlternation::new(1, epoch_len)`) in `AuxDriver::prepare_keyed` ⇒ a depth run ⇒ red; key
    /// the reference on the item, or pick the latent round-robin instead of from the schedule ⇒ the
    /// depth term differs from the scheduled entry's ⇒ red.
    #[test]
    fn two_buckets_interleave_with_per_entry_references() {
        let items = 2usize;
        let buckets = [
            gen_core::train::ResolutionBucket {
                resolution: 32,
                repeats: 1,
            },
            gen_core::train::ResolutionBucket {
                resolution: 48,
                repeats: 1,
            },
        ];
        // Item-major: cache[item * 2 + bucket]; bucket 0 = 4×4 latents, bucket 1 = 6×6.
        let cache: Vec<CacheEntry> = (0..items as u64)
            .flat_map(|i| {
                [4i32, 6].into_iter().map(move |side| {
                    let x0 = random::normal::<f32>(
                        &[4, 1, side, side],
                        None,
                        None,
                        Some(&random::key(300 + 10 * i + side as u64).unwrap()),
                    )
                    .unwrap();
                    let cap = random::normal::<f32>(
                        &[8, 32],
                        None,
                        None,
                        Some(&random::key(400 + i).unwrap()),
                    )
                    .unwrap();
                    eval([&x0, &cap]).unwrap();
                    (x0, cap, None)
                })
            })
            .collect();
        let schedule = BucketSchedule::new(items, &buckets, 7);
        assert_eq!(schedule.n_buckets(), 2);
        let steps = 2 * schedule.epoch_len() as u32;

        let mut dit = tiny_dit();
        let cfg = cfg();
        let (adapter, params) = adapter(&mut dit, &cfg);
        let mut d = prepared_with(&cache, &schedule, 1);
        let mut kinds = Vec::new();
        let mut checked_depth_step = false;
        for n in 1..=steps {
            let (l, _) = step_with(
                &mut dit,
                &params,
                &adapter,
                &cfg,
                &cache,
                &schedule,
                Some(&mut d),
                n,
            );
            let k = (n - 1) as usize;
            let (item, _) = schedule.sample(k);
            let entry = schedule.cache_index(k);
            kinds.push((item, l.aux.is_some()));
            // Every step recomputed with the scheduled entry's latent and reference (and, on depth
            // steps whose entry differs from the item index, that proves the reference key).
            {
                let key =
                    mlx_gen::train::perceptual::AuxAlternation::new(schedule.epoch_len(), 1).key(n);
                let raw = sample_sigma(
                    &cfg.timestep_type,
                    &cfg.timestep_bias,
                    cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(n as u64),
                )
                .unwrap();
                let plan = d.path().plan(key, entry, raw).unwrap();
                let (x0, cap, _) = &cache[entry];
                let noise = random::normal::<f32>(
                    x0.shape(),
                    None,
                    None,
                    Some(
                        &random::key(cfg.seed.wrapping_add(n as u64).wrapping_mul(2) + 1).unwrap(),
                    ),
                )
                .unwrap();
                let (expected, _) = compute_loss_grads(
                    &mut dit,
                    &params,
                    &adapter,
                    cfg.alpha,
                    cfg.rank as f32,
                    x0,
                    cap,
                    plan.noise_level,
                    &noise,
                    false,
                    None,
                    None,
                    Dtype::Float32,
                    Some(AuxStep {
                        path: d.path(),
                        plan: &plan,
                        image: entry,
                    }),
                )
                .unwrap();
                assert_eq!(l, expected, "step {n} (item {item}, entry {entry})");
                checked_depth_step |= l.aux.is_some() && entry != item;
            }
        }
        assert!(
            checked_depth_step,
            "no depth step on an entry != its item: {kinds:?}"
        );
        // The seeded schedule is not the round-robin walk (so the recompute above also pins the
        // latent choice to the schedule).
        assert!((0..steps as usize).any(|k| schedule.cache_index(k) != k % cache.len()));
        // Both buckets are actually visited.
        let entries: std::collections::BTreeSet<usize> = (0..steps as usize)
            .map(|k| schedule.cache_index(k))
            .collect();
        assert_eq!(entries.len(), cache.len());
        assert!(
            kinds.windows(2).all(|w| !(w[0].1 && w[1].1)),
            "two depth steps in a row: {kinds:?}"
        );
        for image in 0..items {
            let mine: Vec<bool> = kinds
                .iter()
                .filter(|(i, _)| *i == image)
                .map(|(_, d)| *d)
                .collect();
            assert!(
                mine.contains(&true) && mine.contains(&false),
                "image {image} ({kinds:?})"
            );
        }
        assert_eq!(d.path().reference_computations(), cache.len());
    }

    /// AC2: the reference depth is computed once per image per job (per (image, bucket) cache
    /// entry with buckets on; one bucket here) — three epochs over three images leave the counter
    /// at 3 (only the `AuxDriver` builds references; the step never does). Mutation: build the
    /// references in `AuxDriver::prepare_keyed` for one entry too few ⇒ red.
    #[test]
    fn reference_depth_is_computed_once_per_image_across_epochs() {
        let mut dit = tiny_dit();
        let cfg = cfg();
        let (adapter, params) = adapter(&mut dit, &cfg);
        let cache = cache();
        let mut d = prepared(&cache, 1);
        assert_eq!(d.path().reference_computations(), cache.len());
        for n in 1..=(3 * cache.len() as u32) {
            step(&mut dit, &params, &adapter, &cfg, &cache, Some(&mut d), n);
        }
        assert_eq!(d.path().reference_computations(), cache.len());
    }

    /// E1: with depth anchoring off nothing is loaded and the step is bit-identical to the
    /// pre-epic-2123 step (the legacy loss closure, reproduced here verbatim); a diffusion-only step
    /// of an enabled path is bit-identical too. Mutation: always add `aux_term` (e.g. a 0-weighted
    /// copy) or reorder the loss reduction ⇒ red.
    #[test]
    fn everything_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path(&TrainingConfig::default())
            .unwrap()
            .is_none());
        assert_eq!(
            perceptual_footprint_gb(&TrainingConfig::default(), 1024, 10),
            0.0
        );

        let mut dit = tiny_dit();
        let cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        let (adapter, params) = adapter(&mut dit, &cfg);
        let cache = cache();
        let (off, g_off) = step(&mut dit, &params, &adapter, &cfg, &cache, None, 1);
        assert_eq!(off.aux, None);

        // The pre-sc-2125 `compute_loss_grads` body for step 1 (same item / sigma / noise).
        let (x0, cap, _) = &cache[0];
        let sigma = sample_sigma(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1),
        )
        .unwrap();
        let noise = random::normal::<f32>(
            x0.shape(),
            None,
            None,
            Some(&random::key(cfg.seed.wrapping_add(1).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let (x_t, target, timestep) = build_batch(x0, &noise, sigma).unwrap();
        let capf = cap.clone();
        let dit_ref = &mut dit;
        let adapter_ref = &adapter;
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            adapter_ref.install_as(dit_ref, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            let sh = x_t.shape();
            let prep = dit_ref
                .prepare((sh[0], sh[1], sh[2], sh[3]), &capf)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let v = dit_ref
                .forward_with(&prep, &x_t, timestep)
                .map_err(|e| Exception::custom(e.to_string()))?;
            Ok(vec![subtract(&v, &target)?.square()?.mean(None)?])
        };
        let (val, g_legacy) = keyed_value_and_grad(legacy)(params.clone(), 0).unwrap();
        eval(g_legacy.values()).unwrap();
        assert_eq!(off.total, val[0].item::<f32>());
        assert_eq!(off.diffusion, Some(off.total));
        let bits =
            |a: &Array| -> Vec<u32> { a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect() };
        for (k, v) in &g_legacy {
            assert_eq!(
                bits(v),
                bits(&g_off[k]),
                "{k}: off grads differ from the legacy step"
            );
        }

        // A diffusion-only step of an enabled path takes exactly the same graph.
        let mut dit2 = tiny_dit();
        let dcfg = self::cfg();
        let (adapter2, params2) = self::adapter(&mut dit2, &dcfg);
        let mut d = prepared(&cache, 1);
        let (on1, g_on1) = step(
            &mut dit2,
            &params2,
            &adapter2,
            &dcfg,
            &cache,
            Some(&mut d),
            1,
        );
        let mut dit3 = tiny_dit();
        let (_, params3) = self::adapter(&mut dit3, &dcfg);
        let (none1, g_none1) = step(&mut dit3, &params3, &adapter2, &dcfg, &cache, None, 1);
        assert_eq!(on1, none1);
        for (k, v) in &g_none1 {
            assert_eq!(bits(v), bits(&g_on1[k]), "{k}");
        }
    }

    /// E7: depth anchoring grows the trainer memory estimate by the TAEF1 + DA2 footprint, more for
    /// a larger DA2, and the pre-flight guard counts it on the dense and the checkpointed path
    /// (host-independent: synthetic budgets).
    #[test]
    fn memory_estimate_includes_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule();
        let small = perceptual_footprint_gb(&on, 1024, 10);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = perceptual_footprint_gb(&on, 1024, 10);
        assert!(
            small > 0.0 && large > small,
            "small {small} GB, large {large} GB"
        );
        // The DA2-Large weights alone are ~1.3 GB.
        assert!(large - small > 1.0, "large - small = {} GB", large - small);

        // The guard against fixed synthetic budgets (never the host's): a budget whose safe share
        // sits halfway between the DiT projection and projection + aux admits the plain run and
        // refuses the depth run — on the dense path and on the (default) checkpointed path.
        // Mutations: drop `+ extra_gb` (dense or checkpointed branch) ⇒ the aux case passes ⇒ red;
        // skip the checkpointed branch's projection ⇒ red.
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let between = |base_gb: f64| ((base_gb + large / 2.0) / 0.85 * GIB) as usize;
        let dense_tokens = (64.0f64 / 16.0).ceil().powi(2) + 544.0;
        let dense_base = projected_dense_peak_gb(dense_tokens, true);
        let budget = between(dense_base);
        assert!(preflight_memory_guard_with_budget(&on, 64, true, 0.0, false, budget).is_ok());
        assert!(preflight_memory_guard_with_budget(&on, 64, true, large, false, budget).is_err());
        let ckpt_base = checkpointed_baseline_gb(true);
        let budget = between(ckpt_base);
        assert!(preflight_memory_guard_with_budget(&on, 1024, true, 0.0, true, budget).is_ok());
        let err = preflight_memory_guard_with_budget(&on, 1024, true, large, true, budget)
            .unwrap_err()
            .to_string();
        // The checkpointed refusal names the enabled aux losses (the shared E7 guard).
        assert!(err.contains("[depth]"), "{err}");
        // A budget comfortably above projection + aux admits the depth run on both paths.
        let roomy = ((dense_base.max(ckpt_base) + large) / 0.85 * GIB) as usize * 2;
        assert!(preflight_memory_guard_with_budget(&on, 64, true, large, false, roomy).is_ok());
        assert!(preflight_memory_guard_with_budget(&on, 1024, true, large, true, roomy).is_ok());
    }

    /// E3: the Z-Image MLX descriptor declares depth anchoring.
    #[test]
    fn descriptor_declares_depth_anchoring() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
        // sc-24831: the face losses ride the same builder arms.
        assert!(trainer_descriptor().techniques.identity_loss);
        assert!(trainer_descriptor().techniques.face_landmark_loss);
    }

    /// sc-24832: Z-Image builds its perceptual path through the builder, so it declares the body
    /// losses the builder's arms provide.
    #[test]
    fn descriptor_declares_the_body_losses() {
        let t = trainer_descriptor().techniques;
        assert!(t.body_proportion_loss && t.body_shape_loss && t.normal_loss);
    }

    /// A missing aux checkpoint is a clear error naming the model, before any caching.
    #[test]
    fn missing_aux_weights_are_named() {
        let tmp = tempfile::tempdir().unwrap();
        let mut c = cfg();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taef1"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c)
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("TAEF1"), "{err}");
    }

    /// sc-24833 (epic 2123 E8): the two latent-space perceptual losses through Z-Image's REAL step
    /// seam — a path holding the VAE anchor (decoded-x0, FLUX.2-encoder graph on formula weights)
    /// and E-LatentLPIPS (latent input; 4-channel formula weights for the 4-channel fixture DiT),
    /// both additive (`every_n = 1`). Each image's references are built once however many steps run;
    /// every step carries the diffusion term AND the weighted aux term; and the aux term alone moves
    /// the LoRA gradient (the same step with the losses zero-weighted differs). Mutations: drop the
    /// `aux_term` from `combine_step_loss` in `compute_loss_grads` ⇒ the gradients match ⇒ red;
    /// build the references in `AuxDriver::prepare_keyed` for one entry too few ⇒ red.
    #[test]
    fn latent_perceptual_losses_train_the_lora_through_the_step_seam() {
        use mlx_gen::train::latent_lpips::{
            formula_weights as lpips_w, LatentLpips, LatentLpipsLoss,
        };
        use mlx_gen::train::perceptual::AuxLoss;
        use mlx_gen::train::tae::{synthetic_tiny_decoder_weights, TinyDecoder};
        use mlx_gen::train::vae_anchor::{
            formula_weights as va_w, VaeAnchorEncoder, VaeAnchorEncoderConfig, VaeAnchorLoss,
        };
        let build = |weight: f32| {
            let sched = AuxLossSchedule {
                weight,
                t_min: 0.0,
                t_max: 1.0,
                every_n: 1,
            };
            let dec_cfg = mlx_gen_perceptual::testing::tiny_decoder_config(4);
            let dec = TinyDecoder::from_weights(
                &synthetic_tiny_decoder_weights(&dec_cfg, 11).unwrap(),
                dec_cfg,
            )
            .unwrap();
            let va_cfg = VaeAnchorEncoderConfig::with_base(32);
            let va =
                VaeAnchorLoss::new(VaeAnchorEncoder::from_weights(&va_w(&va_cfg), va_cfg).unwrap());
            let lp = LatentLpipsLoss::new(LatentLpips::from_weights(&lpips_w(4), 4).unwrap());
            PerceptualPath::new(
                Some(Box::new(dec)),
                vec![
                    AuxLoss {
                        schedule: sched,
                        loss: Box::new(va),
                    },
                    AuxLoss {
                        schedule: sched,
                        loss: Box::new(lp),
                    },
                ],
            )
            .unwrap()
        };
        let mut dit = tiny_dit();
        let cfg = cfg();
        let (adapter, params) = adapter(&mut dit, &cfg);
        let cache = cache_n(2);
        let run = |dit: &mut ZImageTransformer, weight: f32, steps: u32| {
            let mut d = aux_driver(
                build(weight),
                &cache,
                None,
                &single_bucket(&cache),
                1,
                &Default::default(),
            )
            .unwrap();
            let mut out = Vec::new();
            for n in 1..=steps {
                out.push(step(dit, &params, &adapter, &cfg, &cache, Some(&mut d), n));
            }
            (out, d.path().reference_computations())
        };
        let (steps, refs) = run(&mut dit, 0.5, 4);
        assert_eq!(refs, 2, "references are built once per image");
        for (n, (l, _)) in steps.iter().enumerate() {
            let aux = l
                .aux
                .unwrap_or_else(|| panic!("step {n}: additive aux term missing: {l:?}"));
            let diff = l.diffusion.expect("additive: the diffusion term stays");
            assert!(aux > 0.0 && aux.is_finite(), "step {n}: aux {aux}");
            assert!(
                (l.total - (diff + aux)).abs() <= 1e-4 * l.total.abs().max(1.0),
                "{l:?}"
            );
        }
        // Same step, losses scheduled but zero-weighted vs weighted: only the aux term differs.
        let (zero, _) = run(&mut dit, 1e-12, 1);
        let (with, _) = run(&mut dit, 0.5, 1);
        let (gz, gw) = (
            abs_sum(&zero[0].1, ".lora_b"),
            abs_sum(&with[0].1, ".lora_b"),
        );
        assert!(
            (gz - gw).abs() > 1e-6 * gz.max(1e-12),
            "the aux term must move the LoRA gradient: {gz} vs {gw}"
        );
    }
}

/// sc-24832: the Z-Image reference preparation hands each cache entry its item's subject mask —
/// cropped like the image (centre square) and resampled onto that entry's decoded grid — to a
/// mask-reading loss. Mutation: drop `path.attach_subject_masks(masks)` from
/// `aux_driver` ⇒ the probe sees no mask ⇒ red.
#[cfg(test)]
mod subject_mask_reference_tests {
    use super::*;
    use std::any::Any;
    use std::cell::RefCell;
    use std::rc::Rc;

    use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule, LossReference, PerceptualLoss};
    use mlx_gen::train::subject_mask::PerceptualSubjectMasks;
    use mlx_gen::train::tae::{synthetic_tiny_decoder_weights, TinyDecoder};
    use mlx_gen::train::TrainingItem;

    type Seen = Rc<RefCell<Vec<Option<(Vec<i32>, Vec<f32>)>>>>;
    /// Records the mask every reference receives: `(shape, values)`, or `None`.
    struct MaskProbe(Seen);
    impl PerceptualLoss for MaskProbe {
        fn name(&self) -> &'static str {
            "mask-probe"
        }
        fn reference(&self, clean: &Array) -> mlx_gen::Result<Option<LossReference>> {
            self.reference_with_mask(clean, None)
        }
        fn reference_with_mask(
            &self,
            _clean: &Array,
            mask: Option<&Array>,
        ) -> mlx_gen::Result<Option<LossReference>> {
            self.0.borrow_mut().push(mask.map(|m| {
                m.eval().unwrap();
                (m.shape().to_vec(), m.as_slice::<f32>().to_vec())
            }));
            Ok(Some(Box::new(())))
        }
        fn loss(&self, live: &Array, _r: &dyn Any) -> mlx_gen::Result<Array> {
            Ok(live.mean(None)?)
        }
    }

    fn write_png(
        dir: &std::path::Path,
        name: &str,
        f: impl Fn(u32, u32) -> u8,
    ) -> std::path::PathBuf {
        let p = dir.join(name);
        image::GrayImage::from_fn(12, 8, |x, y| image::Luma([f(x, y)]))
            .save(&p)
            .unwrap();
        p
    }

    #[test]
    fn the_reference_path_hands_each_entry_its_items_mask() {
        let dir = tempfile::tempdir().unwrap();
        let img = write_png(dir.path(), "img.png", |_, _| 128);
        let masks = [
            write_png(dir.path(), "right.png", |x, _| if x >= 6 { 255 } else { 0 }),
            write_png(dir.path(), "top.png", |_, y| if y < 4 { 255 } else { 0 }),
        ];
        let items: Vec<TrainingItem> = masks
            .iter()
            .map(|m| {
                let mut it = TrainingItem::captioned(img.clone(), "c".into());
                it.subject_mask_path = Some(m.clone());
                it
            })
            .collect();
        let mut cfg = TrainingConfig::default();
        cfg.body_losses.normal.weight = 0.1;
        cfg.body_losses.normal_restrict_to_subject = true;
        // Two items × one bucket; [4, 1, 4, 4] latents decode to 32×32.
        let cache: Vec<CacheEntry> = (0..2u64)
            .map(|i| {
                let x0 = random::normal::<f32>(
                    &[4, 1, 4, 4],
                    None,
                    None,
                    Some(&random::key(i).unwrap()),
                )
                .unwrap();
                (x0, Array::from_f32(0.0), None)
            })
            .collect();
        let dcfg = mlx_gen_perceptual::testing::tiny_decoder_config(4);
        let dec =
            TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&dcfg, 3).unwrap(), dcfg)
                .unwrap();
        let seen = Rc::new(RefCell::new(Vec::new()));
        let path = PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule: AuxLossSchedule {
                    weight: 0.1,
                    t_min: 0.0,
                    t_max: 1.0,
                    every_n: 1,
                },
                loss: Box::new(MaskProbe(seen.clone())),
            }],
        )
        .unwrap();
        let loaded =
            PerceptualSubjectMasks::load("t", &items, &cfg, 1, CropBox::center_square).unwrap();
        let expected = loaded.clone().unwrap();
        let schedule = BucketSchedule::new(
            cache.len(),
            &[gen_core::train::ResolutionBucket {
                resolution: 256,
                repeats: 1,
            }],
            7,
        );
        aux_driver(path, &cache, loaded, &schedule, 1, &Default::default()).unwrap();
        let seen = seen.borrow();
        assert_eq!(seen.len(), 2);
        for (entry, got) in seen.iter().enumerate() {
            let (shape, values) = got.as_ref().expect("every reference gets its mask");
            assert_eq!(shape, &vec![32, 32]);
            assert_eq!(
                values,
                &expected.pixel_mask(entry, 32, 32).unwrap(),
                "entry {entry}"
            );
        }
        assert_ne!(seen[0], seen[1], "each entry carries its own item's mask");
        // Off ⇒ no mask reaches the loss.
        let none = PerceptualSubjectMasks::load(
            "t",
            &items,
            &TrainingConfig::default(),
            1,
            CropBox::center_square,
        )
        .unwrap();
        assert!(none.is_none());
    }
}
