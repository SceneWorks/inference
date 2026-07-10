//! LoRA/LoKr **training** on the Anima flow-match loop (sc-10522, epic 10512) — pure Rust on mlx-rs.
//!
//! Anima is trainable, so per standing guidance it ships with training. This realizes the core
//! [`Trainer`] contract (gen-core `train.rs`) on the real Anima model — the 28-block Cosmos-Predict2
//! DiT **and** the bundled `AnimaTextConditioner` (the `llm_adapter`). It is modeled on
//! `mlx-gen-z-image::training` (the closest analogue: a Qwen3-encoded flow-match DiT that already
//! trains LoRA + LoKr), and reuses the family-agnostic factor machinery in [`mlx_gen::train::lora`].
//!
//! ## Two hosts, one adapter surface
//! Unlike z-image (one DiT host), Anima adapters route into **two** sub-models via the sc-10521
//! [`AnimaAdapterHost`] seam: `llm_adapter.*` → the conditioner, everything else → the DiT. The
//! trainer enumerates its trainable targets from exactly that host, so the default trainable set is
//! the **508** targets the official `anima-turbo-lora-v0.2` carries — **448** DiT (`blocks.N.*`:
//! self/cross-attn q/k/v/o, mlp ×2, and the three `adaln_modulation_*.{1,2}` down/up pairs) + **60**
//! conditioner (6 blocks × {self/cross-attn q/k/v/o, mlp.0, mlp.2}). Training only the DiT would
//! produce a 448-target file that cannot reproduce the official injection surface.
//!
//! ## The conditioner genuinely trains
//! The conditioner (`llm_adapter`) is a first-class trainable target, so its adapter factors must
//! receive real gradients. We therefore **cache the conditioner's INPUTS** — the (masked) Qwen3
//! `last_hidden_state` + the T5 query-token ids, which are deterministic per caption and produced by
//! the multi-GB Qwen3 encoder (freed post-caching) — and run the (cheap, 6-block) conditioner forward
//! **inside the traced grad graph** each step, with the conditioner adapters injected. Caching the
//! conditioner's *output* instead would make its adapters inert (no gradient path), so the input-cache
//! is the faithful analogue of OneTrainer/mgds `EncodeAnimaText`: the expensive deterministic encoder
//! output is cached, the trained component stays live.
//!
//! ## Flow-match objective (`shift=3.0`)
//! Anima's DiT is a standard flow denoiser: it predicts the velocity `v ≈ ε − x0` and embeds the raw
//! (shifted) σ as its timestep (`pipeline.rs`), so the regression target is `noise − x0` with **no**
//! output negation (the opposite of z-image, whose `forward()` is pre-negated and whose timestep is
//! `1 − σ`). A base timestep is sampled, then run through the same static `shift = 3.0`
//! (`3σ/(1+2σ)`, [`SIGMA_SHIFT`]) the inference schedule uses, and that shifted σ drives both the
//! `x_t = (1−σ)·x0 + σ·noise` interpolation and the DiT timestep.
//!
//! ## Round-trip
//! The trained adapter round-trips through the sc-10521 inference loader: LoRA is saved with the
//! ComfyUI `diffusion_model.` prefix + PEFT `lora_A`/`lora_B` keys and **no alpha** (the shipped Anima
//! convention — the α/rank fold is baked into `lora_B` so scale-1.0 loading is exact for any alpha);
//! LoKr is saved by the shared [`save_lokr`] in the bare-path `lokr_*` convention the sc-10521 LoKr
//! path consumes. Both reconstruct the residual at **bf16** to match the inference loader.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost};
use mlx_gen::gen_core;
use mlx_gen::media::Image;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, save_lokr, LoraParams,
    TrainAdapter,
};
pub use mlx_gen::train::lora::{LokrTarget, LoraTarget};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Precision, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::adapters::AnimaAdapterHost;
use crate::conditioner::AnimaTextConditioner;
use crate::config::{Variant, SIGMA_SHIFT};
use crate::loader::AnimaComponents;
use crate::text_encoder::AnimaQwen3;
use crate::tokenizer::AnimaTokenizers;
use crate::transformer::CosmosDiT;
use crate::vae::QwenVae;

/// The inference LoKr loader (`apply_lokr`) reconstructs the Kronecker delta at bf16 (`loader.rs`);
/// training must match so the adapter round-trips.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Saved-adapter storage dtype. The trainable factors are f32 master-weights, but the shipped Anima
/// adapters (`anima-turbo-lora-v0.2`, `anima-greg-rutkowski-style`) are **bf16**, and the inference
/// loader reconstructs the residual at bf16 regardless — so the saved factors are cast to bf16 to
/// match the shipped convention and halve file size (~138 MB → ~69 MB), with no round-trip loss
/// beyond the bf16 rounding the loader would apply anyway.
const SAVE_DTYPE: Dtype = Dtype::Bfloat16;

/// ComfyUI adapter key prefix the official Anima LoRAs (and the sc-10521 inference loader) use —
/// `diffusion_model.blocks.*` for the DiT, `diffusion_model.llm_adapter.blocks.*` for the conditioner.
const KEY_PREFIX: &str = "diffusion_model.";

// ==================================================================================================
// Flow-match batch construction
// ==================================================================================================

/// The static flow-match time-shift the Anima schedule applies (`shift·σ / (1 + (shift−1)·σ)`),
/// concentrating sampling toward higher noise. Mirrors `pipeline::anima_sigmas` so training and
/// inference share the same σ warp. `shift = 1` is the identity.
fn apply_static_shift(sigma: f32, shift: f32) -> f32 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma` (already shift-warped):
/// `x_t = (1−σ)·x0 + σ·noise`, `target = noise − x0` (the velocity `ε − x0`), `timestep = σ`.
/// Anima's `forward()` is **not** pre-negated (unlike z-image), so the target is the raw velocity and
/// the timestep is the raw σ (matching the inference `TimestepConvention::Sigma`).
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target, sigma))
}

// ==================================================================================================
// Request validation (capability-free, unit-testable)
// ==================================================================================================

/// Recognized `timestep_type` values (`sigmoid` is the fall-through default).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values (`balanced`/`none`/`neutral` are neutral).
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
/// Recognized `loss_type` values (`mse`/`l2` = MSE, `mae`/`l1` = MAE).
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// Normalize a free-form config string the way the parsers do (trim, lowercase, `-`/space → `_`).
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation (mirrors z-image's `validate_request`): rejects an
/// empty dataset, zero rank, zero steps, an unsupported optimizer, and an unrecognized
/// `timestep_type` / `timestep_bias` / `loss_type` (rather than silently falling back to a default).
/// The non-empty target-module resolution is checked in [`Trainer::validate`], which has the loaded
/// model to enumerate adaptable paths against.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("anima trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("anima trainer: rank must be > 0".into());
    }
    if req.config.steps == 0 {
        return Err("anima trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "anima trainer: optimizer '{}' is not available on MLX training (supported: adamw, \
             adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&req.config.timestep_type).as_str()) {
        return Err(format!(
            "anima trainer: timestep_type '{}' is not recognized (supported: {})",
            req.config.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&req.config.timestep_bias).as_str()) {
        return Err(format!(
            "anima trainer: timestep_bias '{}' is not recognized (supported: {})",
            req.config.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&req.config.loss_type).as_str()) {
        return Err(format!(
            "anima trainer: loss_type '{}' is not recognized (supported: {})",
            req.config.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target modules to the full dotted-path set on the Anima adapter surface (DiT
/// `blocks.*` + `llm_adapter.blocks.*`). An empty `lora_target_modules` (the default) trains the whole
/// **508**-target surface the official LoRAs carry; a non-empty set filters those paths by suffix (the
/// same suffix match PEFT's `LoraConfig(target_modules=…)` does). Computed from `&self` refs (no
/// `&mut` host needed) so [`Trainer::validate`] can call it. Mirrors [`AnimaAdapterHost::adaptable_paths`].
fn resolve_target_paths(
    dit: &CosmosDiT,
    conditioner: &AnimaTextConditioner,
    cfg: &TrainingConfig,
) -> Vec<String> {
    let mut all = dit.adaptable_paths();
    all.extend(prefixed_paths("llm_adapter", conditioner));
    if cfg.lora_target_modules.is_empty() {
        return all;
    }
    let suffixes = &cfg.lora_target_modules;
    all.into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

// ==================================================================================================
// The trainer
// ==================================================================================================

/// A LoRA/LoKr trainer for one Anima variant: the frozen base (Cosmos DiT + `AnimaTextConditioner` +
/// Qwen3 TE + Qwen-Image VAE + dual tokenizers), which caches a captioned dataset to VAE latents +
/// Qwen3 conditioner-inputs, then runs the functional-autograd flow-match loop and writes an adapter
/// that round-trips through the sc-10521 inference loader.
pub struct AnimaTrainer {
    descriptor: TrainerDescriptor,
    tokenizers: AnimaTokenizers,
    /// The Qwen3 encoder — in an `Option` so it can be **dropped after caching** (it is idle during
    /// training; every caption is already encoded to its cached `source_hidden`, and it is a multi-GB
    /// resident). The conditioner it feeds is NOT freed — it is a trained target.
    text_encoder: Option<AnimaQwen3>,
    vae: QwenVae,
    dit: CosmosDiT,
    conditioner: AnimaTextConditioner,
}

fn trainer_descriptor_for(variant: Variant) -> TrainerDescriptor {
    TrainerDescriptor {
        id: variant.id(),
        family: "anima",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

pub fn trainer_descriptor_base() -> TrainerDescriptor {
    trainer_descriptor_for(Variant::Base)
}
pub fn trainer_descriptor_aesthetic() -> TrainerDescriptor {
    trainer_descriptor_for(Variant::Aesthetic)
}
pub fn trainer_descriptor_turbo() -> TrainerDescriptor {
    trainer_descriptor_for(Variant::Turbo)
}

pub fn load_trainer_base(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant_trainer(spec, Variant::Base)
}
pub fn load_trainer_aesthetic(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant_trainer(spec, Variant::Aesthetic)
}
pub fn load_trainer_turbo(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant_trainer(spec, Variant::Turbo)
}

/// Construct the trainer from a `split_files/` snapshot (the multi-file Anima layout). No
/// quantization — training needs the dense bf16 base (the single wired precision).
fn load_variant_trainer(spec: &LoadSpec, variant: Variant) -> Result<Box<dyn Trainer>> {
    let id = variant.id();
    if spec.precision != Precision::Bf16 {
        return Err(mlx_gen::Error::Msg(format!(
            "{id} trainer: only the dense bf16 base is wired for training (drop the precision override)"
        )));
    }
    if spec.quantize.is_some() {
        return Err(mlx_gen::Error::Msg(format!(
            "{id} trainer: training needs the dense base; quantized tiers are not trainable"
        )));
    }
    let components = AnimaComponents::load(&spec.weights, variant)?;
    Ok(Box::new(AnimaTrainer {
        descriptor: trainer_descriptor_for(variant),
        tokenizers: components.tokenizers,
        text_encoder: Some(components.text_encoder),
        vae: components.vae,
        dit: components.dit,
        conditioner: components.conditioner,
    }))
}

// Link-time trainer registration (epic 3720) for all three variants, mirroring the generator side.
mlx_gen::register_trainer! {
    trainer_descriptor_base => load_trainer_base,
    trainer_descriptor_aesthetic => load_trainer_aesthetic,
    trainer_descriptor_turbo => load_trainer_turbo,
}

impl Trainer for AnimaTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_request(req)?;
        if resolve_target_paths(&self.dit, &self.conditioner, &req.config).is_empty() {
            return Err(format!(
                "anima trainer: lora_target_modules {:?} matched no adaptable module on the DiT or \
                 conditioner",
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

impl AnimaTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720).
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // Anima's base is bf16 on disk (there is no dense-f32 cast path), so training runs bf16
        // mixed-precision: the frozen base + activation stream are bf16, and the trainable factors /
        // loss / grads / optimizer stay f32 (master-weights). This matches the inference dtype.
        let compute_dtype = Dtype::Bfloat16;

        // --- prepare → cache: VAE latents + (masked Qwen3 states, T5 ids) into memory ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        // (x0 latent, masked Qwen3 source_hidden, T5 query-token ids).
        let mut cache: Vec<(Array, Array, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let nchw = mlx_gen_qwen_image::preprocess_init_image(&img, edge, edge)?; // [1,3,edge,edge]
            let x0 = self.vae.encode(&nchw)?; // [1,16,1,edge/8,edge/8], normalized
            let (source, t5_ids) = self.encode_conditioner_inputs(&item.caption)?;
            eval([&x0, &source, &t5_ids])?;
            cache.push((x0, source, t5_ids));
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("anima trainer: no usable dataset items".into());
        }

        // Every caption is encoded into `cache`; the multi-GB Qwen3 encoder is now dead weight for
        // the rest of the run. Drop it and evict its buffers before the train loop.
        self.text_encoder = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + trainable factors (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(&self.dit, &self.conditioner, cfg);
        let rank = cfg.rank as f32;
        let (adapter, mut params) = {
            let mut host = AnimaAdapterHost {
                dit: &mut self.dit,
                conditioner: &mut self.conditioner,
            };
            match cfg.network_type {
                NetworkType::Lora => {
                    let (targets, params) =
                        build_lora_targets(&mut host, &target_paths, cfg.rank as i32, cfg.seed)?;
                    (TrainAdapter::Lora { targets }, params)
                }
                NetworkType::Lokr => {
                    let (targets, params) = build_lokr_targets(
                        &mut host,
                        &target_paths,
                        cfg.rank as i32,
                        cfg.decompose_factor,
                        cfg.seed,
                    )?;
                    (TrainAdapter::Lokr { targets }, params)
                }
            }
        };
        let alpha = cfg.alpha;
        let mae = {
            let lt = normalize_cfg(&cfg.loss_type);
            lt == "mae" || lt == "l1"
        };

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

        // --- train loop ---
        let mut accumulated: Option<LoraParams> = None;
        let mut update_idx: u32 = 0;
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, source, t5_ids) = &cache[((step - 1) as usize) % cache.len()];
            let sigma = sample_sigma(
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
            let (loss, grads) = compute_loss_grads(
                &mut self.dit,
                &mut self.conditioner,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                source,
                t5_ids,
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
                // The final update can fire with fewer than `accum` grads; divide by the actual
                // in-window count so a short tail step isn't down-scaled.
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
                let ckpt = req
                    .output_dir
                    .join(intermediate_filename(&req.file_name, step));
                save_adapter(&adapter, &params, &target_paths, alpha, rank, cfg, &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before completing a single step: the factors are still the no-op init (LoRA
        // `B = 0` / LoKr zeroed `w1`). Surface the cancellation as the typed `Error::Canceled` rather
        // than writing a valid-looking identity adapter and returning `Ok`.
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(
            &adapter,
            &params,
            &target_paths,
            alpha,
            rank,
            cfg,
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }

    /// Encode a caption to the conditioner's **inputs**: the mask-multiplied Qwen3 `last_hidden_state`
    /// `[1, S, 1024]` (bf16) and the T5 query-token ids `[1, St]` (int32) — the deterministic,
    /// cacheable half of the text path (mirrors `pipeline::encode_prompt` up to the conditioner). The
    /// conditioner itself is run per-step in the grad graph (it is a trained target), so its output is
    /// deliberately NOT cached.
    fn encode_conditioner_inputs(&self, caption: &str) -> Result<(Array, Array)> {
        let te = self.text_encoder.as_ref().ok_or_else(|| {
            mlx_gen::Error::Msg(
                "anima trainer: text encoder already freed (caching after loop)".into(),
            )
        })?;
        let (qwen_ids, qwen_mask) = self.tokenizers.encode_qwen(caption)?;
        let source = te.forward(&qwen_ids, &qwen_mask)?; // [1, S, 1024] bf16
        let mask = qwen_mask.as_dtype(source.dtype())?.expand_dims(2)?; // [1, S, 1]
        let source = multiply(&source, &mask)?;
        let t5_ids = self.tokenizers.encode_t5(caption)?; // [1, St]
        Ok((source, t5_ids))
    }
}

/// Dispatch the save: LoRA is written with the ComfyUI `diffusion_model.` prefix, PEFT `lora_A`/
/// `lora_B` keys, and NO alpha (the α/rank fold baked into `lora_B` — the shipped Anima convention);
/// LoKr uses the shared [`save_lokr`] (bare `lokr_*` keys the sc-10521 LoKr path consumes).
fn save_adapter(
    adapter: &TrainAdapter,
    params: &LoraParams,
    target_paths: &[String],
    alpha: f32,
    rank: f32,
    cfg: &TrainingConfig,
    path: &Path,
) -> Result<()> {
    match adapter {
        TrainAdapter::Lora { .. } => save_anima_lora(params, target_paths, alpha, rank, path),
        TrainAdapter::Lokr { targets } => {
            // Store the Kronecker factors bf16 ([`SAVE_DTYPE`]) too — the inference LoKr loader
            // reconstructs the delta at bf16 ([`LOKR_DTYPE`]) regardless, so casting the f32 master
            // factors here is round-trip-lossless and halves the file, matching the LoRA convention.
            let bf16: LoraParams = params
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.as_dtype(SAVE_DTYPE)?)))
                .collect::<Result<_>>()?;
            save_lokr(&bf16, targets, alpha, rank, cfg.decompose_factor, path)
        }
    }
}

/// Write the trainable LoRA factors in the shipped Anima convention: keys
/// `diffusion_model.{path}.lora_A.weight` `[r,in]` / `diffusion_model.{path}.lora_B.weight` `[out,r]`,
/// with the `alpha/rank` scale **baked into `lora_B`** so the file carries no alpha (α = r ⇒ the
/// sc-10521 inference loader applies it at scale 1.0, exactly reproducing the trained residual). The
/// factors are stored **bf16** ([`SAVE_DTYPE`]) and the metadata is `{"format":"pt"}` only, matching
/// the official `anima-turbo-lora-v0.2` `__metadata__` and dtype.
fn save_anima_lora(
    params: &LoraParams,
    target_paths: &[String],
    alpha: f32,
    rank: f32,
    path: &Path,
) -> Result<()> {
    let scale = Array::from_slice(&[alpha / rank], &[1]);
    // Own the baked `lora_B` arrays so their borrows outlive the entry list.
    let mut owned: Vec<(String, Array)> = Vec::with_capacity(target_paths.len() * 2);
    for p in target_paths {
        let a = params
            .get(format!("{p}.lora_a").as_str())
            .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {p}.lora_a")))?;
        let b = params
            .get(format!("{p}.lora_b").as_str())
            .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {p}.lora_b")))?;
        owned.push((
            format!("{KEY_PREFIX}{p}.lora_A.weight"),
            a.as_dtype(SAVE_DTYPE)?,
        ));
        owned.push((
            format!("{KEY_PREFIX}{p}.lora_B.weight"),
            multiply(b, &scale)?.as_dtype(SAVE_DTYPE)?,
        ));
    }
    let entries: Vec<(String, &Array)> = owned.iter().map(|(k, v)| (k.clone(), v)).collect();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// `{stem}-step{step}{ext}` — the intermediate-checkpoint name for `save_every`.
fn intermediate_filename(file_name: &str, step: u32) -> String {
    let p = Path::new(file_name);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("safetensors");
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("adapter");
    format!("{stem}-step{step}.{ext}")
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

/// Sample a normalised flow-match base timestep `σ ∈ [1e-3, 1−1e-3]` (a faithful port of the
/// SceneWorks `sample_training_timestep`: `sigmoid(randn)` default, `uniform` for linear,
/// `(uniform + sigmoid(randn))/2` weighted; bias `high` → `√σ`, `low` → `σ²`), then run it through
/// the static `shift = 3.0` warp the inference schedule uses. Deterministic in `seed`.
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
    let bias = normalize_cfg(timestep_bias);
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    let shifted = apply_static_shift(t, SIGMA_SHIFT);
    Ok(shifted.clamp(1e-3, 1.0 - 1e-3))
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr) onto BOTH
/// the DiT and the conditioner, run the conditioner (→ `encoder_hidden_states`) then the DiT, regress
/// the velocity `forward()` output toward `noise − x0`, return `(loss, grads)`. The conditioner runs
/// **inside** the traced graph, so its adapter factors receive gradients. `dtype` is the bf16 compute
/// dtype: `x_t` is cast at entry, the LoRA factors are cast inside the traced install, and the DiT/
/// conditioner run bf16; the noising math, loss, and grads stay f32.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &mut CosmosDiT,
    conditioner: &mut AnimaTextConditioner,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    source: &Array,
    t5_ids: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
    let x_t = x_t.as_dtype(dtype)?;
    let src = source.clone();
    let ids = t5_ids.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets (DiT + conditioner) via the combined host, then drop the host so the
        // `&self` forwards can borrow the two sub-models. F-149: NEVER check the cancel flag inside
        // this traced closure (it would be stringified through `Exception::custom` and lose the typed
        // `Error::Canceled`); cancellation is the caller's job at the step boundary.
        {
            let mut host = AnimaAdapterHost {
                dit: &mut *dit,
                conditioner: &mut *conditioner,
            };
            adapter.install_as(&mut host, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        }
        let enc = conditioner
            .forward(&src, &ids, dtype)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let s = Array::from_slice(&[timestep], &[1]);
        let v = dit
            .forward(&x_t, &s, &enc, dtype)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let v = v.as_dtype(Dtype::Float32)?;
        let diff = subtract(&v, &target)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn request(items: usize, steps: u32, rank: u32) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "1girl, silver hair".into(),
                })
                .collect(),
            config: TrainingConfig {
                steps,
                rank,
                ..Default::default()
            },
            output_dir: PathBuf::from("/tmp/anima-trainer-test"),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        }
    }

    #[test]
    fn three_trainer_variants_registered() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            assert!(
                gen_core::registry::trainers().any(|r| (r.descriptor)().id == id),
                "trainer id {id} not registered"
            );
        }
    }

    #[test]
    fn descriptor_advertises_lora_and_lokr() {
        let d = trainer_descriptor_base();
        assert_eq!(d.id, "anima_base");
        assert_eq!(d.family, "anima");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
    }

    #[test]
    fn validate_request_guards() {
        assert!(validate_request(&request(1, 100, 16)).is_ok());
        assert!(validate_request(&request(0, 100, 16)).is_err()); // empty dataset
        assert!(validate_request(&request(1, 0, 16)).is_err()); // zero steps
        assert!(validate_request(&request(1, 100, 0)).is_err()); // zero rank
    }

    #[test]
    fn validate_rejects_unrecognized_schedule_and_loss() {
        let with = |f: fn(&mut TrainingConfig)| {
            let mut r = request(1, 100, 16);
            f(&mut r.config);
            validate_request(&r)
        };
        assert!(with(|c| c.timestep_type = "sgmoid".into()).is_err());
        assert!(with(|c| c.timestep_bias = "hihg_noise".into()).is_err());
        assert!(with(|c| c.loss_type = "huber".into()).is_err());
        assert!(with(|c| c.optimizer = "sophia".into()).is_err());
        // Documented spellings still pass, case/separator-insensitively.
        assert!(with(|c| c.timestep_type = "Linear".into()).is_ok());
        assert!(with(|c| c.timestep_bias = "High-Noise".into()).is_ok());
        assert!(with(|c| c.loss_type = "L1".into()).is_ok());
    }

    #[test]
    fn static_shift_matches_inference_schedule() {
        // shift(σ)=3σ/(1+2σ): shift(1)=1, shift(0)=0, shift(0.5)=1.5/2=0.75.
        assert!((apply_static_shift(1.0, SIGMA_SHIFT) - 1.0).abs() < 1e-6);
        assert!((apply_static_shift(0.0, SIGMA_SHIFT)).abs() < 1e-6);
        assert!((apply_static_shift(0.5, SIGMA_SHIFT) - 0.75).abs() < 1e-6);
        // Monotone increasing on [0,1].
        assert!(apply_static_shift(0.2, SIGMA_SHIFT) < apply_static_shift(0.8, SIGMA_SHIFT));
    }

    #[test]
    fn build_batch_is_flow_match_velocity() {
        // x_t = (1-σ)x0 + σ·noise; target = noise - x0; timestep = σ (raw, not 1-σ).
        let x0 = Array::from_slice(&[2.0f32, 4.0], &[1, 2]);
        let noise = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);
        let (x_t, target, ts) = build_batch(&x0, &noise, 0.25).unwrap();
        assert!((ts - 0.25).abs() < 1e-6);
        // x_t = 0.75*[2,4] + 0.25*[1,1] = [1.75, 3.25]
        assert!((x_t.as_slice::<f32>()[0] - 1.75).abs() < 1e-5);
        assert!((x_t.as_slice::<f32>()[1] - 3.25).abs() < 1e-5);
        // target = [1,1] - [2,4] = [-1,-3]
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0]);
    }

    /// CI-runnable, **weights-free** guard on the trainable-target surface (sc-10522). Builds the DiT +
    /// conditioner *structurally* (placeholder 1×1 weights, no licensed checkpoint, no Metal compute),
    /// then asserts the trainer enumerates exactly **508** targets = **448** DiT (`blocks.*`) + **60**
    /// conditioner (`llm_adapter.blocks.*`). Deliberately **structural** (path strings, no numerics),
    /// so — unlike a Metal golden — it cannot go device-dependently red: a regression that drops the
    /// conditioner leg (the sc-10274 partial-injection class) collapses this to 448 and fails here.
    /// This is the always-on analogue of the real-weights `trainable_surface_is_508_*` (which is
    /// `#[ignore]`d and needs the snapshot); it runs under a plain `cargo test -p mlx-gen-anima`.
    #[test]
    fn trainable_surface_is_508_dit_plus_60_conditioner_weightsfree() {
        use crate::config::{ConditionerConfig, DitConfig};
        let dit = CosmosDiT::structural(DitConfig::anima());
        let cond = AnimaTextConditioner::structural(ConditionerConfig::anima());

        // Empty `lora_target_modules` ⇒ the full 508-target surface the official Anima LoRAs carry.
        let cfg = TrainingConfig::default();
        let paths = resolve_target_paths(&dit, &cond, &cfg);
        assert_eq!(
            paths.len(),
            508,
            "full trainable surface must be 508 (448 DiT + 60 conditioner), got {}",
            paths.len()
        );

        let cond_paths: Vec<&String> = paths
            .iter()
            .filter(|p| p.starts_with("llm_adapter."))
            .collect();
        let dit_paths: Vec<&String> = paths
            .iter()
            .filter(|p| !p.starts_with("llm_adapter."))
            .collect();
        assert_eq!(
            cond_paths.len(),
            60,
            "conditioner (llm_adapter) must contribute exactly 60 targets — dropping it is the \
             sc-10274 partial-injection regression"
        );
        assert_eq!(
            dit_paths.len(),
            448,
            "DiT must contribute exactly 448 targets (28 blocks × 16)"
        );
        assert!(
            cond_paths
                .iter()
                .all(|p| p.starts_with("llm_adapter.blocks.")),
            "every conditioner target must be a per-block llm_adapter path"
        );
        assert!(
            dit_paths.iter().all(|p| p.starts_with("blocks.")),
            "every DiT target must be a per-block path"
        );

        // A non-empty target filter narrows the surface by PEFT-style suffix match, and must stay
        // non-empty and strictly narrower than the full surface.
        let filtered = TrainingConfig {
            lora_target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "output_proj".into(),
            ],
            ..Default::default()
        };
        let attn_only = resolve_target_paths(&dit, &cond, &filtered);
        assert!(
            !attn_only.is_empty() && attn_only.len() < paths.len(),
            "suffix filter must narrow the surface, got {}",
            attn_only.len()
        );
        assert!(
            attn_only
                .iter()
                .all(|p| ["q_proj", "k_proj", "v_proj", "output_proj"]
                    .iter()
                    .any(|s| p.ends_with(&format!(".{s}")))),
            "every filtered path must match a requested suffix"
        );
    }
}
