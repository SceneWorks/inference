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
//! ## Mid-run resume (sc-10642)
//! At each `save_every` the trainer additionally writes a resume bundle (the shared sc-9560 engine in
//! [`mlx_gen::train::checkpoint`]): the optimizer state + the raw trainable factors + `{step,
//! update_idx, optimizer}`. With `cfg.resume`, a fresh run of the same adapter restores that state and
//! continues from `step + 1`. On restore it **asserts the full 448 DiT + 60 `llm_adapter` = 508 target
//! surface** against the live model (never inferred from the file) — a checkpoint that had dropped the
//! 60 conditioner targets would resume an inert conditioner while every structural check still passed
//! (the sc-10522 trap), so `assert_resume_surface_matches` fails it loudly.
//!
//! ## Round-trip
//! The trained adapter round-trips through the sc-10521 inference loader: LoRA is saved with the
//! ComfyUI `diffusion_model.` prefix + PEFT `lora_A`/`lora_B` keys and **no alpha** (the shipped Anima
//! convention — the α/rank fold is baked into `lora_B` so scale-1.0 loading is exact for any alpha);
//! LoKr is saved by the shared [`save_lokr`] in the bare-path `lokr_*` convention the sc-10521 LoKr
//! path consumes. Both reconstruct the residual at **bf16** to match the inference loader.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost};
use mlx_gen::gen_core::{self, BucketSchedule};
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint;
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, save_lokr, LoraParams, TrainAdapter,
};
pub use mlx_gen::train::lora::{LokrTarget, LoraTarget};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxDriver, Parameterization, PerceptualPath, StepPlan,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::taehv::TaehvConfig;
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Precision, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::adapters::AnimaAdapterHost;
use crate::conditioner::AnimaTextConditioner;
use crate::config::{Variant, SIGMA_SHIFT};
use crate::loader::AnimaComponents;
use crate::pipeline::render_preview;
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

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-10641), matching
/// z-image's cap; extra prompts beyond this are ignored (a preview is a quick convergence check, not a
/// full sweep, and each one is an inference denoise on the live 2B DiT).
const SAMPLE_PROMPT_CAP: usize = 4;

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

/// The distinct adapter *target paths* present in a trainable-factor map, split into the DiT
/// (`blocks.*`) surface and the conditioner (`llm_adapter.*`) surface. Every factor is keyed
/// `{path}.<factor>` (`.lora_a`/`.lora_b`/`.lokr_w1`/`.lokr_w2`/`.lokr_w2_a`/`.lokr_w2_b`), so the
/// target path is the key with its trailing factor segment stripped. Used by the sc-10642 resume guard
/// to count the 448 DiT + 60 `llm_adapter` = 508 targets a full-surface run trains.
fn split_target_surface(params: &LoraParams) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut dit = BTreeSet::new();
    let mut cond = BTreeSet::new();
    for key in params.keys() {
        let path = key.rsplit_once('.').map_or(key.as_ref(), |(p, _)| p);
        if path.starts_with("llm_adapter.") {
            cond.insert(path.to_string());
        } else {
            dit.insert(path.to_string());
        }
    }
    (dit, cond)
}

/// The sc-10522 / sc-10642 restore guard. A resumed run rebuilds its trainable surface from the LIVE
/// model (`resolve_target_paths` → `build_lora/lokr_targets`, the full **508** targets = 448 DiT +
/// 60 `llm_adapter` for the default config) and then swaps the checkpoint's factors in. If the
/// checkpoint had dropped the 60 conditioner (`llm_adapter.*`) targets — or otherwise disagreed with
/// this run's surface — resume would silently continue with an **inert conditioner** while every
/// structural check (converging DiT loss, valid adapter file) still passed: the exact sc-10522 trap.
/// So before swapping the factors we ASSERT the checkpoint's factor surface is EXACTLY the one this run
/// rebuilt — same DiT paths, same `llm_adapter` paths, same factor keys — and fail loudly with a typed
/// [`Error`](mlx_gen::Error) otherwise, naming the DiT/conditioner target split so the mismatch is
/// diagnosable. The count is asserted against the live model's surface, never inferred from the file.
fn assert_resume_surface_matches(restored: &LoraParams, expected: &LoraParams) -> Result<()> {
    // Exact factor-key equality is the strongest guarantee — it also catches a LoRA↔LoKr network-type
    // mismatch or a single dropped factor, not just a wholesale missing target class.
    let restored_keys: BTreeSet<&str> = restored.keys().map(|k| k.as_ref()).collect();
    let expected_keys: BTreeSet<&str> = expected.keys().map(|k| k.as_ref()).collect();
    if restored_keys == expected_keys {
        return Ok(());
    }
    let (rd, rc) = split_target_surface(restored);
    let (ed, ec) = split_target_surface(expected);
    Err(mlx_gen::Error::Msg(format!(
        "anima resume: checkpoint trainable surface does not match this run — refusing to resume into a \
         silently-different model (sc-10522). checkpoint carries {rt} targets ({rdn} DiT `blocks.*` + \
         {rcn} `llm_adapter` conditioner); this run rebuilt {et} targets ({edn} DiT + {ecn} \
         conditioner). a checkpoint missing the {ecn} conditioner targets would resume an inert \
         conditioner — start a fresh run or resume the matching adapter.",
        rt = rd.len() + rc.len(),
        rdn = rd.len(),
        rcn = rc.len(),
        et = ed.len() + ec.len(),
        edn = ed.len(),
        ecn = ec.len(),
    )))
}

/// Whether in-training preview sampling (sc-10641) is active for this run: a positive cadence AND at
/// least one prompt AND not already cancelled. `false` ⇒ no conditioner-input pre-encode and no render,
/// so a run that does not opt in (the default) behaves exactly as before.
fn previews_enabled(cfg: &TrainingConfig, cancelled: bool) -> bool {
    cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !cancelled
}

/// Whether micro-step `step` (1-based) lands on the preview cadence. `sample_every == 0` never fires
/// (also guarded upstream by [`previews_enabled`], but kept total here so the predicate is self-contained).
fn preview_due(step: u32, sample_every: u32) -> bool {
    sample_every > 0 && step.is_multiple_of(sample_every)
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
    /// Which Anima variant this trainer wraps — decides whether in-training previews (sc-10641) run CFG
    /// (base/aesthetic) or a single guidance-free forward (turbo).
    variant: Variant,
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
        // LoRA/LoKr only — no control-branch training path (F-006).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets — one cached latent per item per bucket,
        // sampled by `BucketSchedule`; the memory pre-flight sizes for the largest bucket.
        // sc-24828 (epic 2123): honors `subject_mask_loss` on its one (LoRA/LoKr, dense or
        // block-checkpointed) loss path, with a weight map per (item, bucket) entry.
        // sc-24830 (epic 2123): depth anchoring on that same loss path (dense and
        // block-checkpointed) — the shared decoded-x0 perceptual path through TAEW2.1, the
        // Qwen-Image VAE's tiny decoder.
        // sc-24833 (epic 2123): the VAE anchor (same family decoder → FLUX.2 encoder taps, per
        // decoded frame) through the shared aux-loss builder this trainer already drives, wherever
        // depth anchoring is wired. No E-LatentLPIPS: no published weights match this latent
        // family.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
            identity_loss: true,
            face_landmark_loss: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring
            // (decoded-x0 pixel losses through this trainer's x0 decoder).
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            vae_anchor_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
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
    // The weights load lazily (sc-2124): `validate` and `train`'s refusal floors never read them.
    Ok(Box::new(
        mlx_gen::train::lazy::LazyTrainer::new(trainer_descriptor_for(variant), validate_floors, {
            let spec = spec.clone();
            move || load_weights(&spec, variant)
        })
        .validating_on_base_when(mlx_gen::train::lazy::custom_targets),
    ))
}

/// The weight load behind [`load_variant_trainer`], run by [`LazyTrainer`](mlx_gen::train::lazy::LazyTrainer) on first need.
fn load_weights(spec: &LoadSpec, variant: Variant) -> Result<AnimaTrainer> {
    let components = AnimaComponents::load(&spec.weights, variant)?;
    Ok(AnimaTrainer {
        descriptor: trainer_descriptor_for(variant),
        variant,
        tokenizers: components.tokenizers,
        text_encoder: Some(components.text_encoder),
        vae: components.vae,
        dit: components.dit,
        conditioner: components.conditioner,
    })
}

// Explicit trainer registration constants for all three variants.
mlx_gen::register_trainer! {
    pub(crate) const BASE_TRAINER_REGISTRATION = trainer_descriptor_base => load_trainer_base
}
mlx_gen::register_trainer! {
    pub(crate) const AESTHETIC_TRAINER_REGISTRATION =
        trainer_descriptor_aesthetic => load_trainer_aesthetic
}
mlx_gen::register_trainer! {
    pub(crate) const TURBO_TRAINER_REGISTRATION = trainer_descriptor_turbo => load_trainer_turbo
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

impl Trainer for AnimaTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_floors(self.descriptor(), req)?;
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
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
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
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are
        // off). The memory pre-flight and previews size for the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let preview_edge = edges.iter().copied().max().unwrap_or(0);

        // Anima's base is bf16 on disk (there is no dense-f32 cast path), so training runs bf16
        // mixed-precision: the frozen base + activation stream are bf16, and the trainable factors /
        // loss / grads / optimizer stay f32 (master-weights). This matches the inference dtype.
        let compute_dtype = Dtype::Bfloat16;

        // sc-10576 — fail-fast pre-flight memory guard. The dense (non-block-checkpointed) first step
        // materializes the whole DiT forward graph — at 1536² the retained per-block seq² self-attention
        // (≈9216 image tokens) makes that working set exceed unified memory, and the OS hard-kills the
        // worker with an UNCATCHABLE SIGKILL (the run just appears to hang at the last cached latent).
        // We predict it and refuse up front with an actionable, catchable error — BEFORE the (minutes-
        // long) latent caching — whenever gradient checkpointing is NOT enabled (LoRA checkpointing OR
        // the LoKr/dense fallback). With whole-block checkpointing on, the first step fits, so skip it.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        // Epic 2123 E7 (sc-24830): the aux-loss models (TAEW2.1 + every enabled arm) count
        // against the budget on BOTH paths — a checkpointed depth job must not skip admission.
        let aux_gb = perceptual_footprint_gb(cfg, preview_edge, req.items.len() * edges.len());
        if !will_checkpoint || aux_gb > 0.0 {
            preflight_memory_guard(
                cfg,
                &edges,
                compute_dtype == Dtype::Bfloat16,
                aux_gb,
                will_checkpoint,
            )?;
        }

        // Epic 2123 depth anchoring (sc-24830): load the frozen TAEW2.1 decoder +
        // Depth-Anything-V2 before the caching pass, so a missing checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → cache: VAE latents + (masked Qwen3 states, T5 ids) into memory ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        // (x0 latent, masked Qwen3 source_hidden, T5 query-token ids, and — subject-masked loss,
        // sc-24828 — that bucket's latent loss-weight map, `None` when the technique is off),
        // item-major: `cache[item * edges.len() + bucket]` (sc-2127). The conditioner inputs are
        // encoded once per item and shared (refcounted) by every bucket entry.
        let mut cache: Vec<(Array, Array, Array, Option<Array>)> =
            Vec::with_capacity(req.items.len() * edges.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            // sc-24828: the item's subject mask is read + checked once, resampled per bucket.
            let mask = PreparedSubjectMask::load_if_enabled(
                "anima trainer",
                item,
                cfg.subject_mask_loss.as_ref(),
            )?;
            let (source, t5_ids) = self.encode_conditioner_inputs(&item.caption)?;
            eval([&source, &t5_ids])?;
            for (x0, mask_weight) in encode_buckets(&edges, mask.as_ref(), |edge| {
                let nchw = mlx_gen_qwen_image::preprocess_init_image(&img, edge, edge)?; // [1,3,edge,edge]
                self.vae.encode(&nchw) // [1,16,1,edge/8,edge/8], normalized
            })? {
                cache.push((x0, source.clone(), t5_ids.clone(), mask_weight));
            }
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("anima trainer: no usable dataset items".into());
        }

        // sc-10641 — pre-encode the preview-sample prompts' conditioner INPUTS while the Qwen3 encoder
        // is still resident (it is freed just below). We cache the conditioner INPUTS (masked Qwen3
        // states + T5 ids), NOT its output: the conditioner (`llm_adapter`) is a trained target, so each
        // preview must re-run it through the live graph to reflect its adapters (the sc-10522 trap — a
        // cached conditioner output would render silently-inert conditioner adapters). For CFG variants
        // (base/aesthetic) the empty-prompt uncond inputs are cached once too. Skipped when sampling is
        // off (the default) or the run is already cancelled.
        let previews_on = previews_enabled(cfg, req.cancel.is_cancelled());
        let sample_inputs: Vec<(String, Array, Array)> = if previews_on {
            let mut v = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let (source, t5_ids) = self.encode_conditioner_inputs(prompt)?;
                eval([&source, &t5_ids])?;
                v.push((prompt.clone(), source, t5_ids));
            }
            v
        } else {
            Vec::new()
        };
        // Uncond (empty-prompt) conditioner inputs — cached once, only for CFG variants at guidance ≠ 1.
        let uncond_inputs: Option<(Array, Array)> =
            if !sample_inputs.is_empty() && self.variant.uses_cfg() {
                let (s, ids) = self.encode_conditioner_inputs("")?;
                eval([&s, &ids])?;
                Some((s, ids))
            } else {
                None
            };

        // Every caption is encoded into `cache`; the multi-GB Qwen3 encoder is now dead weight for
        // the rest of the run. Drop it and evict its buffers before the train loop.
        self.text_encoder = None;
        mlx_rs::memory::clear_cache();

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference (TAEW2.1 decode of its
        // cached clean latent → DA2 depth) is computed exactly once per job, by the `AuxDriver`
        // before the loop — after the text encoder is freed, so the decoder + DA2 never share
        // residency with it.
        if let Some(path) = perceptual.as_mut() {
            // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
            // cropped like the image and resampled onto its decoded size.
            path.attach_subject_masks(mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
                "anima trainer",
                &req.items,
                cfg,
                edges.len(),
                CropBox::center_square,
            )?);
        }

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

        // sc-10576 — gradient checkpointing. Collect, per DiT block, the adapter-routable LOCAL paths
        // trained on it (`self_attn.q_proj`, `adaln_modulation_mlp.2`, …); the 28-block DiT stack is
        // where the first-step activation memory concentrates (its per-block seq² self-attention), so
        // that is what we whole-block checkpoint. The 60 conditioner (`llm_adapter.*`) targets are NOT
        // collected here — the conditioner runs UN-checkpointed inside the traced grad graph (only 512
        // text tokens), so its factors train through ordinary autograd, and — critically — its gradient
        // path stays live because the DiT checkpoints thread `encoder` as an explicit input (sc-10522).
        let n_layers = self.dit.config().num_layers;
        let block_local_targets = collect_dit_block_local_targets(&target_paths, n_layers);

        // Gradient checkpointing is an OPT-IN option, never auto-forced — a run that would OOM is caught
        // by the fail-fast pre-flight guard above (which recommends this flag) rather than silently
        // changing the user's training dynamics. Only the LoRA path is whole-block checkpointed today;
        // LoKr (a distinct Kronecker reconstruction) falls back to the dense path, exactly like z-image.
        let is_lora = matches!(adapter, TrainAdapter::Lora { .. });
        let use_checkpoint = is_lora && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing: the conditioner is never whole-block checkpointed, so it keeps
        // segment ckpt ON (bounds its retained attention). The DiT keeps segment ckpt ON only when
        // whole-block checkpointing is OFF (the dense / LoKr path) — when whole-block is on, the block
        // recompute already covers attention and nesting would recompute it twice for no memory win.
        self.dit.set_sdpa_checkpoint(!use_checkpoint);
        self.conditioner.set_sdpa_checkpoint(true);

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
            .unwrap_or("adapter")
            .to_string();

        // sc-10642 — mid-run resume (the shared sc-9560 / F-125 engine). When `cfg.resume` is set and a
        // prior interrupted run of THIS adapter left a snapshot in `output_dir`, restore its optimizer
        // state + trainable factors + step/update index and continue from `start_step + 1` rather than
        // step 0. The restored factors REPLACE the fresh `B = 0`/zeroed init built above; the fresh build
        // still ran first so `params` carries the live model's full expected surface for the sc-10522
        // guard. Snapshots land on optimizer-update boundaries at each `save_every`, so resume is
        // bit-exact for `gradient_accumulation = 1` (the default) and when `save_every` is a multiple of
        // the accumulation otherwise.
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                // sc-10522: the checkpoint MUST carry the full 448 DiT + 60 `llm_adapter` surface this
                // run rebuilt, or the conditioner resumes inert. Assert against the live model's surface
                // BEFORE swapping the factors in — a typed error, not a silent inert-conditioner resume.
                assert_resume_surface_matches(&loaded, &params)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[sc-10642] anima resuming '{stem}' from step {start_step} (optimizer update \
                     {update_idx})"
                );
            }
        }

        // --- train loop ---
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: per-image, per-update alternation keys for the perceptual losses, keyed on
        // the real dataset item. A resumed run replays the skipped prefix so the phase matches.
        let mut aux_driver = match perceptual {
            Some(path) => Some(aux_driver(path, &cache, &schedule, accum, start_step)?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let entry = step_cache_index(&schedule, step);
            let (x0, source, t5_ids, mask_weight) = &cache[entry];
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
            // Epic 2123 E8: plan the step's loss terms; an aux-only step trains at the (shifted) σ
            // remapped into the loss window.
            let planned = match aux_driver.as_mut() {
                Some(d) => d.sample(step, &schedule).plan(sigma)?,
                None => None,
            };
            if let Some(p) = &planned {
                sigma = p.plan.noise_level;
            }
            let aux = planned.as_ref().map(|p| AuxStep {
                path: p.path,
                plan: &p.plan,
                image: p.entry,
            });
            let (losses, grads) = compute_step_loss_grads(
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
                mask_weight.as_ref(),
                checkpoint_blocks,
                compute_dtype,
                aux,
            )?;
            last_loss = losses.total;
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
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx, cfg.seed)?;
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
                // sc-10642 — alongside the user-facing adapter checkpoint, write the resume bundle
                // (optimizer state + raw trainable factors + `{step, update_idx, optimizer}`), so an
                // interrupted run can continue from here with `cfg.resume` rather than restarting at 0.
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-10641 — periodic preview samples from the in-progress adapter so the user can watch the
            // LoRA converge (the sc-5637 `TrainingProgress::Sample` contract). Install the CURRENT factors
            // as concrete adapters for the forward-only render (the traced `loss_fn` re-installs them at
            // the next step, so no teardown is needed), then render each cached sample prompt. The render
            // re-runs the conditioner through the LIVE graph (`render_preview` → `conditioner.forward`),
            // so the preview reflects the 60 `llm_adapter` conditioner adapters' training — never a cached
            // output (sc-10522). Best-effort: a render failure logs and continues the (long) run.
            if previews_on && preview_due(step, cfg.sample_every) {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                {
                    let mut host = AnimaAdapterHost {
                        dit: &mut self.dit,
                        conditioner: &mut self.conditioner,
                    };
                    adapter.install_as(&mut host, &params, alpha, rank, lora_dtype, LOKR_DTYPE)?;
                }
                let guidance = if self.variant.uses_cfg() {
                    cfg.sample_guidance_scale
                } else {
                    1.0 // turbo is the merged CFG-free student — a single forward, guidance inert.
                };
                let total = sample_inputs.len() as u32;
                for (i, (prompt, source, t5_ids)) in sample_inputs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match render_preview(
                        &self.dit,
                        &self.conditioner,
                        &self.vae,
                        source,
                        t5_ids,
                        uncond_inputs.as_ref(),
                        cfg.sample_steps.max(1) as usize,
                        guidance,
                        preview_edge,
                        sample_seed,
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
                        // F-117: a cancelled preview denoise exits the preview loop (the outer loop's
                        // cancel check then unwinds the run); other failures skip one preview.
                        Err(mlx_gen::Error::Canceled) => break,
                        Err(e) => eprintln!(
                            "[sc-10641] anima preview sample failed at step {step} (prompt {}): {e} \
                             — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
                // sc-5567 — release the preview's transient forward+VAE-decode residency before
                // training resumes. MLX pools freed buffers, so a full `sample_steps` denoise plus
                // VAE decode on the 2B DiT can push peak residency above the train steady state and
                // SIGKILL a tightly-budgeted run (sc-10576 memory focus). Preview path only — the
                // training-step working set is untouched.
                mlx_rs::memory::clear_cache();
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

/// The cache entry the 1-based training `step` reads (sc-2127). One bucket ⇒ `(step - 1) % items`,
/// the pre-bucket round-robin.
fn step_cache_index(schedule: &BucketSchedule, step: u32) -> usize {
    schedule.cache_index((step - 1) as usize)
}

/// The per-step loss breakdown [`compute_step_loss_grads`] returns (epic 2123 E8).
#[derive(Clone, Copy, Debug, PartialEq)]
struct StepLosses {
    /// The differentiated step loss.
    total: f32,
    /// The diffusion (velocity-regression) term, `None` on an aux-only step.
    diffusion: Option<f32>,
    /// The weighted aux-loss term, `None` when no aux loss contributed this step.
    aux: Option<f32>,
}

/// One aux-loss step's view of the trainer's [`PerceptualPath`].
struct AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: &'a StepPlan,
    /// The step's (item, bucket) cache entry (selects its cached reference).
    image: usize,
}

/// Anima's latent family for the shared aux-loss builder (epic 2123 E8): the Qwen-Image VAE
/// (`mlx_gen_qwen_image::QwenVae`, whose encode applies the per-channel `latents_mean` /
/// `latents_std` — the space the DiT predicts in), decoded by TAEW2.1, the TAEHV checkpoint
/// upstream lists for Qwen-Image, which takes that normalized latent with no scale/shift. A still
/// image is one `T = 1` clip.
fn anima_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Taehv {
        name: "TAEW2.1",
        config: TaehvConfig::taew2_1(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder: `None` when no aux loss is
/// enabled (nothing loads; every step is the plain diffusion step).
fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label: "anima trainer",
            decoder: anima_decoder(),
            latent_lpips: None,
        },
    )
}

/// The cached `[B, C, T, h, w]` latent (`T = 1` for a still) as the decoder's NCHW batch
/// `[B·T, C, h, w]` — each latent frame decoded independently.
fn latent_frames_nchw(latent: &Array) -> Result<Array> {
    let s = latent.shape();
    if s.len() != 5 {
        return Err(mlx_gen::Error::Msg(format!(
            "anima trainer: expected a [B, C, T, h, w] latent, got {s:?}"
        )));
    }
    Ok(latent
        .transpose_axes(&[0, 2, 1, 3, 4])?
        .reshape(&[s[0] * s[2], s[1], s[3], s[4]])?)
}

/// The loop's [`AuxDriver`] (epic 2123 E8): every (item, bucket) cache entry's perceptual
/// reference computed once, from its clean latent; the alternation keyed on the schedule's REAL
/// items with `accum` micro-steps per update, and a resumed prefix `1..=start_step` replayed.
fn aux_driver(
    path: PerceptualPath,
    cache: &[(Array, Array, Array, Option<Array>)],
    schedule: &BucketSchedule,
    accum: u32,
    start_step: u32,
) -> Result<AuxDriver> {
    AuxDriver::prepare(
        path,
        cache.len(),
        |i| latent_frames_nchw(&cache[i].0),
        schedule,
        accum,
        start_step,
    )
}

/// Extra training memory (GB) the enabled perceptual losses add at the largest bucket `edge`
/// (epic 2123 E7): TAEW2.1 + the losses' frozen models plus `entries` cached references. `0` when
/// nothing is enabled.
fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &anima_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// Image tokens the DiT self-attends at a square training `edge`: VAE /8 then patch /2 ⇒ `edge/16`
/// per side, squared. The `+512` is the conditioner's fixed padded text length (cross-attended, not
/// self-attended) — folded in so `s` is the "total token" proxy the projection is fit against.
fn unified_tokens(edge: u32) -> f64 {
    let per_side = (edge as f64 / 16.0).ceil();
    per_side * per_side + 512.0
}

/// Projected DENSE first-step peak GPU memory, in GB, as a function of the token proxy `s`
/// ([`unified_tokens`]) — an empirical fit to peaks measured on the 128 GB target with the Anima base.
///
/// Structure follows the z-image / sc-4874 decomposition `weights + linear·s + quad·s²`: the constant
/// is the resident base (the ~2B-param bf16 Cosmos DiT + the small conditioner + Qwen-Image VAE, after
/// the Qwen3 text encoder is freed post-caching), the linear term is the per-token hidden-state
/// activations retained across the 28 blocks (+ the s·512 cross-attention), and the quadratic term is
/// the residual seq² self-attention. Since the DENSE training path runs with SDPA-segment
/// checkpointing ON ([`AnimaTrainer::train_impl`] sets `dit.set_sdpa_checkpoint(!use_checkpoint)`), the
/// quadratic term is demoted from "one retained `[16-heads, s, s]` matrix per block" to a single
/// layer's backward transient. bf16 roughly halves the weights + activation terms vs an f32 base
/// (Anima has no f32 base, but the parameter is kept for symmetry with the z-image guard).
///
/// **Calibrated** against `first_step_dense_peak_sweep` (128 GB Mac, rank 16, batch 1) — see that
/// `#[ignore]`d test and the `preflight_tests` fit check; refit both if it prints materially different
/// numbers.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        ANIMA_PEAK_CONST_BF16 + ANIMA_PEAK_LINEAR_BF16 * s + ANIMA_PEAK_QUAD_BF16 * s * s
    } else {
        // No f32 Anima base exists; a ~1.8× scale of the bf16 fit is a conservative upper bound.
        1.8 * (ANIMA_PEAK_CONST_BF16 + ANIMA_PEAK_LINEAR_BF16 * s + ANIMA_PEAK_QUAD_BF16 * s * s)
    }
}

// sc-10576 memory-projection coefficients — an exact `a + b·s + c·s²` fit to THREE measured Anima
// dense first-step peaks (edge 512/768/1024 ⇒ s 1536/2816/4608 ⇒ 21.2/32.0/49.4 GB, bf16, rank16) on
// the 128 GB target (`first_step_dense_peak_sweep`). The linear term dominates (per-token activations
// retained across the 28 blocks); the quadratic is small because the dense path runs SDPA-segment
// checkpointing. Extrapolates to ~113 GB at edge 1536 (s 9728) — over this machine's working set, so
// the guard refuses a dense 1536² run. CALIBRATED — re-run the sweep + `preflight_tests` fit check if
// the model/activation shape changes; do not hand-edit.
const ANIMA_PEAK_CONST_BF16: f64 = 9.84;
const ANIMA_PEAK_LINEAR_BF16: f64 = 6.775e-3;
const ANIMA_PEAK_QUAD_BF16: f64 = 3.946e-7;

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. The budget is MLX's own reported
/// memory limit (≈ the device's recommended working set), scaled by 0.85 for worker/host headroom —
/// exceeding it is the regime where the dense run dies. Only consulted when whole-block gradient
/// checkpointing is OFF.
///
/// `edges` are the run's resolution-bucket training edges (sc-2127); the guard sizes for the largest,
/// since any step may sample it (epic 2123 E7). The extra per-bucket latents are not modelled — a
/// 1024² latent is ~1 MB, a rounding error next to the dense first-step working set.
///
/// Epic 2123 E7 (sc-24830): `extra_gb` is the training-time auxiliary models' footprint
/// ([`perceptual_footprint_gb`]); with it the guard also runs with gradient checkpointing on
/// (`checkpointed`), stacking on the resident base ([`checkpointed_baseline_gb`]).
fn preflight_memory_guard(
    cfg: &TrainingConfig,
    edges: &[u32],
    bf16: bool,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    check_budget_with(cfg, edges, bf16, budget_gb, extra_gb, checkpointed)
}

/// The checkpointed baseline the auxiliary-model guard stacks on: the resident-base term of
/// [`projected_dense_peak_gb`] (block checkpointing removes most of the activation terms; the base
/// stays). A lower bound.
fn checkpointed_baseline_gb(bf16: bool) -> f64 {
    projected_dense_peak_gb(0.0, bf16)
}

/// The guard with the auxiliary models' `extra_gb` on top of the DiT projection — the dense
/// projection ([`check_dense_budget_extra`]), or the checkpointed baseline when `checkpointed`
/// (refused through the shared [`mlx_gen_perceptual::check_aux_memory`], naming `cfg`'s enabled
/// aux losses).
fn check_budget_with(
    cfg: &TrainingConfig,
    edges: &[u32],
    bf16: bool,
    budget_gb: f64,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    if !checkpointed {
        return check_dense_budget_extra(edges, bf16, budget_gb, extra_gb);
    }
    let edge = edges.iter().copied().max().unwrap_or(0);
    let projected = checkpointed_baseline_gb(bf16) + extra_gb;
    let safe = budget_gb * 0.85;
    mlx_gen_perceptual::check_aux_memory(
        "anima trainer",
        cfg,
        &format!("a checkpointed training step at resolution {edge}"),
        extra_gb,
        projected,
        safe,
        &format!("{budget_gb:.0} GB MLX limit × 0.85"),
    )
}

/// The pure verdict behind [`preflight_memory_guard`] for an explicit MLX `budget_gb`.
#[cfg(test)]
fn check_dense_budget(edges: &[u32], bf16: bool, budget_gb: f64) -> Result<()> {
    check_dense_budget_extra(edges, bf16, budget_gb, 0.0)
}

/// The dense verdict with `extra_gb` (epic 2123 E7 auxiliary models) on top of the projection.
fn check_dense_budget_extra(
    edges: &[u32],
    bf16: bool,
    budget_gb: f64,
    extra_gb: f64,
) -> Result<()> {
    let edge = edges.iter().copied().max().unwrap_or(0);
    let s = unified_tokens(edge);
    let projected = projected_dense_peak_gb(s, bf16) + extra_gb;
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "anima trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the DiT forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
             would hard-kill the worker (SIGKILL) at the first step with no recoverable error \
             (sc-10576). Enable Gradient Checkpointing (recomputes block activations in the backward) \
             or reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Per-DiT-block LOCAL LoRA target paths (`block_local_targets[i]` for block `i`), extracted from the
/// combined `target_paths`: keep only the DiT `blocks.{i}.{local}` entries (the conditioner's
/// `llm_adapter.blocks.…` entries are deliberately excluded — the conditioner is never whole-block
/// checkpointed). Mirrors z-image's `main_block_local_targets` collection; the order per block matches
/// the params keys `blocks.{i}.{local}.lora_a` that [`build_lora_targets`] produced.
fn collect_dit_block_local_targets(target_paths: &[String], n_layers: usize) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = vec![Vec::new(); n_layers];
    for path in target_paths {
        if path.starts_with("llm_adapter.") {
            continue; // conditioner target — trains through ordinary autograd, not checkpointed
        }
        if let Some((idx, local)) = path.strip_prefix("blocks.").and_then(|r| r.split_once('.')) {
            if let Ok(i) = idx.parse::<usize>() {
                if i < n_layers {
                    out[i].push(local.to_string());
                }
            }
        }
    }
    out
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr) onto BOTH
/// the DiT and the conditioner, run the conditioner (→ `encoder_hidden_states`) then the DiT, regress
/// the velocity `forward()` output toward `noise − x0`, return `(loss, grads)`. The conditioner runs
/// **inside** the traced graph, so its adapter factors receive gradients. `dtype` is the bf16 compute
/// dtype: `x_t` is cast at entry, the LoRA factors are cast inside the traced install, and the DiT/
/// conditioner run bf16; the noising math, loss, and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-DiT-block LOCAL LoRA target paths and switches the DiT
/// forward to the gradient-checkpointed path (sc-10576) — each block recomputes its activations in the
/// backward instead of retaining them, and threads `encoder` as an explicit checkpoint input so the
/// conditioner keeps its gradient. `None` runs the dense (activation-retaining) DiT forward.
#[cfg(test)]
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
    mask_weight: Option<&Array>,
    checkpoint_blocks: Option<&[Vec<String>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        dit,
        conditioner,
        params,
        adapter,
        alpha,
        rank,
        x0,
        source,
        t5_ids,
        sigma,
        noise,
        mae,
        mask_weight,
        checkpoint_blocks,
        dtype,
        None,
    )?;
    Ok((losses.total, grads))
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed and the loss is the weighted perceptual term on the DiT's x0
/// estimate `x_t − σ·v` (the velocity regresses `noise − x0`, σ the shift-warped level), decoded
/// per latent frame by TAEW2.1; summed (`every_n == 1`) aux terms ride along on a diffusion step.
/// The dense and the block-checkpointed DiT forwards land in the same closure, so both carry it.
/// With `aux = None` (or a plan with no aux loss) the traced graph is exactly the pre-epic-2123
/// one.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
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
    mask_weight: Option<&Array>,
    checkpoint_blocks: Option<&[Vec<String>]>,
    dtype: Dtype,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
    let x_t_f32 = x_t.clone();
    let mask_weight = mask_weight.cloned();
    let x_t = x_t.as_dtype(dtype)?;
    let src = source.clone();
    let ids = t5_ids.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets (DiT + conditioner) via the combined host, then drop the host so the
        // `&self` forwards can borrow the two sub-models. On the checkpointed path the DiT block
        // adapters installed here are simply REPLACED inside each checkpoint segment by the explicit-
        // input factors (so they cost nothing there); the conditioner adapters always train through
        // this install (ordinary autograd). F-149: NEVER check the cancel flag inside this traced
        // closure (it would be stringified through `Exception::custom` and lose the typed
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
        let v = match checkpoint_blocks {
            Some(blocks) => dit
                .forward_with_main_checkpointed(&x_t, &s, &enc, dtype, &p, blocks, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => dit
                .forward(&x_t, &s, &enc, dtype)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let v = v.as_dtype(Dtype::Float32)?;
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
                // x0 estimate from the velocity (`noise − x0`): x0 = x_t − σ·v.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma }
                    .recover_x0(&x_t_f32, &v)
                    .and_then(|x| latent_frames_nchw(&x))
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

/// sc-2127 × sc-24828: one item's clean latent per bucket edge (`encode(edge)`, item-major
/// order), each paired with its subject-mask loss weight — the item's already-loaded mask cropped
/// with the center square `center_crop_square` cuts, area-averaged onto THAT bucket's latent grid
/// and laid out like that latent. `None` weights when masked loss is off.
fn encode_buckets(
    edges: &[u32],
    mask: Option<&PreparedSubjectMask>,
    mut encode: impl FnMut(u32) -> Result<Array>,
) -> Result<Vec<(Array, Option<Array>)>> {
    edges
        .iter()
        .map(|&edge| {
            let x0 = encode(edge)?;
            let mask_weight = prepared_subject_mask_weight(
                "anima trainer",
                mask,
                CropBox::center_square,
                x0.shape(),
            )?;
            eval([&x0])?;
            Ok((x0, mask_weight))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    /// sc-24828 × sc-2127: with mask loss on and two buckets, each bucket's weight map has THAT
    /// bucket's latent shape, and the background (right half of the center-square crop, with
    /// `background_weight` 0) is zero at both grids. The 48×32 image's center square is
    /// x ∈ [8, 40); the subject is x < 24 — the crop's left half.
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
        let mask_cfg = gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask =
            mlx_gen::train::subject_mask::PreparedSubjectMask::load("t", &item, &mask_cfg).unwrap();
        let entries = encode_buckets(&[32, 48], Some(&mask), |edge| {
            let g = (edge / 8) as i32;
            Ok(Array::zeros::<f32>(&[1, 16, 1, g, g])?)
        })
        .unwrap();
        assert_eq!(entries.len(), 2);
        for ((x0, w), g) in entries.iter().zip([4usize, 6]) {
            let w = w.as_ref().expect("mask loss on ⇒ a weight map");
            assert_eq!(
                w.shape(),
                x0.shape(),
                "weight must match its own bucket's latent"
            );
            let dense = mlx_rs::ops::multiply(w, Array::ones::<f32>(w.shape()).unwrap()).unwrap();
            let v = dense.as_slice::<f32>();
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

    fn request(items: usize, steps: u32, rank: u32) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "1girl, silver hair".into(),
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
                crate::provider_registry()
                    .unwrap()
                    .trainers()
                    .copied()
                    .any(|r| (r.descriptor)().id == id),
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
        // sc-2127 / sc-24828: every variant honors multi-resolution buckets and subject-masked
        // loss on its one loss path.
        for d in [
            trainer_descriptor_base(),
            trainer_descriptor_aesthetic(),
            trainer_descriptor_turbo(),
        ] {
            assert!(d.techniques.resolution_buckets, "{}", d.id);
            assert!(d.techniques.subject_mask_loss, "{}", d.id);
        }
    }

    fn rb(resolution: u32, repeats: u32) -> gen_core::ResolutionBucket {
        gen_core::ResolutionBucket {
            resolution,
            repeats,
        }
    }

    /// sc-2127: with buckets off the step → cache index is exactly the pre-bucket
    /// `(step - 1) % items`; with buckets [512×16, 768×4, 1024×1] every epoch visits each item 16:4:1.
    #[test]
    fn step_cache_index_matches_legacy_round_robin_and_mixes_buckets() {
        let off = TrainingConfig {
            resolution: 1024,
            ..Default::default()
        };
        let one = BucketSchedule::new(4, &off.training_buckets(), 3);
        for step in 1..=200u32 {
            assert_eq!(step_cache_index(&one, step), ((step - 1) as usize) % 4);
        }
        let on = TrainingConfig {
            resolution_buckets: vec![rb(512, 16), rb(768, 4), rb(1024, 1)],
            ..Default::default()
        };
        let sched = BucketSchedule::new(2, &on.training_buckets(), 3);
        let epoch = sched.epoch_len();
        assert_eq!(epoch, 2 * 21);
        let mut counts = [[0u32; 3]; 2];
        for step in 1..=epoch as u32 {
            let idx = step_cache_index(&sched, step);
            counts[idx / 3][idx % 3] += 1;
        }
        assert_eq!(counts, [[16, 4, 1]; 2]);
    }

    /// sc-2127 / epic 2123 E7: the pre-flight sizes for the LARGEST bucket edge — a [512, 1536] run
    /// is refused under a budget that admits a 512-only run, exactly like a 1536-only run.
    #[test]
    fn preflight_guard_sizes_for_the_largest_bucket() {
        // 45 GB budget → safe ~38 GB: 512² (~21 GB projected) fits, 1536² (~113 GB) does not.
        let small = check_dense_budget(&[512], true, 45.0);
        let mixed = check_dense_budget(&[512, 1536], true, 45.0);
        let mixed_rev = check_dense_budget(&[1536, 512], true, 45.0);
        assert!(small.is_ok(), "512-only must pass: {small:?}");
        let err = mixed.unwrap_err().to_string();
        assert!(err.contains("1536"), "must name the largest edge: {err}");
        assert!(mixed_rev.is_err(), "bucket order must not matter");
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

    // ======================================================================================
    // sc-10576 — gradient checkpointing + pre-flight OOM guard
    // ======================================================================================

    use crate::config::{ConditionerConfig, DitConfig};

    /// A tiny but structurally-complete DiT config (2 heads × 8 = hidden 16, 2 blocks) for a
    /// Metal-cheap grad-parity model. `text_embed_dim` MUST equal the conditioner `target_dim` (the
    /// DiT cross-attends the conditioner output).
    fn tiny_dit_cfg() -> DitConfig {
        DitConfig {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 8,
            num_layers: 2,
            mlp_ratio: 2.0,
            text_embed_dim: 16,
            adaln_lora_dim: 8,
            max_size: (4, 16, 16),
            patch_size: (1, 2, 2),
            rope_scale: (1.0, 4.0, 4.0),
            concat_padding_mask: true,
        }
    }

    fn tiny_cond_cfg() -> ConditionerConfig {
        ConditionerConfig {
            source_dim: 16,
            target_dim: 16,
            model_dim: 16,
            num_layers: 2,
            num_attention_heads: 2,
            mlp_ratio: 2.0,
            target_vocab_size: 32,
            min_sequence_length: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }

    /// `(x0 latent, source Qwen states, T5 ids, noise)` for the tiny model, all f32 (the parity test
    /// runs f32 so the fp tolerance isn't loosened by bf16 rounding).
    fn tiny_inputs(
        dcfg: &DitConfig,
        ccfg: &ConditionerConfig,
        edge: i32,
    ) -> (Array, Array, Array, Array) {
        let hl = edge / 8;
        let x0 = random::normal::<f32>(
            &[1, dcfg.in_channels as i32, 1, hl, hl],
            None,
            None,
            Some(&random::key(11).unwrap()),
        )
        .unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(12).unwrap())).unwrap();
        let source = random::normal::<f32>(
            &[1, 6, ccfg.source_dim as i32],
            None,
            None,
            Some(&random::key(13).unwrap()),
        )
        .unwrap();
        let ids_f = random::uniform::<_, f32>(
            0.0,
            ccfg.target_vocab_size as f32,
            &[1, 4],
            Some(&random::key(14).unwrap()),
        )
        .unwrap();
        let t5_ids = ids_f.as_dtype(Dtype::Int32).unwrap();
        eval([&x0, &noise, &source, &t5_ids]).unwrap();
        (x0, source, t5_ids, noise)
    }

    /// Σ|grad| over the conditioner (`llm_adapter.*`) `lora_b` factors — non-zero iff the encoder
    /// gradient path is live (lora_a starts at B=0, so its grad is 0; lora_b carries the signal).
    fn cond_lora_b_grad(g: &LoraParams) -> f32 {
        g.iter()
            .filter(|(k, _)| k.starts_with("llm_adapter.") && k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .sum()
    }

    /// Σ|grad| over the DiT (`blocks.*`) `lora_b` factors.
    fn dit_lora_b_grad(g: &LoraParams) -> f32 {
        g.iter()
            .filter(|(k, _)| !k.starts_with("llm_adapter.") && k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .sum()
    }

    /// Max relative grad diff between two param maps (per key: `‖a−b‖∞ / max(‖a‖∞, 1e-6)`).
    fn max_rel_diff(ga: &LoraParams, gb: &LoraParams) -> f32 {
        let mut m = 0f32;
        for (k, a) in ga {
            let b = gb.get(k).expect("same keys");
            let num = a
                .subtract(b)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            m = m.max(num / den);
        }
        m
    }

    /// Build a tiny synthetic (DiT + conditioner) and the combined LoRA factor surface on it.
    #[allow(clippy::type_complexity)]
    fn tiny_model_and_adapter() -> (
        CosmosDiT,
        AnimaTextConditioner,
        LoraParams,
        TrainAdapter,
        Vec<String>,
        Vec<Vec<String>>,
    ) {
        let dcfg = tiny_dit_cfg();
        let ccfg = tiny_cond_cfg();
        let mut dit = CosmosDiT::synthetic(dcfg, 42);
        let mut cond = AnimaTextConditioner::synthetic(ccfg, 43);
        let tcfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cond, &tcfg);
        let (targets, params) = {
            let mut host = AnimaAdapterHost {
                dit: &mut dit,
                conditioner: &mut cond,
            };
            build_lora_targets(&mut host, &target_paths, 4, 7).unwrap()
        };
        let blocks = collect_dit_block_local_targets(&target_paths, dcfg.num_layers);
        (
            dit,
            cond,
            params,
            TrainAdapter::Lora { targets },
            target_paths,
            blocks,
        )
    }

    /// Grad-parity: whole-block checkpointed grads == dense grads for BOTH the DiT and the conditioner
    /// (`llm_adapter`) factors, to fp tolerance. Synthetic small model, Metal, no real weights — the
    /// always-on analogue of z-image's `#[ignore]`d `checkpointed_grads_match_dense`. Also asserts the
    /// conditioner actually receives gradient (the encoder path is live) in both legs.
    #[test]
    fn checkpointed_grads_match_dense_dit_and_conditioner() {
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        // Hold every SDPA-segment flag OFF so the ONLY difference between the two legs is whole-block
        // checkpointing of the DiT (isolates the sc-10522 encoder-threading correctness).
        dit.set_sdpa_checkpoint(false);
        cond.set_sdpa_checkpoint(false);
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        let grads_of =
            |dit: &mut CosmosDiT, cond: &mut AnimaTextConditioner, ck: Option<&[Vec<String>]>| {
                let (_l, g) = compute_loss_grads(
                    dit,
                    cond,
                    &params,
                    &adapter,
                    4.0,
                    4.0,
                    &x0,
                    &source,
                    &t5_ids,
                    0.5,
                    &noise,
                    false,
                    None,
                    ck,
                    Dtype::Float32,
                )
                .unwrap();
                eval(g.values()).unwrap();
                g
            };
        let g_dense = grads_of(&mut dit, &mut cond, None);
        let g_ckpt = grads_of(&mut dit, &mut cond, Some(&blocks));

        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-10576] checkpointed-vs-dense grad max rel diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense within fp tolerance: max rel {max_rel:.2e}"
        );
        // Both legs must actually train the conditioner (proves encoder grad flows) and the DiT.
        assert!(
            cond_lora_b_grad(&g_dense) > 1e-6,
            "conditioner must receive gradient (dense)"
        );
        assert!(
            cond_lora_b_grad(&g_ckpt) > 1e-6,
            "conditioner must receive gradient (checkpointed) — the sc-10522 inert-adapter trap"
        );
        assert!(
            dit_lora_b_grad(&g_ckpt) > 1e-6,
            "DiT must receive gradient (checkpointed)"
        );
    }

    /// Grad-parity in the EXACT production combined config (sc-10576). `train_impl` runs a checkpointed
    /// LoRA step with THREE flags at once: DiT whole-block checkpoint ON, DiT sdpa-segment OFF
    /// (`dit.set_sdpa_checkpoint(!use_checkpoint)` → OFF), and conditioner sdpa-segment ON
    /// (`conditioner.set_sdpa_checkpoint(true)`). The other grad-parity tests exercise those mechanisms
    /// only in isolation (whole-block with cond-segment OFF; sdpa-segment on the dense path), so none of
    /// them pins the interaction of all three. This one reproduces the production combination verbatim
    /// and asserts its grads equal the fully-dense reference (no checkpointing anywhere) to fp tolerance
    /// for BOTH the DiT and the conditioner (`llm_adapter`) factors, and that both actually train.
    /// Synthetic small model, Metal, no real weights. Failure-capable: regressing the production forward
    /// to capture `encoder` collapses the conditioner grad to zero and reddens this test (verified by
    /// temporarily routing `compute_loss_grads` through `_encoder_captured`).
    #[test]
    fn production_combined_config_grads_match_dense() {
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        let grads_of = |dit: &mut CosmosDiT,
                        cond: &mut AnimaTextConditioner,
                        dit_seg: bool,
                        cond_seg: bool,
                        ck: Option<&[Vec<String>]>| {
            dit.set_sdpa_checkpoint(dit_seg);
            cond.set_sdpa_checkpoint(cond_seg);
            let (_l, g) = compute_loss_grads(
                dit,
                cond,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &source,
                &t5_ids,
                0.5,
                &noise,
                false,
                None,
                ck,
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };

        // Fully-dense reference: no whole-block checkpointing and every SDPA-segment flag OFF, so every
        // activation is retained — the autograd ground truth.
        let g_dense = grads_of(&mut dit, &mut cond, false, false, None);
        // Production combined config, verbatim: whole-block ON, DiT segment OFF, conditioner segment ON.
        let g_prod = grads_of(&mut dit, &mut cond, false, true, Some(&blocks));

        let max_rel = max_rel_diff(&g_dense, &g_prod);
        eprintln!("[sc-10576] production-combined-vs-dense grad max rel diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "production combined-config grads must match fully-dense within fp tolerance: max rel {max_rel:.2e}"
        );
        // Both factor groups must actually train in the production config (proves the conditioner grad
        // path stays live once whole-block + segment checkpointing are combined — the sc-10522 trap).
        assert!(
            cond_lora_b_grad(&g_prod) > 1e-6,
            "conditioner must receive gradient in the production combined config"
        );
        assert!(
            dit_lora_b_grad(&g_prod) > 1e-6,
            "DiT must receive gradient in the production combined config"
        );
    }

    /// SDPA-segment checkpointing (the dense/LoKr path) must not change grads either: dense grads with
    /// segment ckpt ON == OFF, to fp tolerance.
    #[test]
    fn sdpa_segment_checkpoint_grads_match_retained() {
        let (mut dit, mut cond, params, adapter, _tp, _blocks) = tiny_model_and_adapter();
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);
        let grads_of = |dit: &mut CosmosDiT, cond: &mut AnimaTextConditioner, on: bool| {
            dit.set_sdpa_checkpoint(on);
            cond.set_sdpa_checkpoint(on);
            let (_l, g) = compute_loss_grads(
                dit,
                cond,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &source,
                &t5_ids,
                0.5,
                &noise,
                false,
                None,
                None,
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_off = grads_of(&mut dit, &mut cond, false);
        let g_on = grads_of(&mut dit, &mut cond, true);
        let max_rel = max_rel_diff(&g_off, &g_on);
        eprintln!("[sc-10576] sdpa-seg-ckpt-vs-retained grad max rel diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "SDPA-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// The sc-10522 trap made executable: threading `encoder` keeps the conditioner live; CAPTURING it
    /// (the deliberately-wrong impl) drops the conditioner gradient to ZERO while the DiT still trains
    /// and the loss still falls — exactly the silent inert-adapter failure. A mutation guard: if the
    /// production forward regressed to capturing `encoder`, `checkpointed_grads_match_dense_…` would go
    /// red here (conditioner grad ≈ 0 ≠ the dense non-zero).
    #[test]
    fn captured_encoder_zeros_conditioner_grad_mutation() {
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        dit.set_sdpa_checkpoint(false);
        cond.set_sdpa_checkpoint(false);
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        // Correct (threaded) path.
        let (_l, g_ok) = compute_loss_grads(
            &mut dit,
            &mut cond,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &source,
            &t5_ids,
            0.5,
            &noise,
            false,
            None,
            Some(&blocks),
            Dtype::Float32,
        )
        .unwrap();
        eval(g_ok.values()).unwrap();

        // Wrong (captured) path — same math except `encoder` is a captured constant in each block's
        // checkpoint segment, so the backward produces no cotangent for it.
        let g_bad = grads_encoder_captured(
            &mut dit, &mut cond, &params, &adapter, &x0, &source, &t5_ids, 0.5, &noise, &blocks,
        );

        let ok = cond_lora_b_grad(&g_ok);
        let bad = cond_lora_b_grad(&g_bad);
        let dit_bad = dit_lora_b_grad(&g_bad);
        eprintln!(
            "[sc-10576] conditioner lora_b Σ|grad|: threaded {ok:.3e} vs captured {bad:.3e}; DiT (captured) {dit_bad:.3e}"
        );
        assert!(ok > 1e-6, "threaded encoder: conditioner must train");
        assert!(
            bad < 1e-9,
            "captured encoder: conditioner grad must collapse to ZERO (the trap), got {bad:.3e}"
        );
        assert!(
            dit_bad > 1e-6,
            "captured encoder still trains the DiT — that is why the bug is silent"
        );
    }

    /// sc-24828: the subject-mask weight reaches BOTH backward paths (dense + DiT block-checkpointed,
    /// in the production flag combination) of [`compute_loss_grads`] on the tiny synthetic DiT +
    /// conditioner. An all-ones map equals the unweighted loss; an all-zero map gives loss exactly 0
    /// and all-zero DiT AND conditioner adapter grads on both paths; a half map lands strictly
    /// between and agrees across paths. The latent is `[1, C, 1, H, W]` (frame axis 1), so the weight
    /// is the cached latent's shape — no packing.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        use mlx_gen::train::loss::subject_mask_weight;
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        // Non-zero factors on both sides (lora_b inits at zero, which would zero the lora_a grads
        // trivially).
        let scale = Array::from_slice(&[0.05f32], &[1]);
        let params: LoraParams = params
            .iter()
            .enumerate()
            .map(|(i, (k, v))| {
                let r = random::normal::<f32>(
                    v.shape(),
                    None,
                    None,
                    Some(&random::key(100 + i as u64).unwrap()),
                )
                .unwrap();
                (k.clone(), multiply(&r, &scale).unwrap())
            })
            .collect();
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);
        let shape = x0.shape().to_vec();
        let (gh, gw) = (shape[3] as usize, shape[4] as usize);
        let map = |v: &[f32]| subject_mask_weight(v, gh, gw, &shape).unwrap();
        let mut run = |weight: Option<&Array>, ckpt: bool| {
            // Dense: every segment flag off. Checkpointed: the production combination (DiT
            // whole-block ON + DiT segment OFF + conditioner segment ON).
            dit.set_sdpa_checkpoint(false);
            cond.set_sdpa_checkpoint(ckpt);
            let (l, g) = compute_loss_grads(
                &mut dit,
                &mut cond,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &source,
                &t5_ids,
                0.5,
                &noise,
                false,
                weight,
                ckpt.then_some(blocks.as_slice()),
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            (l, g)
        };
        let n = gh * gw;
        let (plain, g_plain) = run(None, false);
        assert!(cond_lora_b_grad(&g_plain) > 1e-6 && dit_lora_b_grad(&g_plain) > 1e-6);
        let ones = map(&vec![1.0; n]);
        assert!((run(Some(&ones), false).0 - plain).abs() < 1e-6);
        let zeros = map(&vec![0.0; n]);
        for ckpt in [false, true] {
            let (loss, grads) = run(Some(&zeros), ckpt);
            assert_eq!(
                loss, 0.0,
                "ckpt={ckpt}: an all-background map must zero the loss"
            );
            assert!(!grads.is_empty());
            for (k, g) in &grads {
                let m = g.abs().unwrap().max(None).unwrap().item::<f32>();
                assert_eq!(m, 0.0, "ckpt={ckpt}: nonzero adapter grad on {k}");
            }
        }
        let half: Vec<f32> = (0..n)
            .map(|i| if i % gw < gw / 2 { 1.0 } else { 0.0 })
            .collect();
        let half = map(&half);
        let (dense, _) = run(Some(&half), false);
        let (ckpt, _) = run(Some(&half), true);
        assert!(dense > 0.0 && dense < plain, "{dense} vs {plain}");
        assert!(
            (dense - ckpt).abs() < 1e-4,
            "dense {dense} vs checkpoint {ckpt}"
        );
    }

    /// Compute grads via the deliberately-wrong `encoder`-captured checkpoint forward (mutation test).
    #[allow(clippy::too_many_arguments)]
    fn grads_encoder_captured(
        dit: &mut CosmosDiT,
        cond: &mut AnimaTextConditioner,
        params: &LoraParams,
        adapter: &TrainAdapter,
        x0: &Array,
        source: &Array,
        t5_ids: &Array,
        sigma: f32,
        noise: &Array,
        blocks: &[Vec<String>],
    ) -> LoraParams {
        let (x_t, target, timestep) = build_batch(x0, noise, sigma).unwrap();
        let src = source.clone();
        let ids = t5_ids.clone();
        let blk = blocks.to_vec();
        let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            {
                let mut host = AnimaAdapterHost {
                    dit: &mut *dit,
                    conditioner: &mut *cond,
                };
                adapter.install_as(&mut host, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            }
            let enc = cond
                .forward(&src, &ids, Dtype::Float32)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let s = Array::from_slice(&[timestep], &[1]);
            let v = dit
                .forward_with_main_checkpointed_encoder_captured(
                    &x_t,
                    &s,
                    &enc,
                    Dtype::Float32,
                    &p,
                    &blk,
                    4.0,
                )
                .map_err(|e| Exception::custom(e.to_string()))?;
            let diff = subtract(&v, &target)?;
            Ok(vec![diff.square()?.mean(None)?])
        };
        let mut vg = keyed_value_and_grad(loss_fn);
        vg(params.clone(), 0).unwrap().1
    }

    #[test]
    fn collect_dit_block_local_targets_excludes_conditioner() {
        let tp = vec![
            "blocks.0.self_attn.q_proj".to_string(),
            "blocks.1.mlp.layer2".to_string(),
            "blocks.1.adaln_modulation_mlp.2".to_string(),
            "llm_adapter.blocks.0.self_attn.q_proj".to_string(),
        ];
        let out = collect_dit_block_local_targets(&tp, 2);
        assert_eq!(out[0], vec!["self_attn.q_proj".to_string()]);
        assert_eq!(
            out[1],
            vec![
                "mlp.layer2".to_string(),
                "adaln_modulation_mlp.2".to_string()
            ]
        );
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(
            total, 3,
            "the conditioner llm_adapter target must be excluded"
        );
    }

    #[test]
    fn unified_tokens_grows_with_edge() {
        assert_eq!(unified_tokens(512), 32.0 * 32.0 + 512.0);
        assert_eq!(unified_tokens(1024), 64.0 * 64.0 + 512.0);
        assert!(unified_tokens(1536) > unified_tokens(1024));
    }

    /// The pre-flight guard mechanism, deterministically (no real weights): an over-budget projection
    /// returns a catchable, flag-recommending error; a within-budget one is `Ok`. Drives the MLX
    /// memory limit directly so the assertion is machine-independent.
    #[test]
    fn preflight_guard_refuses_over_budget() {
        use mlx_rs::memory::set_memory_limit;
        let prev = set_memory_limit(8 * 1024 * 1024 * 1024); // 8 GB budget → safe ~6.8 GB
        let over = preflight_memory_guard(&TrainingConfig::default(), &[1536], true, 0.0, false);
        set_memory_limit(256 * 1024 * 1024 * 1024); // 256 GB budget → safe ~217 GB
        let under = preflight_memory_guard(&TrainingConfig::default(), &[512], true, 0.0, false);
        set_memory_limit(prev); // restore
        let err = over.unwrap_err().to_string();
        assert!(
            err.contains("Gradient Checkpointing") && err.contains("1536"),
            "over-budget error must be actionable + name the resolution: {err}"
        );
        assert!(
            under.is_ok(),
            "a 512² run under a 256 GB budget must pass the guard"
        );
    }

    // ======================================================================================
    // sc-10641 — in-training preview sampling
    // ======================================================================================

    /// Max abs element-wise diff between two same-shape latents.
    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        a.subtract(b)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>()
    }

    /// The interval contract, deterministically (no weights): previews are enabled only with a positive
    /// cadence + at least one prompt + not cancelled, and fire on EXACTLY the cadence multiples — nowhere
    /// else. A regression to `>=`, an off-by-one, or firing when disabled reddens this.
    #[test]
    fn preview_cadence_contract() {
        let base = TrainingConfig {
            sample_every: 5,
            sample_prompts: vec!["1girl".into()],
            ..Default::default()
        };
        assert!(previews_enabled(&base, false));
        assert!(!previews_enabled(&base, true), "cancelled ⇒ disabled");
        assert!(
            !previews_enabled(
                &TrainingConfig {
                    sample_every: 0,
                    ..base.clone()
                },
                false
            ),
            "cadence 0 ⇒ disabled"
        );
        assert!(
            !previews_enabled(
                &TrainingConfig {
                    sample_prompts: vec![],
                    ..base.clone()
                },
                false
            ),
            "no prompts ⇒ disabled"
        );
        // The default config never opts in.
        assert!(!previews_enabled(&TrainingConfig::default(), false));

        // Fires on multiples of the cadence, nowhere else.
        let due: Vec<u32> = (1..=12).filter(|&s| preview_due(s, 5)).collect();
        assert_eq!(due, vec![5, 10]);
        assert!(!preview_due(1, 5) && !preview_due(7, 5) && !preview_due(11, 5));
        assert!(
            (1..=10).all(|s| !preview_due(s, 0)),
            "cadence 0 never fires"
        );
    }

    /// sc-10641 live-graph guard (the critical correctness point). The preview MUST re-run the
    /// conditioner (`llm_adapter`) through the LIVE graph so it reflects the in-training conditioner
    /// adapters — never a cached output (the sc-10522 trap). Synthetic small model, Metal, no real
    /// weights. A trained (non-zero `lora_b`) adapter is installed on the CONDITIONER ONLY (the DiT
    /// factors stay at the `B=0` no-op), so the ONLY thing that can move the preview latent is the live
    /// conditioner. The production `render_preview_latent` (conditioner run live) is compared against
    /// `render_latent_with_enc` fed a STALE conditioner output captured BEFORE install:
    ///   - live ≠ stale  ⇒ the preview genuinely re-ran the conditioner adapters — FAILS if it ever
    ///     regressed to caching the conditioner output.
    ///   - live == render_latent_with_enc(live enc) ⇒ the conditioner is the sole live-dependent input,
    ///     so the first assertion can't pass for a spurious reason.
    #[test]
    fn preview_samples_conditioner_through_live_graph() {
        let (mut dit, mut cond, params0, adapter, _tp, _blocks) = tiny_model_and_adapter();
        // `tiny_inputs` returns (x0, source, t5_ids, noise); the noise is the preview's starting latent.
        let (_x0, source, t5_ids, init) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        // Stale conditioner output — captured from the BASE conditioner (no adapters installed). This is
        // exactly what caching the conditioner OUTPUT would freeze into the preview (the trap).
        let enc_stale = cond.forward(&source, &t5_ids, Dtype::Float32).unwrap();
        eval([&enc_stale]).unwrap();

        // "Train" the conditioner: give its `lora_b` non-zero values (build_lora_targets inits them to
        // 0). DiT `lora_b` left at 0 (inert) ⇒ the DiT forward is constant across both legs.
        let mut trained = params0.clone();
        for (k, v) in trained.iter_mut() {
            if k.starts_with("llm_adapter.") && k.ends_with(".lora_b") {
                *v = random::normal::<f32>(v.shape(), None, None, Some(&random::key(99).unwrap()))
                    .unwrap();
            }
        }
        {
            let mut host = AnimaAdapterHost {
                dit: &mut dit,
                conditioner: &mut cond,
            };
            adapter
                .install_as(&mut host, &trained, 4.0, 4.0, None, LOKR_DTYPE)
                .unwrap();
        }

        let steps = 3usize;
        // Production preview path: conditioner run LIVE (reflects the trained conditioner adapters).
        let latent_live = crate::pipeline::render_preview_latent(
            &dit,
            &cond,
            &source,
            &t5_ids,
            None,
            &init,
            steps,
            1.0,
            7,
            Dtype::Float32,
            &Default::default(),
        )
        .unwrap();
        // Same DiT, but a STALE (pre-training) conditioner output — the trap.
        let latent_stale = crate::pipeline::render_latent_with_enc(
            &dit,
            &enc_stale,
            None,
            &init,
            steps,
            1.0,
            7,
            Dtype::Float32,
            &Default::default(),
        )
        .unwrap();
        // Positive control: feed render_latent_with_enc the LIVE conditioner output → must equal the
        // live preview (proves the conditioner is the only live-dependent input; no spurious diff).
        let enc_live = cond.forward(&source, &t5_ids, Dtype::Float32).unwrap();
        let latent_ctrl = crate::pipeline::render_latent_with_enc(
            &dit,
            &enc_live,
            None,
            &init,
            steps,
            1.0,
            7,
            Dtype::Float32,
            &Default::default(),
        )
        .unwrap();
        eval([&latent_live, &latent_stale, &latent_ctrl]).unwrap();

        let live_vs_stale = max_abs_diff(&latent_live, &latent_stale);
        let live_vs_ctrl = max_abs_diff(&latent_live, &latent_ctrl);
        eprintln!(
            "[sc-10641] preview latent live-vs-stale {live_vs_stale:.3e}, live-vs-ctrl {live_vs_ctrl:.3e}"
        );
        assert!(
            live_vs_stale > 1e-4,
            "preview must re-run the conditioner LIVE: a cached (stale) conditioner output yields the \
             same latent (max abs diff {live_vs_stale:.3e}) — the sc-10522 inert-adapter trap"
        );
        assert!(
            live_vs_ctrl < 1e-5,
            "render_preview_latent's only live input is the conditioner: feeding the live enc must \
             reproduce the preview (max abs diff {live_vs_ctrl:.3e})"
        );
    }

    // -------- real-weights measurement + validation (sc-10576), #[ignore]d + snapshot-gated --------

    /// Resolve the Anima `split_files/` dir from the required `ANIMA_SNAPSHOT` env var. sc-13668:
    /// there is no implicit default — the source snapshot path must be passed in explicitly.
    fn anima_split() -> Option<std::path::PathBuf> {
        std::env::var("ANIMA_SNAPSHOT")
            .ok()
            .map(std::path::PathBuf::from)
    }

    #[test]
    fn source_root_requires_explicit_env_no_default() {
        let key = "ANIMA_SNAPSHOT";
        let saved = std::env::var(key).ok();
        std::env::remove_var(key);
        assert!(
            anima_split().is_none(),
            "the source split_files/ dir must come from {key}: sc-13668 removed the implicit default"
        );
        std::env::set_var(key, "/sentinel/anima/split_files");
        assert_eq!(
            anima_split(),
            Some(std::path::PathBuf::from("/sentinel/anima/split_files"))
        );
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// Load the real DiT + conditioner (+ keep the VAE resident like the train loop); drop the Qwen3
    /// text encoder + tokenizers, exactly as `train_impl` does post-caching.
    fn load_real_dit_cond(
        split: &std::path::Path,
    ) -> (CosmosDiT, AnimaTextConditioner, crate::vae::QwenVae) {
        use mlx_gen::WeightsSource;
        let comps =
            AnimaComponents::load(&WeightsSource::Dir(split.to_path_buf()), Variant::Base).unwrap();
        let AnimaComponents {
            dit,
            conditioner,
            vae,
            text_encoder,
            tokenizers,
        } = comps;
        drop(text_encoder);
        drop(tokenizers);
        mlx_rs::memory::clear_cache();
        (dit, conditioner, vae)
    }

    /// Run one real bf16 first training step at `edge` and return `(peak GB, loss)`. Synthesizes the
    /// cached inputs (latent + Qwen states + T5 ids) directly — the memory profile is value-independent,
    /// so this needs only the real DiT/conditioner weights, not the VAE/text encoders.
    fn measure_first_step(
        dit: &mut CosmosDiT,
        cond: &mut AnimaTextConditioner,
        edge: i32,
        use_checkpoint: bool,
    ) -> (f32, f32) {
        use mlx_rs::memory::{get_peak_memory, reset_peak_memory};
        let tcfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(dit, cond, &tcfg);
        let (targets, params) = {
            let mut host = AnimaAdapterHost {
                dit,
                conditioner: cond,
            };
            build_lora_targets(&mut host, &target_paths, 16, 7).unwrap()
        };
        let adapter = TrainAdapter::Lora { targets };
        let blocks = collect_dit_block_local_targets(&target_paths, dit.config().num_layers);
        let ck: Option<&[Vec<String>]> = if use_checkpoint { Some(&blocks) } else { None };
        dit.set_sdpa_checkpoint(!use_checkpoint);
        cond.set_sdpa_checkpoint(true);

        let hl = edge / 8;
        let x0 = random::normal::<f32>(
            &[1, 16, 1, hl, hl],
            None,
            None,
            Some(&random::key(1).unwrap()),
        )
        .unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(2).unwrap())).unwrap();
        let source =
            random::normal::<f32>(&[1, 64, 1024], None, None, Some(&random::key(3).unwrap()))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
        let ids = random::uniform::<_, f32>(0.0, 32000.0, &[1, 32], Some(&random::key(4).unwrap()))
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap();
        eval([&x0, &noise, &source, &ids]).unwrap();

        reset_peak_memory();
        let (loss, grads) = compute_loss_grads(
            dit,
            cond,
            &params,
            &adapter,
            16.0,
            16.0,
            &x0,
            &source,
            &ids,
            0.5,
            &noise,
            false,
            None,
            ck,
            Dtype::Bfloat16,
        )
        .unwrap();
        eval(grads.values()).unwrap();
        let peak = get_peak_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
        (peak, loss)
    }

    /// sc-10576 CALIBRATION: sweep the dense (whole-block OFF, SDPA-segment ON) first-step peak at
    /// 512/768/1024 to fit `projected_dense_peak_gb`. Prints measured vs. projected — refit the
    /// `ANIMA_PEAK_*` constants if these move materially.
    #[test]
    #[ignore = "needs the circlestone-labs/Anima snapshot; measures GPU peak (sc-10576 calibration)"]
    fn first_step_dense_peak_sweep() {
        let split = anima_split().expect("set ANIMA_SNAPSHOT to the Anima split_files/ dir");
        let (mut dit, mut cond, _vae) = load_real_dit_cond(&split);
        eprintln!("[sc-10576] dense first-step peak sweep (bf16, rank16, SDPA-seg ckpt ON):");
        for edge in [512, 768, 1024] {
            let (peak, loss) = measure_first_step(&mut dit, &mut cond, edge, false);
            let s = unified_tokens(edge as u32);
            eprintln!(
                "[sc-10576]   edge {edge:>4}  s {s:>7.0}  peak {peak:6.2} GB  loss {loss:.4}  (projected {:6.2} GB)",
                projected_dense_peak_gb(s, true)
            );
        }
    }

    /// sc-10576 VALIDATION: (1) a direct measured A/B at 1024² (both fit) proving whole-block
    /// checkpointing reduces the first-step peak, then (2) the 1536² criterion — checkpointing makes
    /// the step fit, while the dense projection is refused by the pre-flight guard.
    #[test]
    #[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (2B DiT steps at 1024²/1536²)"]
    fn first_step_1536_checkpointed_vs_dense() {
        let split = anima_split().expect("set ANIMA_SNAPSHOT to the Anima split_files/ dir");
        let (mut dit, mut cond, _vae) = load_real_dit_cond(&split);
        let budget = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);

        // (1) Measured A/B at 1024 — both fit, so the reduction is observed, not projected.
        let (dense_1024, _) = measure_first_step(&mut dit, &mut cond, 1024, false);
        let (ckpt_1024, _) = measure_first_step(&mut dit, &mut cond, 1024, true);
        eprintln!(
            "[sc-10576] edge 1024  dense {dense_1024:.2} GB  ckpt {ckpt_1024:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_1024 / dense_1024)
        );
        assert!(
            ckpt_1024 < dense_1024,
            "whole-block checkpointing must reduce the first-step peak: dense {dense_1024:.2} GB vs ckpt {ckpt_1024:.2} GB"
        );

        // (2) The 1536² criterion: checkpointed fits, dense is over the safe budget → guard refuses.
        let (ck_peak, ck_loss) = measure_first_step(&mut dit, &mut cond, 1536, true);
        let dense_proj = projected_dense_peak_gb(unified_tokens(1536), true);
        let refused =
            preflight_memory_guard(&TrainingConfig::default(), &[1536], true, 0.0, false).is_err();
        eprintln!(
            "[sc-10576] edge 1536 CHECKPOINTED peak {ck_peak:.2} GB loss {ck_loss:.4} | budget {budget:.0} GB | dense projected {dense_proj:.1} GB | preflight-refuses {refused}"
        );
        assert!(
            ck_peak as f64 <= budget,
            "checkpointed 1536 must fit this machine's budget: {ck_peak:.2} GB vs {budget:.0} GB"
        );
        assert!(
            refused,
            "dense 1536 (projected {dense_proj:.0} GB) must be refused by the pre-flight guard"
        );
    }

    // ==============================================================================================
    // sc-10642 — mid-run resume (CI, Metal-synthetic; no real weights)
    // ==============================================================================================

    /// Round-trip: a few optimizer steps over the Anima trainable surface → `save_resume` → discover →
    /// `load_resume` restores the optimizer state + trainable factors + step/update index. The restored
    /// factors match bit-for-bit, the step count comes back as N (not 0), and the restored optimizer's
    /// next step matches the uninterrupted optimizer's — the sc-9560 exactness the resume wiring relies
    /// on, exercised on Anima's own two-host (DiT + `llm_adapter`) factor map.
    #[test]
    fn resume_round_trips_optimizer_factors_and_step() {
        let (_dit, _cond, params, _adapter, _tp, _blocks) = tiny_model_and_adapter();
        // Synthetic non-zero grads keyed exactly as the factors (so both optimizer instances step
        // identically). `lora_b` starts at 0, so a real grad is needed to move the state.
        let grads: LoraParams = params
            .iter()
            .map(|(k, v)| {
                let g =
                    random::normal::<f32>(v.shape(), None, None, Some(&random::key(99).unwrap()))
                        .unwrap();
                (k.clone(), g)
            })
            .collect();
        let mut opt = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
        opt.set_lr_scaled(1.0);
        let mut p = params.clone();
        opt.step(&mut p, &grads).unwrap();
        opt.step(&mut p, &grads).unwrap();
        eval(p.values()).unwrap();

        // Per-process scratch dir: the `remove_dir_all` below would otherwise wipe a second
        // concurrent `cargo test` process's fixtures out of the shared `$TMPDIR`.
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let stem = "anima_style";
        checkpoint::save_resume(&dir, stem, 4, 2, &opt, &p).unwrap();

        // Discovery returns the snapshot's step; load restores it.
        let (found, step) = checkpoint::find_latest_resume(&dir, stem).expect("resume snapshot");
        assert_eq!(step, 4, "find_latest_resume returns the snapshot step");
        let mut opt2 = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
        let (loaded, meta) = checkpoint::load_resume(&found, &mut opt2).unwrap();

        // The Anima surface guard passes on a faithful round-trip (fresh-build keys == restored keys).
        assert_resume_surface_matches(&loaded, &params).unwrap();

        // Step count + update index restored — resume continues at N, not 0.
        assert_eq!(meta.step, 4, "resumes at the recorded step, not 0");
        assert_eq!(meta.update_idx, 2, "optimizer-update index restored");
        assert_eq!(meta.optimizer, "adamw");

        // Trainable factors (adapter weights) restored bit-for-bit.
        assert_eq!(loaded.len(), p.len(), "same factor count");
        for (k, v) in &p {
            let l = loaded.get(k).expect("restored factor");
            let d = v
                .subtract(l)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(d == 0.0, "factor {k} not restored bit-exact: |Δ| = {d:e}");
        }

        // Optimizer state restored: the restored optimizer's next step == the uninterrupted one's.
        let mut a = p.clone();
        let mut b = loaded;
        opt.set_lr_scaled(1.0);
        opt2.set_lr_scaled(1.0);
        opt.step(&mut a, &grads).unwrap();
        opt2.step(&mut b, &grads).unwrap();
        eval(a.values()).unwrap();
        eval(b.values()).unwrap();
        let m = max_rel_diff(&a, &b);
        assert!(
            m <= 1e-6,
            "restored optimizer's next step diverged: max_rel {m:e}"
        );
    }

    /// The sc-10522 restore assertion. The guard passes when the checkpoint's factor surface matches the
    /// live model's, and ERRORS (failure-capable) when the checkpoint dropped the conditioner
    /// (`llm_adapter.*`) targets — the trap where the DiT keeps training and every structural check
    /// passes while the conditioner is silently inert — or dropped any single factor.
    #[test]
    fn resume_restore_asserts_full_target_surface() {
        let (_dit, _cond, params, _adapter, _tp, _blocks) = tiny_model_and_adapter();

        // The tiny synthetic surface is a scaled-down analogue of the real 448 DiT + 60 `llm_adapter`
        // split; the guard is size-agnostic (it asserts restored == live surface). Both classes present.
        let (dit_paths, cond_paths) = split_target_surface(&params);
        assert!(!dit_paths.is_empty(), "tiny surface has DiT targets");
        assert!(
            !cond_paths.is_empty(),
            "tiny surface has conditioner targets"
        );

        // Faithful restore (identical surface) passes.
        assert_resume_surface_matches(&params, &params).unwrap();

        // Drop the conditioner (`llm_adapter.*`) factors — the sc-10522 inert-conditioner trap. ERROR.
        let mut dropped = params.clone();
        let cond_keys: Vec<_> = dropped
            .keys()
            .filter(|k| k.starts_with("llm_adapter."))
            .cloned()
            .collect();
        assert!(
            !cond_keys.is_empty(),
            "there are conditioner factors to drop"
        );
        for k in cond_keys {
            dropped.remove(&k);
        }
        let err = assert_resume_surface_matches(&dropped, &params)
            .expect_err("a checkpoint missing the conditioner targets must be refused");
        let msg = err.to_string();
        assert!(msg.contains("sc-10522"), "error names the trap: {msg}");
        assert!(
            msg.contains("0 `llm_adapter` conditioner"),
            "error reports 0 conditioner targets in the checkpoint: {msg}"
        );

        // Dropping a single DiT factor is also caught (wrong count within a class, not a whole class).
        let mut one_missing = params.clone();
        let a_dit = one_missing
            .keys()
            .find(|k| !k.starts_with("llm_adapter.") && k.ends_with(".lora_a"))
            .cloned()
            .expect("a DiT lora_a factor");
        one_missing.remove(&a_dit);
        assert_resume_surface_matches(&one_missing, &params)
            .expect_err("a checkpoint missing a single factor must be refused");
    }
}

/// The empirical fit must reproduce the measured Anima dense first-step peaks (sc-10576) within a few
/// GB, stay monotone in `s`, and put the checkpointing regime on the right side of a typical budget.
#[cfg(test)]
mod preflight_tests {
    use super::{projected_dense_peak_gb, unified_tokens};

    #[test]
    fn projected_peak_reproduces_measured_points() {
        // SCOPE: this guards coefficient-TRANSCRIPTION only, NOT fit accuracy. `projected_dense_peak_gb`
        // is an EXACT 3-point quadratic fit (`a + b·s + c·s²`) through these same three calibration
        // points, so an intact fit ALWAYS reproduces its own interpolation points — this test can catch a
        // fat-fingered/drifted coefficient (or a broken `unified_tokens`), but by construction it cannot
        // vouch for how well the curve predicts a HELD-OUT edge. Real fit accuracy is validated against
        // fresh GPU measurements by the `#[ignore]`d `first_step_dense_peak_sweep` (refit both if it
        // prints materially different peaks). We deliberately do NOT add a synthetic 4th point here: a
        // meaningful held-out check needs a real measurement, and a made-up one would prove nothing.
        //
        // Measured on the 128 GB target (first_step_dense_peak_sweep, bf16, rank16, SDPA-seg ON):
        //   edge 512  (s 1536) → 21.18 GB
        //   edge 768  (s 2816) → 32.05 GB
        //   edge 1024 (s 4608) → 49.44 GB
        // The exact 3-point fit must reproduce each within ~1 GB.
        let approx = |edge: u32, want: f64| {
            let got = projected_dense_peak_gb(unified_tokens(edge), true);
            assert!(
                (got - want).abs() < 1.0,
                "edge {edge}: projected {got:.2} GB vs measured {want:.2} GB"
            );
        };
        approx(512, 21.18);
        approx(768, 32.05);
        approx(1024, 49.44);
    }

    #[test]
    fn projected_peak_puts_1536_over_a_128gb_budget() {
        // Dense 1536² (s 9728) extrapolates well above a 128 GB machine's working set — so the guard
        // refuses it (or it SIGKILLs), which is exactly the regime whole-block checkpointing unblocks.
        let dense_1536 = projected_dense_peak_gb(unified_tokens(1536), true);
        assert!(
            dense_1536 > 100.0,
            "dense 1536² must project over budget (got {dense_1536:.0} GB)"
        );
    }

    #[test]
    fn projected_peak_is_monotone_and_ordered() {
        let p = |edge: u32| projected_dense_peak_gb(unified_tokens(edge), true);
        assert!(p(512) < p(1024));
        assert!(p(1024) < p(1536));
        assert!(p(1536) < p(2048));
        // f32 upper bound is strictly above the bf16 fit.
        let s = unified_tokens(1024);
        assert!(projected_dense_peak_gb(s, false) > projected_dense_peak_gb(s, true));
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the Anima step seam on the tiny synthetic DiT +
/// conditioner (`CosmosDiT::synthetic` / `AnimaTextConditioner::synthetic`, 4-channel latent
/// `[1, 4, 1, 4, 4]`) with a random-init tiny-width TAEHV carrying TAEW2.1's hyperparameters at 4
/// latent channels, and a random-init tiny Depth-Anything-V2. Drives the same
/// [`compute_step_loss_grads`] / [`aux_driver`] `train_impl` runs. Seconds; no weights
/// downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use crate::config::{ConditionerConfig, DitConfig};
    use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule};
    use mlx_gen::train::taehv::{synthetic_taehv_weights, TaehvDecoder};
    use mlx_gen_depth::anchor::{synthetic_weights, tiny_config, DepthAnchorLoss};
    use mlx_gen_depth::DepthAnythingV2;

    fn schedule() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    fn path() -> PerceptualPath {
        let tae = TaehvConfig {
            latent_channels: 4,
            channels: [8, 6, 4, 4],
            ..TaehvConfig::taew2_1()
        };
        let dec =
            TaehvDecoder::from_weights(&synthetic_taehv_weights(&tae, 11).unwrap(), tae).unwrap();
        let da2 = tiny_config();
        let depth = DepthAnchorLoss::new(
            DepthAnythingV2::from_weights(&synthetic_weights(&da2, 12).unwrap(), da2).unwrap(),
        );
        PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule: schedule(),
                loss: Box::new(depth),
            }],
        )
        .unwrap()
    }

    struct Fixture {
        dit: CosmosDiT,
        cond: AnimaTextConditioner,
        params: LoraParams,
        adapter: TrainAdapter,
        blocks: Vec<Vec<String>>,
        x0: Array,
        source: Array,
        t5_ids: Array,
        noise: Array,
    }

    fn fixture() -> Fixture {
        let dcfg = DitConfig {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 8,
            num_layers: 2,
            mlp_ratio: 2.0,
            text_embed_dim: 16,
            adaln_lora_dim: 8,
            max_size: (4, 16, 16),
            patch_size: (1, 2, 2),
            rope_scale: (1.0, 4.0, 4.0),
            concat_padding_mask: true,
        };
        let ccfg = ConditionerConfig {
            source_dim: 16,
            target_dim: 16,
            model_dim: 16,
            num_layers: 2,
            num_attention_heads: 2,
            mlp_ratio: 2.0,
            target_vocab_size: 32,
            min_sequence_length: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        };
        let mut dit = CosmosDiT::synthetic(dcfg, 42);
        let mut cond = AnimaTextConditioner::synthetic(ccfg, 43);
        let tcfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cond, &tcfg);
        let (targets, params) = {
            let mut host = AnimaAdapterHost {
                dit: &mut dit,
                conditioner: &mut cond,
            };
            build_lora_targets(&mut host, &target_paths, 4, 7).unwrap()
        };
        let blocks = collect_dit_block_local_targets(&target_paths, dcfg.num_layers);
        let key = |k| Some(random::key(k).unwrap());
        let x0 = random::normal::<f32>(&[1, 4, 1, 4, 4], None, None, key(11).as_ref()).unwrap();
        let noise = random::normal::<f32>(x0.shape(), None, None, key(12).as_ref()).unwrap();
        let source = random::normal::<f32>(&[1, 6, 16], None, None, key(13).as_ref()).unwrap();
        let t5_ids = random::uniform::<_, f32>(0.0, 32.0, &[1, 4], key(14).as_ref())
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap();
        eval([&x0, &noise, &source, &t5_ids]).unwrap();
        Fixture {
            dit,
            cond,
            params,
            adapter: TrainAdapter::Lora { targets },
            blocks,
            x0,
            source,
            t5_ids,
            noise,
        }
    }

    fn step(
        f: &mut Fixture,
        sigma: f32,
        ckpt: bool,
        aux: Option<AuxStep<'_>>,
    ) -> (StepLosses, LoraParams) {
        f.dit.set_sdpa_checkpoint(!ckpt);
        f.cond.set_sdpa_checkpoint(true);
        let (l, g) = compute_step_loss_grads(
            &mut f.dit,
            &mut f.cond,
            &f.params,
            &f.adapter,
            4.0,
            4.0,
            &f.x0,
            &f.source,
            &f.t5_ids,
            sigma,
            &f.noise,
            false,
            None,
            ckpt.then_some(f.blocks.as_slice()),
            Dtype::Float32,
            aux,
        )
        .unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    /// The one-item, one-bucket schedule `train_impl` builds for a single cached entry.
    fn one_item() -> BucketSchedule {
        BucketSchedule::new(1, &TrainingConfig::default().training_buckets(), 7)
    }

    fn cache_of(f: &Fixture, n: usize) -> Vec<(Array, Array, Array, Option<Array>)> {
        (0..n)
            .map(|_| (f.x0.clone(), f.source.clone(), f.t5_ids.clone(), None))
            .collect()
    }

    fn bits(a: &Array) -> Vec<u32> {
        a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
    }

    /// AC (a)+(b), dense and block-checkpointed: a depth step (key 2) computes no diffusion term,
    /// its total IS the weighted depth term, and both the DiT's and the conditioner's zero-init
    /// LoRA-B factors get a nonzero finite gradient through the decoded x0; a diffusion step
    /// (key 1) carries no depth term. Mutation: force `diffusion_on = true` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only_on_both_paths() {
        let mut f = fixture();
        let d = aux_driver(path(), &cache_of(&f, 1), &one_item(), 1, 0).unwrap();
        let p = d.into_path();
        for ckpt in [false, true] {
            let plan = p.plan(1, 0, 0.5).unwrap();
            let (diff, _) = step(
                &mut f,
                plan.noise_level,
                ckpt,
                Some(AuxStep {
                    path: &p,
                    plan: &plan,
                    image: 0,
                }),
            );
            assert_eq!(diff.aux, None, "ckpt={ckpt}");
            assert_eq!(Some(diff.total), diff.diffusion);
            let plan = p.plan(2, 0, 0.5).unwrap();
            assert!(!plan.diffusion);
            let (depth, g) = step(
                &mut f,
                plan.noise_level,
                ckpt,
                Some(AuxStep {
                    path: &p,
                    plan: &plan,
                    image: 0,
                }),
            );
            assert_eq!(depth.diffusion, None, "ckpt={ckpt}");
            let aux = depth.aux.expect("depth term");
            assert!(aux > 0.0 && aux.is_finite(), "ckpt={ckpt}: {aux}");
            assert_eq!(depth.total, aux);
            let lora_b = |cond: bool| -> f32 {
                g.iter()
                    .filter(|(k, _)| {
                        k.starts_with("llm_adapter.") == cond && k.ends_with(".lora_b")
                    })
                    .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
                    .sum()
            };
            let (dit_b, cond_b) = (lora_b(false), lora_b(true));
            assert!(
                dit_b > 0.0 && dit_b.is_finite(),
                "ckpt={ckpt}: DiT LoRA-B {dit_b}"
            );
            assert!(
                cond_b > 0.0 && cond_b.is_finite(),
                "ckpt={ckpt}: cond LoRA-B {cond_b}"
            );
        }
    }

    /// AC (c): depth off ⇒ bit-identical to the pre-epic-2123 step (its closure reproduced), and a
    /// diffusion-only step of an enabled path takes the same graph. Mutation: scale the diffusion
    /// reduction (×1.0001) ⇒ red.
    #[test]
    fn depth_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path(&TrainingConfig::default())
            .unwrap()
            .is_none());
        assert_eq!(
            perceptual_footprint_gb(&TrainingConfig::default(), 1024, 4),
            0.0
        );
        let mut f = fixture();
        let (off, g_off) = step(&mut f, 0.5, false, None);
        assert_eq!(off.aux, None);
        let (x_t, target, timestep) = build_batch(&f.x0, &f.noise, 0.5).unwrap();
        let (src, ids) = (f.source.clone(), f.t5_ids.clone());
        let dit = &mut f.dit;
        let cond = &mut f.cond;
        let adapter = &f.adapter;
        dit.set_sdpa_checkpoint(true);
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            {
                let mut host = AnimaAdapterHost {
                    dit: &mut *dit,
                    conditioner: &mut *cond,
                };
                adapter.install_as(&mut host, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            }
            let enc = cond
                .forward(&src, &ids, Dtype::Float32)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let s = Array::from_slice(&[timestep], &[1]);
            let v = dit
                .forward(&x_t, &s, &enc, Dtype::Float32)
                .map_err(|e| Exception::custom(e.to_string()))?
                .as_dtype(Dtype::Float32)?;
            Ok(vec![reduce_loss(&subtract(&v, &target)?, None, false)?])
        };
        let (val, g_legacy) = keyed_value_and_grad(legacy)(f.params.clone(), 0).unwrap();
        eval(g_legacy.values()).unwrap();
        assert_eq!(off.total, val[0].item::<f32>());
        for (k, v) in &g_legacy {
            assert_eq!(bits(v), bits(&g_off[k]), "{k}");
        }
        let d = aux_driver(path(), &cache_of(&f, 1), &one_item(), 1, 0).unwrap();
        let p = d.into_path();
        let plan = p.plan(1, 0, 0.5).unwrap();
        let (on, g_on) = step(
            &mut f,
            0.5,
            false,
            Some(AuxStep {
                path: &p,
                plan: &plan,
                image: 0,
            }),
        );
        assert_eq!(on, off);
        for (k, v) in &g_off {
            assert_eq!(bits(v), bits(&g_on[k]), "{k}");
        }
    }

    /// With two resolution buckets every image strictly alternates diffusion / depth across its
    /// buckets in visit order (key = real item; reference = (item, bucket) entry), references once
    /// per entry. Mutation: key on the cache entry ⇒ red.
    #[test]
    fn two_buckets_alternate_per_item_with_per_entry_references() {
        let f = fixture();
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
        let cache = cache_of(&f, items * buckets.len());
        let schedule = BucketSchedule::new(items, &buckets, 7);
        let mut d = aux_driver(path(), &cache, &schedule, 1, 0).unwrap();
        let steps = 2 * schedule.epoch_len() as u32;
        let mut kinds = Vec::new();
        for s in 1..=steps {
            let plan = d.sample(s, &schedule).plan(0.5).unwrap().unwrap().plan;
            kinds.push((schedule.sample((s - 1) as usize).0, !plan.diffusion));
        }
        assert!((0..steps as usize).any(|k| schedule.cache_index(k) != schedule.sample(k).0));
        for image in 0..items {
            let mine: Vec<bool> = kinds
                .iter()
                .filter(|(i, _)| *i == image)
                .map(|(_, d)| *d)
                .collect();
            let alternating: Vec<bool> = (0..mine.len()).map(|v| v % 2 == 1).collect();
            assert_eq!(mine, alternating, "image {image} ({kinds:?})");
        }
        assert_eq!(d.path().reference_computations(), cache.len());
    }

    /// AC (d), E7: depth grows the estimate by TAEW2.1 + DA2 (more for Large) and the guard counts
    /// it on the dense and the checkpointed path (synthetic budgets). Mutations: drop `+ extra_gb`
    /// from either projection ⇒ red.
    #[test]
    fn memory_estimate_includes_the_aux_models_on_both_paths() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule();
        let small = perceptual_footprint_gb(&on, 1024, 4);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = perceptual_footprint_gb(&on, 1024, 4);
        assert!(
            small > 0.0 && large - small > 1.0,
            "small {small} large {large}"
        );
        let dense = projected_dense_peak_gb(unified_tokens(512), true);
        let between = |base: f64| (base + large / 2.0) / 0.85;
        assert!(check_budget_with(&on, &[512], true, between(dense), 0.0, false).is_ok());
        assert!(check_budget_with(&on, &[512], true, between(dense), large, false).is_err());
        let ck = checkpointed_baseline_gb(true);
        assert!(check_budget_with(&on, &[512], true, between(ck), 0.0, true).is_ok());
        let err = check_budget_with(&on, &[512], true, between(ck), large, true)
            .expect_err("checkpointed depth job over budget")
            .to_string();
        assert!(err.contains("[depth]"), "{err}");
    }

    /// AC (e): every Anima descriptor declares depth anchoring; a missing TAEW2.1 checkpoint is an
    /// error naming it; the 5-D latent maps to the decoder's NCHW frames.
    #[test]
    fn descriptors_declare_depth_and_missing_decoder_is_named() {
        for d in [
            trainer_descriptor_base(),
            trainer_descriptor_aesthetic(),
            trainer_descriptor_turbo(),
        ] {
            assert!(d.techniques.depth_anchoring, "{}", d.id);
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut c = TrainingConfig::default();
        c.depth_anchoring.schedule = schedule();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taehv"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c).err().unwrap().to_string();
        assert!(err.contains("TAEW2.1"), "{err}");
        let z = Array::zeros::<f32>(&[1, 16, 2, 3, 5]).unwrap();
        assert_eq!(latent_frames_nchw(&z).unwrap().shape(), &[2, 16, 3, 5]);
        assert!(latent_frames_nchw(&Array::zeros::<f32>(&[16, 3, 5]).unwrap()).is_err());
    }
}
