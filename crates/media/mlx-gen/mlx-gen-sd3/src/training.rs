//! LoRA/LoKr **training** on the SD3.5 MMDiT, in pure Rust on mlx-rs (T2 sc-7883, epic 7841) — the
//! MLX-native SD3.5 LoRA-training base. LoRAs train on the `stabilityai/stable-diffusion-3.5-large`
//! MMDiT and apply back at `sd3_5_large` (and family-arch-identical Large-Turbo) inference (the
//! Lens / Z-Image / Krea precedent: same architecture, no base-model gating, family-match suffices).
//!
//! [`Sd3LoraTrainer`] realizes the core [`Trainer`] contract on the real 38-block
//! joint MMDiT, mirroring `KreaRawTrainer` / `mlx_gen_krea` — the model crates don't use mlx-rs's
//! `Module` system (hand-rolled `&self` forwards over raw `Array`s), so training uses the **functional
//! autograd**: the trainable factors live OUTSIDE the model in a [`LoraParams`] map, re-injected each
//! step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s via the shared core
//! seam ([`mlx_gen::train::lora`]), stepped with `keyed_value_and_grad` + the core [`TrainOptimizer`] +
//! `clip_grad_norm`. The injection mirrors the inference reload op-for-op (the [`crate::adapters`]
//! apply path), so the trained adapter round-trips through that loader.
//!
//! ## What is SD3-specific (everything else reuses the family-agnostic core unchanged)
//! - **Flow-match velocity target = `noise − x0`** with **NO sign flip** (the Krea convention, the
//!   OPPOSITE of the Z-Image trainer): the SD3 MMDiT [`forward`](crate::transformer::Sd3Transformer::forward)
//!   returns the RAW un-negated flow-match velocity (the pipeline feeds it to the Euler step
//!   un-negated, [`crate::pipeline::denoise_cfg`]), so the regression target IS the velocity itself.
//!   `x_t = (1 − t)·x0 + t·noise`.
//! - **CRITICAL: the DiT timestep is the diffusers-scale `t·1000`** (the scheduler's
//!   `NUM_TRAIN_TIMESTEPS` — see [`crate::transformer::Sd3Transformer::forward`] and
//!   [`crate::pipeline`] `timestep * NUM_TRAIN_TIMESTEPS`). The trainer samples a normalized
//!   `t ∈ (0,1)` for the noising (`x_t`/target), then SCALES it to `t·1000` at the forward call. This
//!   is the top SD3-vs-Krea/z-image delta (those pass `t` raw); pinned by a test.
//! - **Latents** by the SD3.5 16-ch VAE encode: `preprocess_init_image` (resize + `[−1,1]` NCHW) →
//!   [`Vae::encode`](mlx_gen_z_image::vae::Vae) → `[1, 16, edge/8, edge/8]` (NO temporal axis, NO
//!   pack/transpose — the SD3 latent stays plain NCHW, unlike z-image's packed DiT input).
//! - **Conditioning** is the SD3 triple-TE aggregator's `(context [1,333,4096], pooled [1,2048])`
//!   pair (cached per sample). The three encoders (CLIP-L + CLIP-G + T5-XXL) load in an `Option` so
//!   they can be dropped after caching — they are idle during training (every caption is cached) yet
//!   multi-GB resident (T5-XXL dominates). Loaded Q8 for the trainer (smaller footprint; matches the
//!   deployment quant).
//! - **logit-normal default t-sampling** (the SD3.5 training recipe): `u~U(0,1)`, `t = σ(m + s·Φ⁻¹(u))`
//!   (`m=0, s=1` = the standard logit-normal `σ(N(0,1))`), via the Acklam probit `ndtri` + logistic
//!   ported from `mlx-gen-ideogram/src/scheduler.rs` (the resolution-aware INFERENCE mean-shift is NOT
//!   used). `sigmoid`/`uniform`/`weighted` remain for parity.
//! - **Targets** default to the joint-block attention — image stream `to_q`/`to_k`/`to_v`/`to_out.0`
//!   and text stream `add_q_proj`/`add_k_proj`/`add_v_proj`/`to_add_out` (both joint streams). The
//!   `attn2` (Medium MMDiT-X) targets are enumerable so the Medium trainer (T4 sc-7885) is a
//!   validation story, not a re-architecture; FFN is opt-in.
//!
//! Registered under the **`sd3_5_large`** id (the LoRA-training base; the adapter applies to Large /
//! Large-Turbo inference — family-match, no base-model gating).
//!
//! ## Memory hardening (the Krea sc-7577 / z-image analog — SD3.5-Large at 8.1B is the largest base)
//! - **SDPA-segment checkpointing** is always on in training: the joint SDPA runs inside an
//!   `mlx::checkpoint` so its backward recomputes attention rather than retaining the `[heads, S, S]`
//!   probability matrix. Numerically identical.
//! - **`gradient_checkpointing`** (the SceneWorks toggle) is an opt-in OPTION (LoRA only): each of the
//!   38 joint blocks recomputes its activations in the backward via
//!   [`Sd3Transformer::forward_with_blocks_checkpointed`](crate::transformer::Sd3Transformer::forward_with_blocks_checkpointed),
//!   threading the per-block LoRA factors as explicit checkpoint inputs — MANDATORY for the 8.1B Large
//!   at production resolution. LoKr keeps the dense path (caught by the guard).
//! - **Fail-fast OOM preflight guard** projects the dense first-step peak and refuses (recommending
//!   the toggle) before the minutes-long caching, converting an uncatchable SIGKILL into an actionable
//!   error.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::{self, BucketSchedule};
use mlx_gen::img2img::preprocess_init_image;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, adapter_optimizer_update, average_grads, build_lokr_targets,
    build_lora_targets, LoraParams, TrainAdapter,
};
use mlx_gen::train::loss::{prepared_subject_mask_weight, reduce_loss};
use mlx_gen::train::perceptual::{
    combine_step_loss, AuxAlternation, Parameterization, PerceptualPath, StepPlan,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::train::subject_mask::{CropBox, PreparedSubjectMask};
use mlx_gen::train::tae::TinyDecoderSpec;
use mlx_gen::{
    Error, LoadSpec, Modality, NetworkType, Precision, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::config::Sd3Variant;
use crate::loader;
use crate::pipeline::encode_prompt;
use crate::text::{Sd3Conditioning, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;

/// Registry id for the SD3.5 LoRA-training base (the `stabilityai/stable-diffusion-3.5-large` MMDiT).
/// The trained adapter records `baseModel: sd3_5_large` / `family: sd3` and applies at `sd3_5_large`
/// (and the arch-identical Large-Turbo) inference — the family-match cross-apply, no base-model gating.
pub const SD3_5_LARGE_TRAINER_ID: &str = crate::config::SD3_5_LARGE_ID;

/// Registry id for the SD3.5-**Medium** LoRA-training base (the `stabilityai/stable-diffusion-3.5-medium`
/// MMDiT-X — 24 blocks, hidden 1536, dual-attention `attn2` on the first 13 blocks). A distinct, smaller
/// transformer arch (NOT the Large 38-block plain MMDiT) → its own registered trainer. The trained
/// adapter records `baseModel: sd3_5_medium` and applies back at `sd3_5_medium` inference (family-match,
/// no base-model gating). The trainer **body is identical** to Large's — the `attn2` dual-attention
/// targets are enumerated by the arch-driven [`AdaptableHost`], so the only Medium-specific code is
/// loading the Medium arch (T4 sc-7885).
pub const SD3_5_MEDIUM_TRAINER_ID: &str = crate::config::SD3_5_MEDIUM_ID;

/// The LoKr delta-reconstruction dtype, matching the inference loader so a trained LoKr round-trips
/// through the apply path. bf16 — the family compute dtype.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// The three text encoders are loaded Q8 for the trainer: they are frozen and used only to cache
/// caption conditioning once, then dropped before the train loop (the free pattern). Q8 also matches
/// the deployment quant.
const TRAINER_ENCODER_BITS: i32 = 8;

/// The number of train timesteps the diffusers SD3 scheduler embeds — the MMDiT forward expects
/// `t·NUM_TRAIN_TIMESTEPS` (NOT the raw `t ∈ (0,1)` that z-image/Krea pass). See the module docs.
const NUM_TRAIN_TIMESTEPS: f32 = crate::pipeline::NUM_TRAIN_TIMESTEPS;

/// The default target modules: the joint-block attention projections — image stream
/// (`to_q`/`to_k`/`to_v`/`to_out.0`) + text stream (`add_q_proj`/`add_k_proj`/`add_v_proj`/
/// `to_add_out`). Both streams (the standard SD3 PEFT attention surface). The FFN (`net.0.proj`/
/// `net.2`) and the adaLN modulation linears are reachable as explicit targets but not default.
const DEFAULT_TARGET_MODULES: [&str; 8] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
];

/// The SD3.5 default flow-match timestep distribution: logit-normal (the SD3 training recipe). The
/// trainer uses this when `timestep_type` is unset/`"default"`; `logit_normal` selects it explicitly.
const SD3_DEFAULT_TIMESTEP_TYPE: &str = "logit_normal";

/// Recognized `timestep_type` values [`sample_sigma`] branches on (plus the logit-normal default):
/// the SD3-native `logit_normal` and the cross-family `sigmoid`/`linear`/`uniform`/`weighted` (parity).
const TIMESTEP_TYPES: [&str; 6] = [
    "logit_normal",
    "default",
    "sigmoid",
    "linear",
    "uniform",
    "weighted",
];
/// Recognized `timestep_bias` values [`sample_sigma`] branches on (plus the neutral default).
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
/// Recognized `loss_type` values — `mae`/`l1` → MAE, `mse`/`l2` → the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// `(x_t, target)` for a single sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`,
/// `target = noise − x0` (the velocity the **raw** SD3 MMDiT output is regressed onto — NO sign flip;
/// the SAME sign as Krea, the OPPOSITE of z-image). The DiT timestep is scaled to `t·1000` by the
/// caller (see [`compute_loss_grads`]); `build_batch` works entirely in the un-scaled `t ∈ (0,1)`.
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The production [`Trainer`] for the SD3.5-Large MMDiT: a frozen base (triple TE, MMDiT, 16-ch VAE,
/// CLIP/T5 tokenizers) that caches a captioned dataset to VAE-latents + triple-TE conditioning, then
/// runs the functional-autograd LoRA/LoKr loop with the core runtime glue (LR schedule, gradient
/// accumulation, checkpoint cadence, cancel, progress bands), writing a PEFT adapter that reloads
/// through the inference path ([`crate::adapters::apply_sd3_adapters`]).
pub struct Sd3LoraTrainer {
    descriptor: TrainerDescriptor,
    clip_tokenizer: ClipBpeTokenizer,
    /// Per-encoder CLIP pad ids (sc-9581): CLIP-L pads with eos (49407), bigG with `!` (0). Must
    /// match the inference path so cached training conditioning is padded identically.
    clip_pad: crate::loader::Sd3ClipPad,
    t5_tokenizer: mlx_gen::tokenizer::TextTokenizer,
    /// The three text encoders, in an `Option` so they can be **dropped after caching** — idle during
    /// training (every caption is cached) yet a multi-GB resident (T5-XXL dominates).
    encoders: Option<Sd3TextEncoders>,
    transformer: Sd3Transformer,
    vae: Vae,
    /// The compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

/// The trainer descriptor for `variant` (Large or Medium — the two LoRA-training bases). Both expose
/// the identical LoRA/LoKr image-training capability surface; they differ only in the registry id (and,
/// at load, the MMDiT-X arch the Medium variant carries). Large-Turbo is NOT a training base — it is the
/// distilled inference variant of the Large arch, and a Large-trained adapter applies to it by
/// family-match.
fn trainer_descriptor_for(variant: Sd3Variant) -> TrainerDescriptor {
    TrainerDescriptor {
        id: variant.id(),
        family: "sd3",
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
        // sc-2127 (epic 2123): honors `resolution_buckets` — one cached latent per item per bucket
        // edge, walked through a `BucketSchedule`; the pre-flight guard sizes for the largest edge.
        // sc-24828 (epic 2123): honors `subject_mask_loss` on its one (LoRA/LoKr, dense or
        // block-checkpointed) loss path, with a weight map per (item, bucket) entry.
        // sc-24830 (epic 2123): depth anchoring — the shared decoded-x0 perceptual path (TAESD3
        // decode of the flow x0 estimate → Depth-Anything-V2 → cached round-trip reference) on the
        // dense and block-checkpointed forwards.
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

/// The SD3.5-Large trainer descriptor (the default base; preserves the historical helper name).
fn trainer_descriptor() -> TrainerDescriptor {
    trainer_descriptor_for(Sd3Variant::Large)
}

/// The SD3.5-Medium (MMDiT-X) trainer descriptor.
fn medium_trainer_descriptor() -> TrainerDescriptor {
    trainer_descriptor_for(Sd3Variant::Medium)
}

/// Construct a SD3.5 trainer for `variant` from a snapshot directory (the diffusers multi-component
/// tree). The MMDiT is loaded **dense** (the adapter host) with the variant's arch — Large's 38-block
/// plain MMDiT or Medium's 24-block MMDiT-X (dual-attention `attn2` on the first 13 blocks); the
/// encoders are Q8. `spec.precision` selects the compute dtype (bf16 default / f32 tight-gate); the
/// snapshot ships bf16, so f32 widens it via [`Sd3Transformer::cast_weights`] and bf16 casts the dense
/// load. The train loop is variant-agnostic — the `attn2` targets are enumerated by the arch-driven
/// [`AdaptableHost`], so Medium is covered without a train-loop delta (T4 sc-7885).
pub fn load_trainer_for(spec: &LoadSpec, variant: Sd3Variant) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "sd3 trainer expects a snapshot directory (transformer/ text_encoder{,_2,_3}/ \
                 tokenizer{,_2,_3}/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let arch = variant.arch();
    let clip_tokenizer = loader::load_clip_tokenizer(&root)?;
    let clip_pad = loader::load_clip_pad_ids(&root)?;
    let t5_tokenizer = loader::load_t5_tokenizer(&root)?;
    let mut encoders = loader::load_text_encoders(&root)?;
    encoders.quantize(TRAINER_ENCODER_BITS)?;
    let mut transformer = loader::load_transformer(&root, &arch)?;
    if transformer.compute_dtype() != dtype {
        transformer.cast_weights(dtype)?;
    }
    let vae = loader::load_vae(&root)?;
    Ok(Box::new(Sd3LoraTrainer {
        descriptor: trainer_descriptor_for(variant),
        clip_tokenizer,
        clip_pad,
        t5_tokenizer,
        encoders: Some(encoders),
        transformer,
        vae,
        dtype,
    }))
}

/// Construct the SD3.5-**Large** trainer (the default base). See [`load_trainer_for`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_trainer_for(spec, Sd3Variant::Large)
}

/// Construct the SD3.5-**Medium** (MMDiT-X) trainer. The body is identical to Large's — only the loaded
/// arch differs (24-block MMDiT-X, dual-attention `attn2` on blocks 0..12), and the dual-attention
/// targets are picked up by the arch-driven [`AdaptableHost`]. See [`load_trainer_for`] (T4 sc-7885).
pub fn load_trainer_medium(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_trainer_for(spec, Sd3Variant::Medium)
}

// The trainer registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`. Both
// LoRA-training bases register here — Large (plain MMDiT) and Medium (MMDiT-X dual-attention).
mlx_gen::register_trainer! {
    pub(crate) const LARGE_TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}
mlx_gen::register_trainer! {
    pub(crate) const MEDIUM_TRAINER_REGISTRATION = medium_trainer_descriptor => load_trainer_medium
}

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without a loaded
/// trainer. Rejects an empty dataset, zero rank/steps, an unsupported optimizer, and an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type`. An EMPTY `timestep_type` is accepted (→ the SD3
/// logit-normal default).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err("sd3 trainer: dataset is empty".into());
    }
    if cfg.rank == 0 {
        return Err("sd3 trainer: rank must be > 0".into());
    }
    if cfg.steps == 0 {
        return Err("sd3 trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "sd3 trainer: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    let tt = normalize_cfg(&cfg.timestep_type);
    if !tt.is_empty() && !TIMESTEP_TYPES.contains(&tt.as_str()) {
        return Err(format!(
            "sd3 trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "sd3 trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "sd3 trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for Sd3LoraTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared control-training floor (F-006): a LoRA-only trainer must reject a control-branch
        // request (typed `Unsupported`) rather than silently training a plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
        // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        validate_request(req)?;
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "sd3 trainer: lora_target_modules {:?} matched no adaptable module on the SD3 MMDiT \
                 (defaults are to_q/to_k/to_v/to_out.0 + add_q_proj/add_k_proj/add_v_proj/to_add_out \
                 on the joint blocks)",
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

impl Sd3LoraTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_request(req)?;
        let cfg = &req.config;

        let target_paths = resolve_target_paths(&self.transformer, cfg);
        if target_paths.is_empty() {
            return Err(format!(
                "sd3 trainer: lora_target_modules {:?} matched no adaptable module on the SD3 MMDiT",
                cfg.lora_target_modules
            )
            .into());
        }

        // The MMDiT compute dtype is fixed at load (`spec.precision`); enforce `train_dtype` against
        // it (never a silent no-op). The common case (config default bf16 + LoadSpec default Bf16)
        // matches, so this only fires on an explicit f32-vs-bf16 mismatch.
        let want_bf16 = {
            let t = cfg.train_dtype.trim();
            t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == Dtype::Bfloat16;
        if want_bf16 != loaded_bf16 {
            return Err(format!(
                "sd3 trainer: train_dtype '{}' does not match the loaded precision ({}). Load the \
                 trainer with {} to train at that dtype.",
                cfg.train_dtype,
                if loaded_bf16 { "bf16" } else { "f32" },
                if want_bf16 {
                    "Precision::Bf16"
                } else {
                    "Precision::Fp32"
                }
            )
            .into());
        }
        let compute_dtype = self.dtype;
        // bf16 mixed precision: cast the folded LoRA residual to the activation dtype so the adapted
        // Linear stays bf16 (else it silently re-promotes the chain to f32). f32 → no cast.
        let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);

        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one training edge per resolution bucket (just `[resolution]` when buckets are
        // off). The pre-flight guard sizes for the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let edge = preflight_edge(&edges);

        // T2 — fail-fast pre-flight memory guard (the Krea/z-image analog). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`;
        // at high resolution that working set can exceed unified memory and the OS hard-kills the
        // worker with an UNCATCHABLE SIGKILL. We predict it and refuse up front with a catchable,
        // actionable error BEFORE the (minutes-long) latent caching, UNLESS the run will
        // block-checkpoint (LoRA + the toggle). LoKr always takes the dense path, so it is guarded.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        // F-035: select the preflight constants by variant — Medium (2.5B/24-block) was refused via
        // the Large (8.1B/38-block) constants (~3× overstated). The trainer is built for one
        // variant; derive it from the registered descriptor id.
        let variant = if self.descriptor.id == crate::config::SD3_5_MEDIUM_ID {
            Sd3Variant::Medium
        } else {
            Sd3Variant::Large
        };
        // Epic 2123 E7: the training-time aux models (TAESD3 + Depth-Anything-V2) count against the
        // budget on BOTH paths; one cached reference per (item, bucket) entry, sized at the largest
        // edge. A checkpointed run with no aux models stays unguarded (see the guard).
        let aux_gb = perceptual_footprint_gb(cfg, edge, req.items.len() * edges.len());
        preflight_memory_guard(edge, want_bf16, variant, aux_gb, will_checkpoint)?;

        // Epic 2123 depth anchoring: load the frozen decoder + aux models before the caching pass,
        // so a missing/corrupt aux checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: VAE-latents + triple-TE conditioning into memory ---
        on_progress(TrainingProgress::LoadingModel); // base is already resident from load_trainer
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). Each entry: clean latent,
        // triple-TE conditioning, and (subject-masked loss, sc-24828) that bucket's latent
        // loss-weight map — `None` when the technique is off.
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
            // sc-24828: the item's subject mask is read + checked once, resampled per bucket.
            let mask = PreparedSubjectMask::load_if_enabled(
                "sd3 trainer",
                item,
                cfg.subject_mask_loss.as_ref(),
            )?;
            let encoders = self.encoders.as_ref().ok_or_else(|| {
                Error::Msg(
                    "sd3 trainer: text encoders already freed (caching after train loop)".into(),
                )
            })?;
            let cond = encode_prompt(
                encoders,
                &self.clip_tokenizer,
                self.clip_pad,
                &self.t5_tokenizer,
                &item.caption,
            )?;
            eval([&cond.context, &cond.pooled])?;
            // The caption conditioning is resolution-independent: encode it once, then one latent per
            // bucket edge (the MMDiT derives its patch grid from the latent's own shape).
            for (x0, mask_weight) in encode_buckets(&edges, mask.as_ref(), |edge| {
                encode_init_latents(&self.vae, &img, edge) // [1, 16, edge/8, edge/8]
            })? {
                let cond = Sd3Conditioning {
                    context: cond.context.clone(),
                    pooled: cond.pooled.clone(),
                };
                cache.push((x0, cond, mask_weight));
            }
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("sd3 trainer: no usable dataset items".into());
        }

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference is computed exactly once
        // per job, here, before the loop.
        if let Some(path) = perceptual.as_mut() {
            prepare_perceptual_references(path, &cache)?;
        }

        // Every caption is cached now — free the three encoders (T5-XXL dominates) and evict buffers
        // before the train loop, reclaiming that resident for the 8.1B MMDiT working set.
        self.encoders = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
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

        // T2 — gradient checkpointing. Collect, per joint block, the adapter-routable LOCAL paths
        // trained on it (e.g. `"attn.to_q"`), in trained-file order — the factors a checkpoint segment
        // threads as explicit inputs. Only `transformer_blocks.*` targets are checkpointed; any global
        // targets (`context_embedder`/`proj_out`) train dense through `self`.
        let n_layers = self.transformer.num_blocks();
        let mut block_local_targets: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        block_local_targets[i].push(local.to_string());
                    }
                }
            }
        }
        // Opt-in OPTION (the SceneWorks "Gradient Checkpointing" toggle), never auto-forced — a run
        // that would OOM is caught instead by the pre-flight guard above. LoRA only — LoKr falls back
        // to the dense path.
        let use_checkpoint =
            matches!(adapter, TrainAdapter::Lora { .. }) && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing is ALWAYS on in training. When whole-block checkpointing is on,
        // the per-block SDPA flag goes OFF (the block recompute already covers attention).
        self.transformer.set_sdpa_checkpoint(!use_checkpoint);

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
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items for
        // a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: per-image, per-update alternation keys for the perceptual losses, keyed on
        // the real dataset item (not the (item, bucket) entry). A resumed run replays the skipped
        // prefix so the phase matches.
        let mut alternation = perceptual
            .as_ref()
            .map(|_| AuxAlternation::new(cache.len() / edges.len(), accum));
        if let Some(alt) = alternation.as_mut() {
            for step in 1..=start_step {
                alt.key(step, schedule.sample((step - 1) as usize).0);
            }
        }
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
                perceptual.as_mut().zip(alternation.as_mut()),
                step,
                mae,
                compute_dtype,
                lora_dtype,
                checkpoint_blocks,
            )?;
            last_loss = losses.total;
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
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                adapter.save(&params, alpha, rank, cfg.decompose_factor, "", &ckpt)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before a single step completed (`steps == 0` is rejected by `validate`): the
        // factors are still the `B = 0` no-op init. Surface the cancellation rather than writing a
        // valid-looking identity adapter as a trained artifact.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save final adapter (the diffusers/PEFT format the inference apply path loads) ---
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

/// Number of caption tokens assumed by the pre-flight projection. The SD3 unified sequence is
/// `img_len + ctx_len` and the context is a fixed 333 tokens (77 CLIP + 256 T5).
const PREFLIGHT_TXT_TOKENS: f64 = 333.0;

/// `(weights, linear, quad)` conservative initial constants for [`projected_dense_peak_gb`], by
/// variant. Large = the 8.1B 38-block MMDiT (the historical single set). F-035: Medium is the 2.5B
/// 24-block MMDiT-X — using Large's constants refused valid Medium dense runs with a ~3× overstated
/// projection. The Medium resident base + per-token linear term scale ~24/38 of Large (the quad term
/// is per-block SDPA-segment, unchanged); still conservative-initial estimates, refit from a sweep.
const PREFLIGHT_F32_LARGE: (f64, f64, f64) = (32.0, 1.20e-2, 3.0e-7);
const PREFLIGHT_BF16_LARGE: (f64, f64, f64) = (16.0, 6.0e-3, 1.5e-7);
const PREFLIGHT_F32_MEDIUM: (f64, f64, f64) = (10.0, 7.6e-3, 3.0e-7);
const PREFLIGHT_BF16_MEDIUM: (f64, f64, f64) = (5.0, 3.8e-3, 1.5e-7);

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the
/// unified token count `s = img_len + ctx_len`. The structure follows the Krea/z-image
/// `weights + linear·s + quad·s²` decomposition: the constant is the resident MMDiT base (Large bf16
/// ~16 GB / f32 ~32 GB; Medium ~5 / ~10 GB; the encoders are freed before the train loop), the linear
/// term is the per-token activations across the joint blocks, and the quadratic term is the seq²
/// attention transient — demoted to a single block's backward transient by the always-on SDPA-segment
/// checkpointing. F-035: parameterized by variant — the constants differ between the 8.1B Large (38
/// blocks) and 2.5B Medium (24 blocks) MMDiT.
///
/// **These constants are a CONSERVATIVE INITIAL ESTIMATE** — they err toward refusing borderline runs
/// (recommending Gradient Checkpointing) rather than allowing a SIGKILL, and are to be refit from a
/// real-weight sweep. `projection_is_monotonic_and_conservative` pins the shape.
fn projected_dense_peak_gb(s: f64, bf16: bool, variant: Sd3Variant) -> f64 {
    let (f32_c, bf16_c) = match variant {
        Sd3Variant::Large | Sd3Variant::LargeTurbo => (PREFLIGHT_F32_LARGE, PREFLIGHT_BF16_LARGE),
        Sd3Variant::Medium => (PREFLIGHT_F32_MEDIUM, PREFLIGHT_BF16_MEDIUM),
    };
    let c = if bf16 { bf16_c } else { f32_c };
    c.0 + c.1 * s + c.2 * s * s
}

/// The edge the pre-flight guard sizes for: the LARGEST bucket edge (epic 2123 E7) — the dense first
/// step's working set is set by the biggest latent the run will ever train on, whichever bucket the
/// schedule happens to draw first.
fn preflight_edge(edges: &[u32]) -> u32 {
    edges.iter().copied().max().unwrap_or(0)
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. Consulted when gradient
/// checkpointing is OFF, and — whenever the training-time aux models add memory (`extra_gb`,
/// epic 2123 E7) — when it is on too.
fn preflight_memory_guard(
    edge: u32,
    bf16: bool,
    variant: Sd3Variant,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    check_preflight_budget_with_aux(edge, bf16, budget_gb, variant, extra_gb, checkpointed)
}

/// The pure guard logic (no MLX global state, so it is unit-testable): refuse if the projected dense
/// first-step peak exceeds `budget_gb × 0.85`. `edge` is the bucketed training edge; the SD3 unified
/// token count is `(edge/16)²` (latent /8, patch 2) plus the fixed 333-token context.
#[cfg(test)]
fn check_preflight_budget(
    edge: u32,
    bf16: bool,
    budget_gb: f64,
    variant: Sd3Variant,
) -> Result<()> {
    check_preflight_budget_with_aux(edge, bf16, budget_gb, variant, 0.0, false)
}

/// [`check_preflight_budget`] plus the epic-2123 aux models (`extra_gb`, [`perceptual_footprint_gb`])
/// on top of the MMDiT projection. With `checkpointed`, the projection is the resident base
/// (`projected_dense_peak_gb(0)`: no fitted checkpointed curve exists, so the resident MMDiT is the
/// lower bound the aux models stack on), and a checkpointed run with no aux models is not guarded.
fn check_preflight_budget_with_aux(
    edge: u32,
    bf16: bool,
    budget_gb: f64,
    variant: Sd3Variant,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    if checkpointed && extra_gb <= 0.0 {
        return Ok(());
    }
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + PREFLIGHT_TXT_TOKENS;
    let projected = if checkpointed {
        projected_dense_peak_gb(0.0, bf16, variant)
    } else {
        projected_dense_peak_gb(s, bf16, variant)
    } + extra_gb;
    let safe = budget_gb * 0.85;
    if projected > safe && checkpointed {
        return Err(format!(
            "sd3 trainer: a checkpointed training step at resolution {edge} with the \
             depth-anchoring models (~{extra_gb:.1} GB for the tiny decoder and Depth-Anything-V2) \
             needs at least ~{projected:.0} GB, exceeding this machine's ~{safe:.0} GB safe budget \
             ({budget_gb:.0} GB MLX limit × 0.85). Use a smaller depth model or reduce the training \
             resolution."
        )
        .into());
    }
    if projected > safe {
        return Err(format!(
            "sd3 trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the \
             OS would hard-kill the worker (SIGKILL) at the first step with no recoverable error. \
             Enable Gradient Checkpointing (recomputes block activations in the backward) or reduce \
             the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default [`DEFAULT_TARGET_MODULES`]) to full dotted
/// paths by matching them against every adapter-routable module on the MMDiT — the same suffix-match
/// PEFT's `LoraConfig(target_modules=…)` does. The DEFAULT set is restricted to the joint
/// `transformer_blocks`; an explicit `lora_target_modules` matches anywhere (incl. the global
/// projections).
fn resolve_target_paths(transformer: &Sd3Transformer, cfg: &TrainingConfig) -> Vec<String> {
    let default = cfg.lora_target_modules.is_empty();
    let suffixes: Vec<String> = if default {
        DEFAULT_TARGET_MODULES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    AdaptableHost::adaptable_paths(transformer)
        .into_iter()
        .filter(|path| {
            (!default || path.starts_with("transformer_blocks."))
                && suffixes
                    .iter()
                    .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = image::open(path)
        .map_err(|e| Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Encode a center-cropped square image into an SD3.5 training latent `[1, 16, edge/8, edge/8]` — the
/// 16-ch VAE encode the MMDiT predicts in: `preprocess_init_image` (resize + `[−1,1]` NCHW) →
/// [`Vae::encode`] (16-ch latent in the scaled/shifted latent space, `[1,16,edge/8,edge/8]`). Unlike
/// z-image's packed DiT input, the SD3 latent stays plain NCHW — NO pack/transpose, NO temporal axis.
fn encode_init_latents(vae: &Vae, image: &Image, edge: u32) -> Result<Array> {
    let pre = preprocess_init_image(image, edge, edge)?; // [1, 3, edge, edge] in [-1, 1]
    vae.encode(&pre) // [1, 16, edge/8, edge/8]
}

/// Sample a normalized flow-match timestep `t ∈ [1e-3, 1−1e-3]`. The SD3.5 default is **logit-normal**
/// (`u~U(0,1)`, `t = σ(m + s·Φ⁻¹(u))`, `m=0, s=1`; the Acklam probit + logistic ported from
/// `mlx-gen-ideogram/src/scheduler.rs`, WITHOUT the resolution-aware inference mean-shift). The
/// cross-family `sigmoid`/`uniform`/`linear`/`weighted` (a faithful port of the SceneWorks
/// `sample_training_timestep`) remain for parity; bias `high` → `√t`, `low` → `t²`. Deterministic in
/// `seed`. An empty / `"default"` `timestep_type` selects the logit-normal default.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = {
        let n = normalize_cfg(timestep_type);
        if n.is_empty() || n == "default" {
            SD3_DEFAULT_TIMESTEP_TYPE.to_string()
        } else {
            n
        }
    };
    let t = match ttype.as_str() {
        "logit_normal" => {
            // u ~ U(0,1) → t = σ(m + s·ndtri(u)); m=0, s=1 (= σ(N(0,1)), the standard logit-normal).
            let u =
                random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>() as f64;
            logistic(0.0 + 1.0 * ndtri(u)) as f32
        }
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed ^ 0x9E37_79B9)?;
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
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// Logistic (sigmoid) `σ(y) = 1/(1+e^-y)`. Ported from `mlx-gen-ideogram/src/scheduler.rs`.
fn logistic(y: f64) -> f64 {
    1.0 / (1.0 + (-y).exp())
}

/// Inverse normal CDF (probit), Acklam's rational approximation (|err| ≲ 1.15e-9). Endpoints map to
/// ±∞ so the logistic squashes them back into (0,1) before the trainer clamp. Ported from
/// `mlx-gen-ideogram/src/scheduler.rs` (the T1-designated reference).
fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), run the
/// SD3 MMDiT at the **`t·1000`-scaled** timestep, regress the **raw** `forward()` velocity onto
/// `noise − x0` (NO sign flip), return `(loss, grads)`. `dtype` is the training compute dtype; the
/// LoRA factors are cast inside the traced install (`lora_dtype`), so the MMDiT graph runs at `dtype`;
/// the noising math, loss, and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-joint-block LOCAL LoRA target paths and switches the
/// forward to the gradient-checkpointed path. `None` runs the dense forward.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut Sd3Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Sd3Conditioning,
    t: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> Result<(f32, LoraParams)> {
    let (losses, grads) = compute_step_loss_grads(
        transformer,
        params,
        adapter,
        alpha,
        rank,
        x0,
        cond,
        t,
        noise,
        mae,
        mask_weight,
        dtype,
        lora_dtype,
        checkpoint_blocks,
        None,
    )?;
    Ok((losses.total, grads))
}

/// The per-step loss breakdown [`compute_step_loss_grads`] returns (epic 2123 E8).
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
    /// The step's reference key — its (item, bucket) cache entry.
    entry: usize,
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed (it contributes zero) and the loss is the weighted perceptual term
/// on the model's x0 estimate `x0 = x_t − t·v` (the raw SD3 velocity regresses `noise − x0` with
/// `x_t = (1−t)·x0 + t·noise`; the latent is already the plain NCHW, SD3-normalized
/// `(z − shift)·scale` latent TAESD3 decodes). With `aux = None` (or a diffusion-only plan with no
/// aux loss) the traced graph is exactly the pre-epic-2123 one; both the dense and the
/// block-checkpointed forwards carry the aux term.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    transformer: &mut Sd3Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Sd3Conditioning,
    t: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpoint_blocks: Option<&[Vec<String>]>,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (x_t_f32, target) = build_batch(x0, noise, t)?;
    let mask_weight = mask_weight.cloned();
    let x_t = x_t_f32.as_dtype(dtype)?; // no-op in f32 mode
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    // CRITICAL: the SD3 MMDiT embeds the diffusers-scale timestep `t·1000` (NUM_TRAIN_TIMESTEPS),
    // NOT the raw `t ∈ (0,1)`. `build_batch` above used the un-scaled `t` for the noising; the forward
    // gets `t·1000`. (z-image/Krea pass `t` raw — this is the top SD3 parity delta.)
    let timestep = Array::from_slice(&[t * NUM_TRAIN_TIMESTEPS], &[1]);
    let context = cond.context.clone();
    let pooled = cond.pooled.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets so the dense path (and any non-checkpointed global targets) train
        // through ordinary autograd; on the checkpointed path the joint-block adapters installed here
        // are replaced inside each checkpoint segment by the explicit-input factors.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        // Training drives the compute dtype EXPLICITLY (`dtype` = the bf16 train dtype, or f32). The
        // inference `forward` is f32-pinned and must not be used here; both training paths take the
        // explicit-dtype seam so bf16 training runs the heavy matmuls in bf16.
        let v = match checkpoint_blocks {
            Some(locals) => transformer
                .forward_with_blocks_checkpointed(
                    &x_t, &context, &pooled, &timestep, &p, locals, alpha, dtype,
                )
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward_with(&x_t, &context, &pooled, &timestep, dtype)
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
                // x0 estimate in f32 from the raw velocity: x0 = x_t − t·v.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma: t }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.entry, &x0_hat)
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

/// SD3's x0 decoder for the shared aux-loss builder (epic 2123 E8): TAESD3 (`madebyollin/taesd3`,
/// the SD3 16-channel latent API).
fn taesd3_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Tiny {
        name: "TAESD3",
        config: TinyDecoderSpec::taesd3(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder: `None` when no aux loss is
/// enabled (nothing loads; every step is the plain diffusion step).
fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label: "sd3 trainer",
            decoder: taesd3_decoder(),
            latent_lpips: Some(gen_core::train::LatentLpipsFamily::Sd3),
        },
    )
}

/// Extra training memory (GB) the enabled aux losses add at the bucketed `edge` with `entries`
/// cached references (epic 2123 E7). `0` when no aux loss is enabled.
fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &taesd3_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// One `train_impl` cache entry: the clean NCHW latent, its conditioning, and its subject-mask
/// weight (sc-24828; `None` when off).
type CacheEntry = (Array, Sd3Conditioning, Option<Array>);

/// Compute every cache entry's perceptual reference once (its clean NCHW latent), keyed per
/// (item, bucket) entry — each bucket's latent decodes to its own size.
fn prepare_perceptual_references(path: &mut PerceptualPath, cache: &[CacheEntry]) -> Result<()> {
    for (i, (x0, _, _)) in cache.iter().enumerate() {
        path.ensure_reference(i, x0)?;
    }
    Ok(())
}

/// One training micro-step on the 1-based `step`: pick the step's cached (item, bucket) entry,
/// sample its `t` and noise (seeded, exactly as before epic 2123), plan the step's loss terms through
/// the perceptual path (when one is configured: the alternation key comes from the item's own update
/// count, and an aux-only step trains at `t` remapped into the loss window), and run
/// [`compute_step_loss_grads`]. With no perceptual path every step is the plain diffusion step,
/// bit-identical to the pre-epic-2123 loop.
#[allow(clippy::too_many_arguments)]
fn run_train_step(
    transformer: &mut Sd3Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    cfg: &TrainingConfig,
    cache: &[CacheEntry],
    schedule: &BucketSchedule,
    perceptual: Option<(&mut PerceptualPath, &mut AuxAlternation)>,
    step: u32,
    mae: bool,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> Result<(StepLosses, LoraParams)> {
    let k = (step - 1) as usize;
    let (item, _bucket) = schedule.sample(k);
    let entry = schedule.cache_index(k);
    let (x0, cond, mask_weight) = &cache[entry];
    let mut t = sample_sigma(
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
    let plan;
    let aux = match perceptual {
        Some((path, alternation)) => {
            // Normally a no-op (references were computed once, before the loop).
            path.ensure_reference(entry, x0)?;
            plan = path.plan(alternation.key(step, item), entry, t)?;
            t = plan.noise_level;
            let path: &PerceptualPath = path;
            Some(AuxStep {
                path,
                plan: &plan,
                entry,
            })
        }
        None => None,
    };
    compute_step_loss_grads(
        transformer,
        params,
        adapter,
        cfg.alpha,
        cfg.rank as f32,
        x0,
        cond,
        t,
        &noise,
        mae,
        mask_weight.as_ref(),
        dtype,
        lora_dtype,
        checkpoint_blocks,
        aux,
    )
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
                "sd3 trainer",
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
    use mlx_gen::CancelFlag;
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
            Ok(Array::zeros::<f32>(&[1, 16, g, g])?)
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

    fn base_config() -> TrainingConfig {
        TrainingConfig {
            rank: 8,
            steps: 10,
            ..Default::default()
        }
    }

    fn req_with(config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items: vec![mlx_gen::TrainingItem {
                image_path: PathBuf::from("/tmp/x.png"),
                caption: "a swatch".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            }],
            config,
            output_dir: PathBuf::from("/tmp/sd3_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_large_base_id() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "sd3_5_large");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
        // sc-24828: the one loss path (dense + block-checkpointed) reduces through the subject-mask
        // weight; the latent is plain NCHW `[1, 16, H, W]` (no packing), so the weight takes its shape.
        assert!(d.techniques.subject_mask_loss);
        assert!(medium_trainer_descriptor().techniques.subject_mask_loss);
    }

    #[test]
    fn medium_descriptor_is_the_medium_base_id() {
        // T4 (sc-7885): the Medium (MMDiT-X) trainer is a distinct registered base — same capability
        // surface as Large, different id. Large-Turbo is NOT a training base (it shares Large's arch).
        let d = medium_trainer_descriptor();
        assert_eq!(d.id, "sd3_5_medium");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
        // The two training bases are distinct ids.
        assert_ne!(d.id, trainer_descriptor().id);
    }

    #[test]
    fn both_training_bases_reachable_via_registry() {
        // Both the Large and Medium trainer constants compose into the family catalog.
        let ids: Vec<&str> = crate::provider_registry()
            .unwrap()
            .trainers()
            .copied()
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(
            ids.contains(&SD3_5_LARGE_TRAINER_ID),
            "large trainer id not registered (ids: {ids:?})"
        );
        assert!(
            ids.contains(&SD3_5_MEDIUM_TRAINER_ID),
            "medium trainer id not registered (ids: {ids:?})"
        );
    }

    #[test]
    fn validate_rejects_empty_dataset_and_zero_rank_steps() {
        let mut r = req_with(base_config());
        r.items.clear();
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("dataset is empty"));

        let r = req_with(TrainingConfig {
            rank: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("rank"));

        let r = req_with(TrainingConfig {
            steps: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    #[test]
    fn validate_accepts_gradient_checkpointing_and_logit_normal() {
        let r = req_with(TrainingConfig {
            gradient_checkpointing: true,
            timestep_type: "logit_normal".into(),
            ..base_config()
        });
        assert!(validate_request(&r).is_ok());
        // The SD3 default (empty timestep_type) is accepted (→ logit-normal).
        let r = req_with(TrainingConfig {
            timestep_type: "".into(),
            ..base_config()
        });
        assert!(validate_request(&r).is_ok());
    }

    #[test]
    fn validate_rejects_unrecognized_optimizer_timestep_loss() {
        for cfg in [
            TrainingConfig {
                optimizer: "nope".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_type: "bogus".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_bias: "sideways".into(),
                ..base_config()
            },
            TrainingConfig {
                loss_type: "huber".into(),
                ..base_config()
            },
        ] {
            assert!(validate_request(&req_with(cfg)).is_err());
        }
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Logit-Normal".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_velocity_with_no_sign_flip() {
        // target = noise − x0 (the RAW SD3 MMDiT velocity; the SAME sign as Krea, OPPOSITE z-image),
        // and x_t = (1−t)·x0 + t·noise. NO sign flip on the target. (The DiT timestep is scaled to
        // t·1000 at the forward call — see `t1000_timestep_scaling_pins_diffusers_scale`.)
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let (x_t, target) = build_batch(&x0, &noise, 0.25).unwrap();
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]); // noise − x0, NOT x0 − noise
        let xt = x_t.as_slice::<f32>();
        for (got, want) in xt.iter().zip([1.75f32, 3.25, 4.75].iter()) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn t1000_timestep_scaling_pins_diffusers_scale() {
        // The trainer scales the normalized flow-match `t ∈ (0,1)` to the diffusers `t·1000` the SD3
        // MMDiT embeds (NUM_TRAIN_TIMESTEPS), distinct from z-image/Krea which pass `t` raw. This pins
        // that constant and the scaling the forward call applies (the top SD3 parity risk).
        assert_eq!(NUM_TRAIN_TIMESTEPS, 1000.0);
        for t in [1e-3f32, 0.25, 0.5, 0.9999] {
            let scaled = t * NUM_TRAIN_TIMESTEPS;
            assert!((scaled - t * 1000.0).abs() < 1e-3, "t={t} scaled={scaled}");
        }
    }

    #[test]
    fn logit_normal_sampler_is_deterministic_in_range_and_monotone() {
        // Deterministic in seed; in (1e-3, 1−1e-3); and the bias monotone (high pushes toward 1, low
        // toward 0). Also: the median (u=0.5 path is stochastic, so check the distribution mean ≈ 0.5
        // for the standard logit-normal m=0,s=1 over many seeds).
        for bias in ["balanced", "high", "low"] {
            let a = sample_sigma("logit_normal", bias, 42).unwrap();
            let b = sample_sigma("logit_normal", bias, 42).unwrap();
            assert_eq!(a, b, "logit_normal/{bias} must be deterministic in seed");
            assert!(
                (1e-3..=1.0 - 1e-3).contains(&a),
                "logit_normal/{bias} t={a} out of range"
            );
        }
        assert!(
            sample_sigma("logit_normal", "high", 7).unwrap()
                > sample_sigma("logit_normal", "low", 7).unwrap()
        );
        // Standard logit-normal σ(N(0,1)) has mean ≈ 0.5 (symmetric about 0.5); pin it loosely.
        let mut sum = 0.0f64;
        let n = 400u64;
        for s in 0..n {
            sum +=
                sample_sigma("logit_normal", "balanced", s.wrapping_mul(7919) + 1).unwrap() as f64;
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.06,
            "logit-normal mean {mean} not ≈ 0.5"
        );
        // Empty / "default" timestep_type routes to the logit-normal default (same as named).
        assert_eq!(
            sample_sigma("", "balanced", 123).unwrap(),
            sample_sigma("logit_normal", "balanced", 123).unwrap()
        );
        assert_eq!(
            sample_sigma("default", "balanced", 123).unwrap(),
            sample_sigma("logit_normal", "balanced", 123).unwrap()
        );
    }

    #[test]
    fn ndtri_matches_known_probit_values() {
        // Φ⁻¹(0.5)=0, Φ⁻¹(0.975)≈1.959964, Φ⁻¹(0.025)≈−1.959964 (the reference probit; the Acklam
        // approximation is |err| ≲ 1.15e-9).
        assert!(ndtri(0.5).abs() < 1e-6);
        assert!((ndtri(0.975) - 1.959_963_984_540_054).abs() < 1e-4);
        assert!((ndtri(0.025) + 1.959_963_984_540_054).abs() < 1e-4);
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri(1.0), f64::INFINITY);
    }

    #[test]
    fn sigmoid_parity_sampler_in_range_and_deterministic() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                let b = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, b, "{kind}/{bias} must be deterministic");
                assert!((1e-3..=1.0 - 1e-3).contains(&a), "{kind}/{bias} t={a} oob");
            }
        }
    }

    #[test]
    fn default_target_modules_are_both_joint_streams() {
        // The default training surface is the standard SD3 PEFT attention surface: image + text stream
        // attention projections (both joint streams).
        assert_eq!(
            DEFAULT_TARGET_MODULES,
            [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out"
            ]
        );
    }

    #[test]
    fn preflight_guard_fires_over_budget_and_passes_under() {
        // A 16 GB-class budget (safe ≈ 13.6 GB): the 16 GB bf16 MMDiT base alone exceeds it → dense
        // 1024 must be refused with an actionable error recommending Gradient Checkpointing.
        let err = check_preflight_budget(1024, true, 16.0, Sd3Variant::Large)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "got: {err}");
        assert!(
            err.contains("1024"),
            "error should name the resolution: {err}"
        );
        // A 128 GB-class budget (safe ≈ 108 GB) comfortably fits dense 1024 in both dtypes.
        assert!(check_preflight_budget(1024, true, 128.0, Sd3Variant::Large).is_ok());
        assert!(check_preflight_budget(1024, false, 128.0, Sd3Variant::Large).is_ok());
    }

    #[test]
    fn both_descriptors_declare_resolution_buckets() {
        // sc-2127: the shared technique floor only lets `resolution_buckets` through when declared.
        assert!(trainer_descriptor().techniques.resolution_buckets);
        assert!(medium_trainer_descriptor().techniques.resolution_buckets);
    }

    #[test]
    fn preflight_sizes_for_the_largest_bucket_edge() {
        let rb = |resolution, repeats| gen_core::ResolutionBucket {
            resolution,
            repeats,
        };
        // Buckets off: the guard edge is the single legacy edge.
        let mut cfg = TrainingConfig {
            resolution: 512,
            ..base_config()
        };
        assert_eq!(preflight_edge(&bucket_edges(&cfg)), 512);
        // Buckets [512, 1024] with `resolution` 512: the guard must size for 1024 (epic 2123 E7).
        cfg.resolution_buckets = vec![rb(512, 16), rb(1024, 1)];
        let edges = bucket_edges(&cfg);
        assert_eq!(edges, vec![512, 1024]);
        assert_eq!(preflight_edge(&edges), 1024);
        // A 40 GB-class budget (safe 34 GB): dense bf16 Large fits 512 (~24 GB) but not 1024
        // (~48 GB) — so the bucketed run is refused even though `resolution` alone would pass.
        assert!(check_preflight_budget(512, true, 40.0, Sd3Variant::Large).is_ok());
        assert!(
            check_preflight_budget(preflight_edge(&edges), true, 40.0, Sd3Variant::Large).is_err()
        );
    }

    #[test]
    fn reachable_via_trainer_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .trainers()
                .copied()
                .any(|r| (r.descriptor)().id == SD3_5_LARGE_TRAINER_ID),
            "trainer id {SD3_5_LARGE_TRAINER_ID} not registered"
        );
    }

    #[test]
    fn projection_is_monotonic_and_conservative() {
        for bf16 in [false, true] {
            let (s512, s768, s1024) = (1024.0 + 333.0, 2304.0 + 333.0, 4096.0 + 333.0);
            assert!(
                projected_dense_peak_gb(s512, bf16, Sd3Variant::Large)
                    < projected_dense_peak_gb(s768, bf16, Sd3Variant::Large)
            );
            assert!(
                projected_dense_peak_gb(s768, bf16, Sd3Variant::Large)
                    < projected_dense_peak_gb(s1024, bf16, Sd3Variant::Large)
            );
        }
        assert!(
            projected_dense_peak_gb(4429.0, true, Sd3Variant::Large)
                < projected_dense_peak_gb(4429.0, false, Sd3Variant::Large)
        );
        // The Large bf16 base (no tokens) is ~the resident 8.1B MMDiT weights (≥ 14 GB). Medium's
        // base is lower (~5 GB) — F-035 parameterized this by variant.
        assert!(projected_dense_peak_gb(0.0, true, Sd3Variant::Large) >= 14.0);
        assert!(projected_dense_peak_gb(0.0, true, Sd3Variant::Medium) < 14.0);
        assert!(projected_dense_peak_gb(0.0, true, Sd3Variant::Medium) >= 4.0);
    }

    /// sc-24828: the subject-mask weight reaches BOTH backward paths (dense + block-checkpointed) of
    /// [`compute_loss_grads`] on a tiny synthetic MMDiT (random weights for every converter key, so no
    /// real weights). An all-ones map equals the unweighted loss; an all-zero map gives loss exactly 0
    /// and all-zero adapter grads on both paths; a half map lands strictly between and agrees across
    /// paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        use mlx_gen::train::loss::subject_mask_weight;
        let arch = crate::config::Sd3Arch {
            num_layers: 2,
            head_dim: 8,
            num_heads: 2,
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            joint_attention_dim: 24,
            pooled_projection_dim: 20,
            caption_projection_dim: 16,
            pos_embed_max_size: 8,
            time_proj_dim: 16,
            dual_attention_layers: 0,
        };
        let key = random::key(7).unwrap();
        let scale = Array::from_slice(&[0.02f32], &[1]);
        let mut w = mlx_gen::weights::Weights::empty();
        for e in crate::convert::expected_transformer_tensors(&arch) {
            let shape: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
            let t = multiply(
                random::normal::<f32>(&shape, None, None, Some(&key)).unwrap(),
                &scale,
            )
            .unwrap();
            w.insert(e.key, t);
        }
        let mut dit = Sd3Transformer::from_weights(&w, &arch).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        assert!(!target_paths.is_empty());
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
        // Non-zero factors on both sides (the LoRA up-projection inits at zero, which would zero the
        // down-projection grads trivially).
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
        let adapter = TrainAdapter::Lora { targets };
        let mut locals: Vec<Vec<String>> = vec![Vec::new(); dit.num_blocks()];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    locals[i].push(local.to_string());
                }
            }
        }
        assert!(locals.iter().any(|l| !l.is_empty()));
        let shape = [1i32, 16, 4, 4];
        let x0 = random::normal::<f32>(&shape, None, None, Some(&random::key(1).unwrap())).unwrap();
        let noise =
            random::normal::<f32>(&shape, None, None, Some(&random::key(2).unwrap())).unwrap();
        let cond = Sd3Conditioning {
            context: random::normal::<f32>(&[1, 5, 24], None, None, Some(&random::key(3).unwrap()))
                .unwrap(),
            pooled: random::normal::<f32>(&[1, 20], None, None, Some(&random::key(4).unwrap()))
                .unwrap(),
        };
        let map = |v: &[f32]| subject_mask_weight(v, 4, 4, &shape).unwrap();
        let mut run = |weight: Option<&Array>, ckpt: bool| {
            let (l, g) = compute_loss_grads(
                &mut dit,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &cond,
                0.5,
                &noise,
                false,
                weight,
                Dtype::Float32,
                None,
                ckpt.then_some(locals.as_slice()),
            )
            .unwrap();
            eval(g.values()).unwrap();
            (l, g)
        };
        let (plain, _) = run(None, false);
        let ones = map(&[1.0; 16]);
        assert!((run(Some(&ones), false).0 - plain).abs() < 1e-6);
        let zeros = map(&[0.0; 16]);
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
        let half: Vec<f32> = (0..16).map(|i| if i % 4 < 2 { 1.0 } else { 0.0 }).collect();
        let half = map(&half);
        let (dense, _) = run(Some(&half), false);
        let (ckpt, _) = run(Some(&half), true);
        assert!(dense > 0.0 && dense < plain, "{dense} vs {plain}");
        assert!(
            (dense - ckpt).abs() < 1e-4,
            "dense {dense} vs checkpoint {ckpt}"
        );
    }
}

// ===========================================================================================
// T2 (sc-7883) — real-weight adaptable-paths + checkpoint-parity harness (weight-gated).
//
//   SD3_LARGE_DIR=/path/to/stable-diffusion-3.5-large \
//     cargo test -p mlx-gen-sd3 --release --lib real_weight_repro -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod real_weight_repro {
    use super::*;
    use std::path::PathBuf;

    /// The `stabilityai/stable-diffusion-3.5-large` snapshot root from the required `SD3_LARGE_DIR`
    /// env var. sc-13668: there is no implicit default — the source snapshot path must be passed in
    /// explicitly.
    fn snapshot() -> Option<PathBuf> {
        std::env::var("SD3_LARGE_DIR").ok().map(PathBuf::from)
    }

    /// The `stabilityai/stable-diffusion-3.5-medium` snapshot root from the required `SD3_MEDIUM_DIR`
    /// env var. sc-13668: there is no implicit default — the source snapshot path must be passed in
    /// explicitly. T4 (sc-7885).
    fn medium_snapshot() -> Option<PathBuf> {
        std::env::var("SD3_MEDIUM_DIR").ok().map(PathBuf::from)
    }

    #[test]
    fn source_roots_require_explicit_env_no_default() {
        for (key, sentinel, resolver) in [
            (
                "SD3_LARGE_DIR",
                "/sentinel/sd3-large",
                snapshot as fn() -> Option<PathBuf>,
            ),
            (
                "SD3_MEDIUM_DIR",
                "/sentinel/sd3-medium",
                medium_snapshot as fn() -> Option<PathBuf>,
            ),
        ] {
            let saved = std::env::var(key).ok();
            std::env::remove_var(key);
            assert!(
                resolver().is_none(),
                "the source snapshot root must come from {key}: sc-13668 removed the implicit default"
            );
            std::env::set_var(key, sentinel);
            assert_eq!(resolver(), Some(PathBuf::from(sentinel)));
            match saved {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }

    /// The real MMDiT's default target surface resolves to the joint-block attention only: 38 blocks ×
    /// {to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj} = 266, plus to_add_out on all but the
    /// final context_pre_only block = 37 → 303 targets, none on globals.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-large weights; run as its own process"]
    fn default_targets_resolve_to_joint_block_attention() {
        let root =
            snapshot().expect("set SD3_LARGE_DIR to the stable-diffusion-3.5-large snapshot root");
        let dit = loader::load_transformer(&root, &Sd3Variant::Large.arch()).unwrap();
        let cfg = TrainingConfig::default();
        let paths = resolve_target_paths(&dit, &cfg);
        let n = dit.num_blocks();
        // 7 streams on every block + to_add_out on every non-final block.
        assert_eq!(paths.len(), n * 7 + (n - 1), "{} targets", n * 7 + (n - 1));
        assert!(paths.iter().all(|p| p.starts_with("transformer_blocks.")));
        assert!(paths.iter().any(|p| p.ends_with(".attn.to_q")));
        assert!(paths.iter().any(|p| p.ends_with(".attn.add_q_proj")));
        // The final context_pre_only block has no to_add_out.
        assert!(!paths.contains(&format!("transformer_blocks.{}.attn.to_add_out", n - 1)));
    }

    /// T4 (sc-7885) — the real Medium MMDiT-X default target surface resolves to the joint-block
    /// attention PLUS the dual-attention `attn2` on the first 13 blocks: 24 joint blocks ×
    /// {to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj} = 168, plus to_add_out on all but
    /// the final context_pre_only block (23), plus attn2's {to_q,to_k,to_v,to_out.0} on the 13 dual
    /// blocks (52). The 9-chunk SD35AdaLayerNormZeroX norm1 is NOT a target (modulation producer).
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-medium weights; run as its own process"]
    fn medium_default_targets_include_attn2_on_dual_blocks() {
        let root = medium_snapshot()
            .expect("set SD3_MEDIUM_DIR to the stable-diffusion-3.5-medium snapshot root");
        let dit = loader::load_transformer(&root, &Sd3Variant::Medium.arch()).unwrap();
        assert_eq!(dit.num_blocks(), 24, "Medium has 24 joint blocks");
        let cfg = TrainingConfig::default();
        let paths = resolve_target_paths(&dit, &cfg);

        // attn2 on every dual block (0..=12), all four locals; never on the plain blocks (13..23).
        for i in 0..13usize {
            for local in ["to_q", "to_k", "to_v", "to_out.0"] {
                assert!(
                    paths.contains(&format!("transformer_blocks.{i}.attn2.{local}")),
                    "dual block {i} attn2.{local} must be a default target"
                );
            }
        }
        for i in 13..24usize {
            assert!(
                !paths
                    .iter()
                    .any(|p| p.starts_with(&format!("transformer_blocks.{i}.attn2."))),
                "plain block {i} must have NO attn2 target"
            );
        }
        // norm1 (the modulation producer) is never targeted.
        assert!(!paths.iter().any(|p| p.contains(".norm1")));

        // Count: 24 joint blocks × 7 + 23 to_add_out + 13 dual blocks × 4 attn2 = 168 + 23 + 52 = 243.
        let expected = 24 * 7 + 23 + 13 * 4;
        assert_eq!(
            paths.len(),
            expected,
            "Medium default-target count {expected}"
        );
        let attn2_count = paths.iter().filter(|p| p.contains(".attn2.")).count();
        assert_eq!(attn2_count, 52, "13 dual blocks × 4 attn2 locals");
    }

    /// Whole-block gradient checkpointing must not change the math: the checkpointed forward+grads must
    /// match the dense path within fp tolerance. Run in f32 at a tiny resolution.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-large weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let root =
            snapshot().expect("set SD3_LARGE_DIR to the stable-diffusion-3.5-large snapshot root");
        let mut dit = loader::load_transformer(&root, &Sd3Variant::Large.arch()).unwrap();
        dit.cast_weights(Dtype::Float32).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
        let adapter = TrainAdapter::Lora { targets };
        let mut locals: Vec<Vec<String>> = vec![Vec::new(); dit.num_blocks()];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    locals[i].push(local.to_string());
                }
            }
        }

        // Tiny synthetic batch (latent 32×32 → img tokens 256; the math is resolution-agnostic).
        let x0 =
            random::normal::<f32>(&[1, 16, 32, 32], None, None, Some(&random::key(1).unwrap()))
                .unwrap();
        let noise =
            random::normal::<f32>(&[1, 16, 32, 32], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        let context =
            random::normal::<f32>(&[1, 333, 4096], None, None, Some(&random::key(3).unwrap()))
                .unwrap();
        let pooled =
            random::normal::<f32>(&[1, 2048], None, None, Some(&random::key(4).unwrap())).unwrap();
        let cond = Sd3Conditioning { context, pooled };

        dit.set_sdpa_checkpoint(true);
        let (_l, g_dense) = compute_loss_grads(
            &mut dit,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &cond,
            0.5,
            &noise,
            false,
            None,
            Dtype::Float32,
            None,
            None,
        )
        .unwrap();
        eval(g_dense.values()).unwrap();

        dit.set_sdpa_checkpoint(false);
        let (_l, g_ckpt) = compute_loss_grads(
            &mut dit,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &cond,
            0.5,
            &noise,
            false,
            None,
            Dtype::Float32,
            None,
            Some(&locals),
        )
        .unwrap();
        eval(g_ckpt.values()).unwrap();

        let mut max_rel = 0f32;
        for (k, a) in &g_dense {
            let b = g_ckpt.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        eprintln!("[sc-7883] checkpointed-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense: max rel {max_rel:.2e}"
        );
    }

    /// END-TO-END TRAINING SMOKE + ROUND-TRIP (the T2 acceptance proof): train a tiny LoRA (few steps,
    /// small rank, bf16 + gradient checkpointing) on a tiny synthetic dataset through the real
    /// [`Sd3LoraTrainer`], save the adapter, then RELOAD it via [`crate::adapters::apply_sd3_adapters`]
    /// at `sd3_5_large` generation and render — confirming the adapter loads, applies, and produces a
    /// coherent image (the round-trip). Memory-hardened for the 8.1B Large via the trainer's bf16 +
    /// block checkpointing.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-large weights + Metal; run as its own process"]
    fn training_smoke_round_trip() {
        use mlx_gen::runtime::{AdapterKind, AdapterSpec, LoadSpec, WeightsSource};
        use mlx_gen::{GenerationOutput, GenerationRequest, NetworkType, TrainingItem};

        let root =
            snapshot().expect("set SD3_LARGE_DIR to the stable-diffusion-3.5-large snapshot root");
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();

        // --- tiny synthetic dataset: 2 solid-color 256² PNGs with captions ---
        let mk_png = |path: &std::path::Path, rgb: [u8; 3]| {
            let mut img = image::RgbImage::new(256, 256);
            for px in img.pixels_mut() {
                *px = image::Rgb(rgb);
            }
            img.save(path).unwrap();
        };
        let img_a = tmp.join("a.png");
        let img_b = tmp.join("b.png");
        mk_png(&img_a, [200, 40, 40]);
        mk_png(&img_b, [40, 60, 200]);
        let items = vec![
            TrainingItem {
                image_path: img_a.clone(),
                caption: "sks a solid crimson swatch".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            },
            TrainingItem {
                image_path: img_b.clone(),
                caption: "sks a solid cobalt swatch".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            },
        ];

        // --- train (bf16 + gradient checkpointing; tiny rank/steps at 256²) ---
        // LoadSpec default precision is Bf16 — exactly the trainer's bf16 compute path.
        let mut trainer = load_trainer(&LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("load sd3 trainer");
        let cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 6,
            resolution: 256,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            gradient_checkpointing: true,
            train_dtype: "bf16".into(),
            timestep_type: "logit_normal".into(),
            ..Default::default()
        };
        let adapter_path = tmp.join("sd3_smoke_lora.safetensors");
        let req = TrainingRequest {
            items,
            config: cfg,
            output_dir: tmp.clone(),
            file_name: "sd3_smoke_lora.safetensors".into(),
            trigger_words: vec!["sks".into()],
            cancel: mlx_gen::CancelFlag::new(),
        };
        let mut losses: Vec<f32> = Vec::new();
        let out = trainer
            .train(&req, &mut |p| {
                if let TrainingProgress::Training { step, loss, .. } = p {
                    eprintln!("[sc-7883 smoke] step {step} loss {loss:.5}");
                    losses.push(loss);
                }
            })
            .expect("training run");
        eprintln!(
            "[sc-7883 smoke] TRAINED: steps={} final_loss={:.5} adapter={}",
            out.steps,
            out.final_loss,
            out.adapter_path.display()
        );
        assert!(out.adapter_path.exists(), "adapter file written");
        assert!(out.steps == 6, "ran all steps");
        assert!(
            out.final_loss.is_finite() && out.final_loss > 0.0,
            "finite loss"
        );
        // Drop the trainer (frees the resident MMDiT) before loading the inference model.
        drop(trainer);
        mlx_rs::memory::clear_cache();

        // --- round-trip: reload the trained adapter at sd3_5_large generation ---
        let spec = LoadSpec::new(WeightsSource::Dir(root))
            .with_quant(mlx_gen::Quant::Q8) // Q8 keeps the 8.1B inference footprint in budget
            .with_adapters(vec![AdapterSpec::new(
                adapter_path.clone(),
                1.0,
                AdapterKind::Lora,
            )]);
        let model = crate::model::load(&spec).expect("load sd3_5_large WITH the trained adapter");
        let gen_req = GenerationRequest {
            prompt: "sks a solid crimson swatch".into(),
            width: 512,
            height: 512,
            steps: Some(8),
            seed: Some(1),
            count: 1,
            ..Default::default()
        };
        let gout = model
            .generate(&gen_req, &mut |_| {})
            .expect("generate WITH the reloaded adapter");
        let img = match gout {
            GenerationOutput::Images(mut v) => v.remove(0),
            _ => panic!("expected an image"),
        };
        assert_eq!((img.width, img.height), (512, 512));
        // Coherence: not all-black / not NaN-collapsed (a broken adapter apply produces a degenerate
        // frame). Check the mean luminance is in a sane mid-range and there is real variance.
        let n = img.pixels.len() as f64;
        let mean = img.pixels.iter().map(|&b| b as f64).sum::<f64>() / n;
        let var = img
            .pixels
            .iter()
            .map(|&b| (b as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        eprintln!("[sc-7883 smoke] ROUND-TRIP render mean={mean:.1} var={var:.1}");
        let png = tmp.join("sd3_smoke_render.png");
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&png)
            .unwrap();
        eprintln!("[sc-7883 smoke] wrote {}", png.display());
        assert!(
            mean > 5.0 && mean < 250.0,
            "render mean luminance sane (coherent)"
        );
        assert!(
            var > 1.0,
            "render has real variance (not a flat/degenerate frame)"
        );
    }

    /// T4 (sc-7885) — END-TO-END Medium (MMDiT-X) TRAINING SMOKE + ROUND-TRIP (the T4 acceptance
    /// proof): train a tiny LoRA (few steps, small rank, bf16 + gradient checkpointing) on a tiny
    /// synthetic dataset through the real [`Sd3LoraTrainer`] loaded for **Medium** (24-block MMDiT-X,
    /// dual-attention `attn2` on the first 13 blocks). The default target set captures `attn2` (the
    /// arch-driven AdaptableHost). Save the adapter, then RELOAD it via
    /// [`crate::adapters::apply_sd3_adapters`] at `sd3_5_medium` generation and render — confirming the
    /// Medium-trained adapter (incl. the dual-attention modules) loads, applies, and produces a
    /// coherent image. Memory-friendly: Medium is 2.5B, far under the Large budget.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-medium weights + Metal; run as its own process"]
    fn medium_training_smoke_round_trip() {
        use mlx_gen::runtime::{AdapterKind, AdapterSpec, LoadSpec, WeightsSource};
        use mlx_gen::{GenerationOutput, GenerationRequest, NetworkType, TrainingItem};

        let root = medium_snapshot()
            .expect("set SD3_MEDIUM_DIR to the stable-diffusion-3.5-medium snapshot root");
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();

        // --- tiny synthetic dataset: 2 solid-color 256² PNGs with captions ---
        let mk_png = |path: &std::path::Path, rgb: [u8; 3]| {
            let mut img = image::RgbImage::new(256, 256);
            for px in img.pixels_mut() {
                *px = image::Rgb(rgb);
            }
            img.save(path).unwrap();
        };
        let img_a = tmp.join("a.png");
        let img_b = tmp.join("b.png");
        mk_png(&img_a, [200, 40, 40]);
        mk_png(&img_b, [40, 60, 200]);
        let items = vec![
            TrainingItem {
                image_path: img_a.clone(),
                caption: "sks a solid crimson swatch".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            },
            TrainingItem {
                image_path: img_b.clone(),
                caption: "sks a solid cobalt swatch".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            },
        ];

        // --- train via the MEDIUM trainer (bf16 + gradient checkpointing; tiny rank/steps at 256²) ---
        let mut trainer = load_trainer_medium(&LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("load sd3 MEDIUM trainer");
        assert_eq!(
            trainer.descriptor().id,
            SD3_5_MEDIUM_TRAINER_ID,
            "trainer is the Medium base"
        );
        let cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 6,
            resolution: 256,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            gradient_checkpointing: true,
            train_dtype: "bf16".into(),
            timestep_type: "logit_normal".into(),
            ..Default::default()
        };
        let adapter_path = tmp.join("sd3_medium_smoke_lora.safetensors");
        let req = TrainingRequest {
            items,
            config: cfg,
            output_dir: tmp.clone(),
            file_name: "sd3_medium_smoke_lora.safetensors".into(),
            trigger_words: vec!["sks".into()],
            cancel: mlx_gen::CancelFlag::new(),
        };
        let mut losses: Vec<f32> = Vec::new();
        let out = trainer
            .train(&req, &mut |p| {
                if let TrainingProgress::Training { step, loss, .. } = p {
                    eprintln!("[sc-7885 medium smoke] step {step} loss {loss:.5}");
                    losses.push(loss);
                }
            })
            .expect("medium training run");
        eprintln!(
            "[sc-7885 medium smoke] TRAINED: steps={} final_loss={:.5} adapter={}",
            out.steps,
            out.final_loss,
            out.adapter_path.display()
        );
        assert!(out.adapter_path.exists(), "adapter file written");
        assert_eq!(out.steps, 6, "ran all steps");
        assert!(
            out.final_loss.is_finite() && out.final_loss > 0.0,
            "finite loss"
        );

        // Confirm the trained adapter ACTUALLY carries attn2 (dual-attention) factors — the T4 proof
        // that the Medium dual-attention modules were trained, not just the joint attention.
        let trained = mlx_gen::weights::Weights::from_file(&adapter_path)
            .expect("load the trained Medium adapter back");
        let has_attn2 = trained.keys().any(|k| k.contains(".attn2."));
        assert!(
            has_attn2,
            "the Medium-trained adapter must contain attn2 (dual-attention) factors"
        );
        eprintln!(
            "[sc-7885 medium smoke] adapter has attn2 factors: {has_attn2} (total tensors: {})",
            trained.keys().count()
        );

        // Drop the trainer (frees the resident MMDiT) before loading the inference model.
        drop(trainer);
        mlx_rs::memory::clear_cache();

        // --- round-trip: reload the trained adapter at sd3_5_medium generation ---
        let spec = LoadSpec::new(WeightsSource::Dir(root))
            .with_quant(mlx_gen::Quant::Q8)
            .with_adapters(vec![AdapterSpec::new(
                adapter_path.clone(),
                1.0,
                AdapterKind::Lora,
            )]);
        let model =
            crate::model::load_medium(&spec).expect("load sd3_5_medium WITH the trained adapter");
        let gen_req = GenerationRequest {
            prompt: "sks a solid crimson swatch".into(),
            width: 512,
            height: 512,
            steps: Some(8),
            seed: Some(1),
            count: 1,
            ..Default::default()
        };
        let gout = model
            .generate(&gen_req, &mut |_| {})
            .expect("generate WITH the reloaded Medium adapter");
        let img = match gout {
            GenerationOutput::Images(mut v) => v.remove(0),
            _ => panic!("expected an image"),
        };
        assert_eq!((img.width, img.height), (512, 512));
        let n = img.pixels.len() as f64;
        let mean = img.pixels.iter().map(|&b| b as f64).sum::<f64>() / n;
        let var = img
            .pixels
            .iter()
            .map(|&b| (b as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        eprintln!("[sc-7885 medium smoke] ROUND-TRIP render mean={mean:.1} var={var:.1}");
        let png = tmp.join("sd3_medium_smoke_render.png");
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&png)
            .unwrap();
        eprintln!("[sc-7885 medium smoke] wrote {}", png.display());
        assert!(
            mean > 5.0 && mean < 250.0,
            "render mean luminance sane (coherent)"
        );
        assert!(
            var > 1.0,
            "render has real variance (not a flat/degenerate frame)"
        );
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the SD3 step seam ([`run_train_step`] /
/// [`compute_step_loss_grads`]) on a tiny random-init MMDiT (2 joint blocks, 16 latent channels;
/// random weights for every converter key), a random-init tiny 16-channel TAESD3-layout decoder and
/// a random-init tiny Depth-Anything-V2. Seconds; no weights downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use mlx_gen::train::perceptual::AuxLossSchedule;
    use mlx_rs::transforms::eval;

    fn tiny_dit() -> Sd3Transformer {
        let arch = crate::config::Sd3Arch {
            num_layers: 2,
            head_dim: 8,
            num_heads: 2,
            patch_size: 2,
            in_channels: 16,
            out_channels: 16,
            joint_attention_dim: 24,
            pooled_projection_dim: 20,
            caption_projection_dim: 16,
            pos_embed_max_size: 8,
            time_proj_dim: 16,
            dual_attention_layers: 0,
        };
        let scale = Array::from_slice(&[0.05f32], &[1]);
        let mut w = mlx_gen::weights::Weights::empty();
        for (i, e) in crate::convert::expected_transformer_tensors(&arch)
            .into_iter()
            .enumerate()
        {
            let shape: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
            let key = random::key(1000 + i as u64).unwrap();
            let t = multiply(
                random::normal::<f32>(&shape, None, None, Some(&key)).unwrap(),
                &scale,
            )
            .unwrap();
            w.insert(e.key, t);
        }
        Sd3Transformer::from_weights(&w, &arch).unwrap()
    }

    fn aux_sched() -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        }
    }

    fn path_with(s: AuxLossSchedule) -> PerceptualPath {
        mlx_gen_perceptual::testing::tiny_depth_path(16, s).unwrap()
    }

    fn cfg() -> TrainingConfig {
        let mut cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        cfg.depth_anchoring.schedule = aux_sched();
        cfg
    }

    fn rnd(shape: &[i32], k: u64) -> Array {
        let a = random::normal::<f32>(shape, None, None, Some(&random::key(k).unwrap())).unwrap();
        eval([&a]).unwrap();
        a
    }

    /// `n` items: clean NCHW `[1, 16, 6, 4]` latents (48×32 decoded) + tiny conditioning.
    fn cache_n(n: u64, h: i32, w: i32) -> Vec<CacheEntry> {
        (0..n)
            .map(|i| {
                (
                    rnd(&[1, 16, h, w], 100 + i),
                    Sd3Conditioning {
                        context: rnd(&[1, 5, 24], 200 + i),
                        pooled: rnd(&[1, 20], 300 + i),
                    },
                    None,
                )
            })
            .collect()
    }

    struct Fixture {
        dit: Sd3Transformer,
        adapter: TrainAdapter,
        params: LoraParams,
        locals: Vec<Vec<String>>,
    }

    fn fixture(cfg: &TrainingConfig) -> Fixture {
        let mut dit = tiny_dit();
        let paths = resolve_target_paths(&dit, cfg);
        assert!(!paths.is_empty());
        let (targets, params) =
            build_lora_targets(&mut dit, &paths, cfg.rank as i32, cfg.seed).unwrap();
        let mut locals: Vec<Vec<String>> = vec![Vec::new(); dit.num_blocks()];
        for p in &paths {
            if let Some((idx, local)) = p
                .strip_prefix("transformer_blocks.")
                .and_then(|r| r.split_once('.'))
            {
                locals[idx.parse::<usize>().unwrap()].push(local.to_string());
            }
        }
        Fixture {
            dit,
            adapter: TrainAdapter::Lora { targets },
            params,
            locals,
        }
    }

    fn single_bucket(n: usize) -> BucketSchedule {
        BucketSchedule::new(
            n,
            &[gen_core::ResolutionBucket {
                resolution: 48,
                repeats: 1,
            }],
            7,
        )
    }

    fn step(
        f: &mut Fixture,
        cfg: &TrainingConfig,
        cache: &[CacheEntry],
        schedule: &BucketSchedule,
        path: Option<(&mut PerceptualPath, &mut AuxAlternation)>,
        n: u32,
        ckpt: bool,
    ) -> (StepLosses, LoraParams) {
        let locals = f.locals.clone();
        let (l, g) = run_train_step(
            &mut f.dit,
            &f.params,
            &f.adapter,
            cfg,
            cache,
            schedule,
            path,
            n,
            false,
            Dtype::Float32,
            None,
            ckpt.then_some(locals.as_slice()),
        )
        .unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    fn lora_b_sum(g: &LoraParams) -> f32 {
        let v: Vec<f32> = g
            .iter()
            .filter(|(k, _)| k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .collect();
        assert!(!v.is_empty());
        v.iter().sum()
    }

    /// The SD3 x0 recovery inverts `build_batch`: the raw SD3 velocity regresses `noise − x0`, so
    /// with the exact target as the prediction `x_t − t·v` is x0. Mutation: recover with
    /// `FlowX0MinusNoise` (the opposite sign) ⇒ red.
    #[test]
    fn flow_recovery_inverts_the_sd3_noising() {
        let x0 = rnd(&[1, 16, 6, 4], 1);
        let noise = rnd(&[1, 16, 6, 4], 2);
        for t in [0.05f32, 0.5, 0.95] {
            let (x_t, target) = build_batch(&x0, &noise, t).unwrap();
            let rec = Parameterization::FlowNoiseMinusX0 { sigma: t }
                .recover_x0(&x_t, &target)
                .unwrap();
            let err = rec
                .subtract(&x0)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(err < 1e-5, "t={t}: {err}");
        }
    }

    /// AC (a)/(b), dense and block-checkpointed: a depth step trains the LoRA through the depth term
    /// alone (no diffusion term, total == aux, non-zero finite LoRA-B grad); a diffusion step has no
    /// depth term. Mutation: force `diffusion_on = true` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only() {
        for ckpt in [false, true] {
            let cfg = cfg();
            let mut f = fixture(&cfg);
            let cache = cache_n(1, 6, 4);
            let schedule = single_bucket(1);
            let mut p = path_with(aux_sched());
            prepare_perceptual_references(&mut p, &cache).unwrap();
            let mut alt = AuxAlternation::new(1, 1);
            let (diff, _) = step(
                &mut f,
                &cfg,
                &cache,
                &schedule,
                Some((&mut p, &mut alt)),
                1,
                ckpt,
            );
            assert_eq!(diff.aux, None, "ckpt={ckpt}");
            assert_eq!(Some(diff.total), diff.diffusion);
            let (depth, g) = step(
                &mut f,
                &cfg,
                &cache,
                &schedule,
                Some((&mut p, &mut alt)),
                2,
                ckpt,
            );
            assert_eq!(depth.diffusion, None, "ckpt={ckpt}");
            let aux = depth.aux.expect("depth term");
            assert!(aux > 0.0 && aux.is_finite(), "ckpt={ckpt}: {aux}");
            assert_eq!(depth.total, aux);
            let gb = lora_b_sum(&g);
            assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B |Σ| = {gb}");
        }
    }

    /// The aux step trains at `t` remapped into the window `[0.6, 0.8]` — equal to a direct
    /// `compute_step_loss_grads` at `plan.noise_level`, and the depth term is the depth loss of the
    /// explicitly recovered `x_t − t·v`. Mutations: keep the sampled `t` ⇒ red; recover with the
    /// opposite flow sign ⇒ red.
    #[test]
    fn aux_step_trains_at_the_remapped_noise_level() {
        let mut cfg = cfg();
        let window = AuxLossSchedule {
            t_min: 0.6,
            t_max: 0.8,
            ..aux_sched()
        };
        cfg.depth_anchoring.schedule = window;
        let mut f = fixture(&cfg);
        let cache = cache_n(1, 6, 4);
        let schedule = single_bucket(1);
        let mut p = path_with(window);
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(1, 1);
        step(
            &mut f,
            &cfg,
            &cache,
            &schedule,
            Some((&mut p, &mut alt)),
            1,
            false,
        );
        let (depth, _) = step(
            &mut f,
            &cfg,
            &cache,
            &schedule,
            Some((&mut p, &mut alt)),
            2,
            false,
        );
        assert!(depth.aux.is_some());
        let raw = sample_sigma(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(2),
        )
        .unwrap();
        let plan = p.plan(2, 0, raw).unwrap();
        assert!((0.6..=0.8).contains(&plan.noise_level));
        assert_ne!(plan.noise_level, raw);
        let noise = random::normal::<f32>(
            &[1, 16, 6, 4],
            None,
            None,
            Some(&random::key(cfg.seed.wrapping_add(2).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let (x0, cond, _) = &cache[0];
        let (expected, _) = compute_step_loss_grads(
            &mut f.dit,
            &f.params,
            &f.adapter,
            cfg.alpha,
            cfg.rank as f32,
            x0,
            cond,
            plan.noise_level,
            &noise,
            false,
            None,
            Dtype::Float32,
            None,
            None,
            Some(AuxStep {
                path: &p,
                plan: &plan,
                entry: 0,
            }),
        )
        .unwrap();
        assert_eq!(depth, expected);

        // The depth term is the depth loss of the explicitly recovered x0 = x_t − t·v.
        f.adapter
            .install_as(&mut f.dit, &f.params, 4.0, 4.0, None, LOKR_DTYPE)
            .unwrap();
        let t = plan.noise_level;
        let (x_t, _) = build_batch(x0, &noise, t).unwrap();
        let v = f
            .dit
            .forward_with(
                &x_t,
                &cond.context,
                &cond.pooled,
                &Array::from_slice(&[t * NUM_TRAIN_TIMESTEPS], &[1]),
                Dtype::Float32,
            )
            .unwrap();
        let x0_hat = subtract(&x_t, multiply(&v, Array::from_f32(t)).unwrap()).unwrap();
        let want = p
            .aux_loss(&plan, 0, &x0_hat)
            .unwrap()
            .unwrap()
            .weighted
            .item::<f32>();
        let got = depth.aux.unwrap();
        assert!(
            (want - got).abs() <= 1e-5 * want.abs().max(1.0),
            "{want} vs {got}"
        );
    }

    /// Alternation per image (N = 2, every image gets both kinds in 2·N steps) and per-entry
    /// references across two buckets of different latent sizes (every depth step on bucket 1 uses
    /// bucket 1's reference; one reference per entry). Mutations: key the plan on the global step
    /// ⇒ red; pass the item as the `AuxStep` entry ⇒ a bucket-1 depth step compares against
    /// bucket 0's reference ⇒ red.
    #[test]
    fn alternation_is_per_image_with_per_entry_references() {
        let cfg = cfg();
        let mut f = fixture(&cfg);
        let items = 2usize;
        // Item-major: bucket 0 = 6×4 latents, bucket 1 = 4×6.
        let mut cache = Vec::new();
        for (a, b) in cache_n(2, 6, 4).into_iter().zip(cache_n(2, 4, 6)) {
            cache.push(a);
            cache.push(b);
        }
        let schedule = BucketSchedule::new(
            items,
            &[
                gen_core::ResolutionBucket {
                    resolution: 48,
                    repeats: 1,
                },
                gen_core::ResolutionBucket {
                    resolution: 32,
                    repeats: 1,
                },
            ],
            7,
        );
        let mut p = path_with(aux_sched());
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(items, 1);
        let steps = 2 * schedule.epoch_len() as u32;
        let mut kinds = Vec::new();
        let mut depth_on_bucket1 = false;
        for n in 1..=steps {
            let (l, _) = step(
                &mut f,
                &cfg,
                &cache,
                &schedule,
                Some((&mut p, &mut alt)),
                n,
                false,
            );
            let k = (n - 1) as usize;
            kinds.push((schedule.sample(k).0, l.aux.is_some()));
            depth_on_bucket1 |= l.aux.is_some() && schedule.cache_index(k) % 2 == 1;
        }
        for image in 0..items {
            let mine: Vec<bool> = kinds.iter().filter(|k| k.0 == image).map(|k| k.1).collect();
            // Each image strictly alternates diffusion / depth across its own visits.
            let alternating: Vec<bool> = (0..mine.len()).map(|v| v % 2 == 1).collect();
            assert_eq!(mine, alternating, "image {image}: {kinds:?}");
        }
        assert!(depth_on_bucket1, "{kinds:?}");
        assert_eq!(p.reference_computations(), cache.len());
    }

    /// E1: depth off ⇒ nothing loaded, no footprint, and the step is bit-identical to the
    /// pre-epic-2123 closure (reproduced verbatim); a diffusion-only step of an enabled path is
    /// bit-identical too. Mutation: flip MAE/MSE in the diffusion term ⇒ red.
    #[test]
    fn everything_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path(&TrainingConfig::default())
            .unwrap()
            .is_none());
        assert_eq!(
            perceptual_footprint_gb(&TrainingConfig::default(), 1024, 10),
            0.0
        );
        let off_cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            seed: 7,
            ..Default::default()
        };
        let mut f = fixture(&off_cfg);
        let cache = cache_n(2, 6, 4);
        let schedule = single_bucket(2);
        let (off, g_off) = step(&mut f, &off_cfg, &cache, &schedule, None, 1, false);
        assert_eq!(off.aux, None);

        let (x0, cond, _) = &cache[0];
        let t = sample_sigma(
            &off_cfg.timestep_type,
            &off_cfg.timestep_bias,
            off_cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1),
        )
        .unwrap();
        let noise = random::normal::<f32>(
            x0.shape(),
            None,
            None,
            Some(&random::key(off_cfg.seed.wrapping_add(1).wrapping_mul(2) + 1).unwrap()),
        )
        .unwrap();
        let (x_t, target) = build_batch(x0, &noise, t).unwrap();
        let timestep = Array::from_slice(&[t * NUM_TRAIN_TIMESTEPS], &[1]);
        let (context, pooled) = (cond.context.clone(), cond.pooled.clone());
        let dit = &mut f.dit;
        let adapter = &f.adapter;
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            adapter.install_as(dit, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            let v = dit
                .forward_with(&x_t, &context, &pooled, &timestep, Dtype::Float32)
                .map_err(|e| Exception::custom(e.to_string()))?;
            Ok(vec![reduce_loss(&subtract(&v, &target)?, None, false)?])
        };
        let (val, g_legacy) = keyed_value_and_grad(legacy)(f.params.clone(), 0).unwrap();
        eval(g_legacy.values()).unwrap();
        assert_eq!(off.total, val[0].item::<f32>());
        let bits =
            |a: &Array| -> Vec<u32> { a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect() };
        for (k, v) in &g_legacy {
            assert_eq!(bits(v), bits(&g_off[k]), "{k}");
        }

        let on_cfg = cfg();
        let mut f2 = fixture(&on_cfg);
        let mut p = path_with(aux_sched());
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(2, 1);
        let (on1, g_on1) = step(
            &mut f2,
            &on_cfg,
            &cache,
            &schedule,
            Some((&mut p, &mut alt)),
            1,
            false,
        );
        let mut f3 = fixture(&on_cfg);
        let (none1, g_none1) = step(&mut f3, &on_cfg, &cache, &schedule, None, 1, false);
        assert_eq!(on1, none1);
        for (k, v) in &g_none1 {
            assert_eq!(bits(v), bits(&g_on1[k]), "{k}");
        }
    }

    /// E7: the estimate grows by TAESD3 + DA2 (more for Large); the guard refuses at a synthetic
    /// budget between the MMDiT projection and projection + aux on the dense and the checkpointed
    /// path; a checkpointed run without aux is unguarded. Mutations: drop `+ extra_gb` ⇒ red;
    /// return early for every checkpointed run ⇒ red.
    #[test]
    fn memory_estimate_includes_the_aux_models() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = aux_sched();
        let small = perceptual_footprint_gb(&on, 1024, 10);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = perceptual_footprint_gb(&on, 1024, 10);
        assert!(small > 0.0 && large - small > 1.0, "{small} / {large}");
        let v = Sd3Variant::Large;
        let tokens = (1024.0f64 / 16.0).powi(2) + PREFLIGHT_TXT_TOKENS;
        for (checkpointed, base) in [
            (false, projected_dense_peak_gb(tokens, true, v)),
            (true, projected_dense_peak_gb(0.0, true, v)),
        ] {
            let budget = (base + large / 2.0) / 0.85;
            assert!(
                check_preflight_budget_with_aux(1024, true, budget, v, 0.0, checkpointed).is_ok()
            );
            assert!(
                check_preflight_budget_with_aux(1024, true, budget, v, large, checkpointed)
                    .is_err(),
                "checkpointed={checkpointed}"
            );
        }
        assert!(check_preflight_budget_with_aux(1024, true, 0.001, v, 0.0, true).is_ok());
    }

    /// Round-robin over N = 2 images (one bucket): a global-step key would lock image 0 to odd
    /// (diffusion) steps forever; per-image keys give each image both kinds within 2·N steps.
    /// Mutation: key the plan on the global step ⇒ red.
    #[test]
    fn round_robin_images_each_get_depth_steps() {
        let cfg = cfg();
        let mut f = fixture(&cfg);
        let cache = cache_n(2, 6, 4);
        let schedule = single_bucket(2);
        let mut p = path_with(aux_sched());
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let mut alt = AuxAlternation::new(2, 1);
        let kinds: Vec<(usize, bool)> = (1..=4)
            .map(|n| {
                let (l, _) = step(
                    &mut f,
                    &cfg,
                    &cache,
                    &schedule,
                    Some((&mut p, &mut alt)),
                    n,
                    false,
                );
                (schedule.sample((n - 1) as usize).0, l.aux.is_some())
            })
            .collect();
        for image in 0..2 {
            let mine: Vec<bool> = kinds.iter().filter(|k| k.0 == image).map(|k| k.1).collect();
            assert!(mine.contains(&true) && mine.contains(&false), "{kinds:?}");
        }
    }

    /// E3: both SD3.5 descriptors declare depth anchoring.
    #[test]
    fn descriptors_declare_depth_anchoring() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
        assert!(medium_trainer_descriptor().techniques.depth_anchoring);
    }

    /// A missing decoder is a named error (TAESD3).
    #[test]
    fn missing_aux_weights_are_named() {
        let tmp = tempfile::tempdir().unwrap();
        let mut c = cfg();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taesd3"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c)
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("TAESD3"), "{err}");
    }
}
