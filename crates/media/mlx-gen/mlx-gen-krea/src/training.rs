//! LoRA/LoKr **training** on the Krea 2 **Raw** DiT, in pure Rust on mlx-rs (sc-7577, epic 7565 P3) —
//! the MLX-native Krea LoRA-training base. LoRAs train on the undistilled `krea/Krea-2-Raw` 12B DiT and
//! apply at `krea_2_turbo` inference (the Lens / Z-Image precedent: same architecture, no base-model
//! gating, family-match suffices — sc-7578).
//!
//! [`KreaRawTrainer`] realizes the core [`Trainer`] contract on the real 28-block
//! single-stream Krea DiT, mirroring `LensTrainer` / `mlx_gen_lens` and `ZImageTurboTrainer` — the model
//! crates don't use mlx-rs's `Module` system (hand-rolled `&self` forwards over raw `Array`s), so
//! training uses the **functional autograd**: the trainable factors live OUTSIDE the model in a
//! [`LoraParams`] map, re-injected each step into the target
//! [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s via the shared core seam
//! ([`mlx_gen::train::lora`]), stepped with `keyed_value_and_grad` + the core [`TrainOptimizer`] +
//! `clip_grad_norm`. The injection mirrors the inference reload op-for-op, so the trained adapter
//! round-trips through the sc-7578 apply path bit-for-bit.
//!
//! ## What is Krea-specific (everything else reuses the family-agnostic core unchanged)
//! - **Flow-match velocity target = `noise − x0`** with the DiT **timestep = `t`** (the noise fraction)
//!   fed directly. The Krea DiT [`forward`](crate::transformer::Krea2Transformer::forward) returns the
//!   **raw** velocity (no negation — the Turbo pipeline feeds it to the Euler step un-negated, sc-7571),
//!   so the regression target is the velocity itself (the **same** sign as Lens; the OPPOSITE of the
//!   Z-Image trainer, whose Rust `forward()` negates). `x_t = (1 − t)·x0 + t·noise`.
//! - **Latents** by the Qwen-Image VAE encode: `preprocess_init_image` (resize + `[−1,1]` NCHW) →
//!   [`QwenVae::encode`](crate::vae::QwenVae) (per-channel-normalized 16-ch latent) → drop the singleton
//!   temporal axis → `[1, 16, edge/8, edge/8]`, exactly the latent the DiT predicts in.
//! - **Caption features** are the Qwen3-VL-4B condition encoder's stacked 12 select-layers
//!   `[1, n_tok, 12, 2560]` (the DiT's `text_fusion` aggregator consumes them). Single-conditional (no
//!   CFG) for the regression, B = 1 → `mask = None`. The encoder loads **Q8** (~4.5 GB vs ~8 GB dense
//!   bf16) — frozen, used only to cache caption features, then dropped before the train loop (the 32
//!   GB-Mac free pattern); Q8 also matches the published Turbo TE.
//! - **Targets** default to the single-stream block attention `to_q`/`to_k`/`to_v`/`to_out.0` (the
//!   `AdaptableHost for Krea2Transformer` paths, sc-7577); LoKr reconstructs at `LOKR_DTYPE`.
//!
//! Registered under the **`krea_2_raw`** id (the LoRA-training base; arch-identical to `krea_2_turbo`,
//! so the adapter applies to Turbo inference — sc-7578).
//!
//! ## Memory hardening (the z-image sc-4874/4886/4887 / Lens sc-5170 analog — Krea is the largest base)
//! - **SDPA-segment checkpointing** is always on in training (LoRA and LoKr): the gated attention's
//!   SDPA runs inside an `mlx::checkpoint` so its backward recomputes the attention rather than
//!   retaining the `[heads, joint, joint]` probability matrix (the dominant seq² term). Numerically
//!   identical.
//! - **`gradient_checkpointing`** (the SceneWorks toggle) is an opt-in OPTION (LoRA only): each of the
//!   28 single-stream blocks recomputes its activations in the backward via
//!   [`Krea2Transformer::forward_with_blocks_checkpointed`](crate::transformer::Krea2Transformer::forward_with_blocks_checkpointed),
//!   threading the per-block LoRA factors as explicit checkpoint inputs so the adapter graph survives
//!   the recompute. LoKr keeps the dense path (caught by the guard) — mirroring z-image / Lens.
//! - **Fail-fast OOM preflight guard** — `preflight_memory_guard` projects the dense first-step peak
//!   from resolution and, when checkpointing is off and the run would exceed this machine's memory
//!   budget, returns a catchable, actionable error BEFORE the (minutes-long) latent caching — converting
//!   the otherwise-uncatchable SIGKILL into a recommendation to enable the toggle.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::{self, BucketSchedule};
use mlx_gen::image::decoded_to_image;
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
use mlx_gen::train::taehv::TaehvConfig;
use mlx_gen::{
    run_flow_sampler, CancelFlag, Error, LoadSpec, Modality, NetworkType, Precision, Progress,
    Result, TimestepConvention, TrainOptimizer, Trainer, TrainerDescriptor, TrainingConfig,
    TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::loader::{load_text_encoder, load_transformer};
use crate::pipeline::krea_cfg_combine;
use crate::schedule::{dynamic_mu, krea_sigmas};
use crate::text_encoder::{KreaTextEncoder, KreaTokenizer};
use crate::transformer::Krea2Transformer;
use crate::vae::{load_vae, QwenVae};

/// Registry id for the Krea LoRA-training base (the undistilled `krea/Krea-2-Raw` DiT). The trained
/// adapter records `baseModel: krea_2_raw` / `family: krea_2` and applies at `krea_2_turbo` inference
/// (sc-7578) — the family-match cross-apply, no base-model gating (the Lens / Z-Image precedent).
pub const KREA_2_RAW_TRAINER_ID: &str = "krea_2_raw";

/// The LoKr delta-reconstruction dtype, matching what the sc-7578 inference loader will use so a
/// trained LoKr round-trips through the apply path. bf16 — the family compute dtype.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// The Qwen3-VL-4B condition encoder is loaded Q8 for the trainer (~4.5 GB vs ~8 GB dense bf16): it is
/// frozen and used only to cache caption features once, then dropped before the train loop. Q8 also
/// matches the published Turbo TE (the turnkey packs it Q8/Q4), so the cached features match deployment.
const TRAINER_ENCODER_BITS: i32 = 8;

/// The default target modules: the single-stream block joint-attention projections (`to_out` is the
/// `nn.ModuleList([Linear, Identity])`, so the trainable Linear is `to_out.0`). The Krea-specific
/// sigmoid `to_gate` and the SwiGLU FFN are reachable as explicit targets but are not in the default
/// set (the standard PEFT attention surface).
const DEFAULT_TARGET_MODULES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// Recognized `timestep_type` values [`sample_sigma`] branches on (plus the `sigmoid` default).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
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
/// `target = noise − x0` (the velocity the **raw** Krea DiT output is regressed onto; the DiT timestep
/// is `t` itself, fed by the caller — see the module docs for the sign vs Z-Image).
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The production [`Trainer`] for the base `krea/Krea-2-Raw` DiT: a frozen base (Qwen3-VL-4B encoder +
/// single-stream DiT + Qwen-Image VAE + tokenizer) that caches a captioned dataset to
/// VAE-latents/caption-features, then runs the functional-autograd LoRA/LoKr loop with the core runtime
/// glue (LR schedule, gradient accumulation, checkpoint cadence, cancel, progress bands), writing a PEFT
/// adapter that reloads through the inference path.
pub struct KreaRawTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: KreaTokenizer,
    /// The Qwen3-VL-4B encoder, in an `Option` so it can be **dropped after caching** — it is idle
    /// during training (every caption is already cached), yet a multi-GB resident.
    encoder: Option<KreaTextEncoder>,
    transformer: Krea2Transformer,
    vae: QwenVae,
    /// The compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: KREA_2_RAW_TRAINER_ID,
        family: "krea_2",
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
        // edge, walked through a `BucketSchedule`; the pre-flight guard and the preview render size
        // for the largest edge.
        // sc-24828 (epic 2123): honors `subject_mask_loss` on its one (LoRA/LoKr, dense or
        // block-checkpointed) loss path, with a weight map per (item, bucket) entry.
        // sc-24830 (epic 2123): depth anchoring on that same loss path (dense and
        // block-checkpointed) — the shared decoded-x0 perceptual path through TAEW2.1, the
        // Qwen-Image VAE's tiny decoder.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// Construct the trainer from a `krea/Krea-2-Raw` snapshot directory (the diffusers multi-component
/// tree: `tokenizer/ text_encoder/ transformer/ vae/`). The DiT is loaded **dense** (the adapter host);
/// the encoder is Q8. `spec.precision` selects the compute dtype (bf16 default / f32 tight-gate); the
/// Raw snapshot ships bf16, so f32 widens it via [`Krea2Transformer::cast_weights`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "krea trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let tokenizer = KreaTokenizer::from_snapshot(&root)?;
    let mut encoder = load_text_encoder(&root)?;
    encoder.quantize(TRAINER_ENCODER_BITS)?;
    let mut transformer = load_transformer(&root)?;
    if transformer.compute_dtype() != dtype {
        transformer.cast_weights(dtype)?;
    }
    let vae = load_vae(&root)?;
    Ok(Box::new(KreaRawTrainer {
        descriptor: trainer_descriptor(),
        tokenizer,
        encoder: Some(encoder),
        transformer,
        vae,
        dtype,
    }))
}

// The trainer registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without a loaded
/// trainer. Rejects an empty dataset, zero rank, **zero steps** (a 0-step run would write a no-op
/// `B = 0` identity adapter), an unsupported optimizer, and an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type`. `gradient_checkpointing` is a supported toggle (the
/// checkpointed DiT forward + the OOM preflight guard are wired in [`KreaRawTrainer::train_impl`]).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err("krea trainer: dataset is empty".into());
    }
    if cfg.rank == 0 {
        return Err("krea trainer: rank must be > 0".into());
    }
    if cfg.steps == 0 {
        return Err("krea trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "krea trainer: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(format!(
            "krea trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "krea trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "krea trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for KreaRawTrainer {
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
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would train zero
        // parameters yet "succeed". Catch it here, where the loaded DiT is available to match against.
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "krea trainer: lora_target_modules {:?} matched no adaptable module on the Krea DiT \
                 (defaults are to_q/to_k/to_v/to_out.0 on the single-stream blocks)",
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

impl KreaRawTrainer {
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
                "krea trainer: lora_target_modules {:?} matched no adaptable module on the Krea DiT",
                cfg.lora_target_modules
            )
            .into());
        }

        // The DiT compute dtype is fixed at load (`spec.precision`); enforce `train_dtype` against it
        // (never a silent no-op). The common case (TrainingConfig default bf16 + LoadSpec default Bf16)
        // matches, so this only fires on an explicit f32-vs-bf16 mismatch.
        let want_bf16 = {
            let t = cfg.train_dtype.trim();
            t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == Dtype::Bfloat16;
        if want_bf16 != loaded_bf16 {
            return Err(format!(
                "krea trainer: train_dtype '{}' does not match the loaded precision ({}). Load the \
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
        // off). The pre-flight guard and the preview render use the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let edge = preflight_edge(&edges);

        // sc-7577 — fail-fast pre-flight memory guard (the z-image/Lens analog). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`;
        // at high resolution that working set can exceed unified memory and the OS hard-kills the
        // worker with an UNCATCHABLE SIGKILL. We predict it and refuse up front with a catchable,
        // actionable error BEFORE the (minutes-long) latent caching, UNLESS the run will
        // block-checkpoint (LoRA + the toggle). LoKr always takes the dense path, so it is guarded.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        // Epic 2123 E7 (sc-24830): the depth-anchoring models (TAEW2.1 + Depth-Anything-V2) count
        // against the budget on BOTH paths — a checkpointed depth job must not skip admission.
        let aux_gb = perceptual_footprint_gb(cfg, edge, req.items.len() * edges.len());
        if !will_checkpoint || aux_gb > 0.0 {
            preflight_memory_guard(edge, want_bf16, aux_gb, will_checkpoint)?;
        }

        // Epic 2123 depth anchoring (sc-24830): load the frozen TAEW2.1 decoder +
        // Depth-Anything-V2 before the caching pass, so a missing checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: VAE-latents + caption features into memory ---
        on_progress(TrainingProgress::LoadingModel); // base model is already resident from load_trainer
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). Each entry: clean latent,
        // caption features, and (subject-masked loss, sc-24828) that bucket's latent loss-weight
        // map — `None` when the technique is off.
        let mut cache: Vec<(Array, Array, Option<Array>)> =
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
                "krea trainer",
                item,
                cfg.subject_mask_loss.as_ref(),
            )?;
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg(
                    "krea trainer: text encoder already freed (caching after train loop)".into(),
                )
            })?;
            let context = encode_caption(&self.tokenizer, encoder, &item.caption)?;
            eval([&context])?;
            // The caption context is resolution-independent: encode it once, then one latent per
            // bucket edge (the DiT derives its patch grid / RoPE from the latent's own shape).
            for (x0, mask_weight) in encode_buckets(&edges, mask.as_ref(), |edge| {
                encode_latents(&self.vae, &img, edge) // [1, 16, edge/8, edge/8]
            })? {
                cache.push((x0, context.clone(), mask_weight));
            }
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("krea trainer: no usable dataset items".into());
        }

        // sc-5637 — pre-encode the preview-sample prompts (positive context per prompt) + one shared
        // empty-prompt (unconditional) context for the CFG preview, while the encoder is still resident.
        let sample_caps: Vec<(String, Array)> = if cfg.sample_every > 0
            && !cfg.sample_prompts.is_empty()
            && !req.cancel.is_cancelled()
        {
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg("krea trainer: text encoder already freed (sample pre-encode)".into())
            })?;
            let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let ctx = encode_caption(&self.tokenizer, encoder, prompt)?;
                eval([&ctx])?;
                caps.push((prompt.clone(), ctx));
            }
            caps
        } else {
            Vec::new()
        };
        let sample_neg: Option<Array> = if !sample_caps.is_empty() {
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg(
                    "krea trainer: text encoder already freed (sample neg pre-encode)".into(),
                )
            })?;
            let neg = encode_caption(&self.tokenizer, encoder, "")?;
            eval([&neg])?;
            Some(neg)
        } else {
            None
        };
        let sampling_enabled = !sample_caps.is_empty();

        // Every caption is cached now — free the 4 B-param encoder and evict its buffers before the
        // train loop, reclaiming that resident for the DiT working set.
        self.encoder = None;
        mlx_rs::memory::clear_cache();

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference (TAEW2.1 decode of its
        // cached clean latent → DA2 depth) is computed exactly once per job, here — after the text
        // encoder is freed, so the decoder + DA2 never share residency with it.
        if let Some(path) = perceptual.as_mut() {
            prepare_perceptual_references(path, &cache)?;
        }

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

        // sc-7577 — gradient checkpointing. Collect, per single-stream block, the adapter-routable
        // LOCAL paths trained on it (e.g. `"attn.to_q"`), in trained-file order — the factors a
        // checkpoint segment threads as explicit inputs. Only `transformer_blocks.*` targets are
        // checkpointed; any text-fusion / global targets train dense through `self`.
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
        // Opt-in OPTION (the SceneWorks "Gradient Checkpointing" toggle), never auto-forced — a run that
        // would OOM is caught instead by the pre-flight guard above. LoRA only — LoKr falls back to the
        // dense path.
        let use_checkpoint =
            matches!(adapter, TrainAdapter::Lora { .. }) && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing is ALWAYS on in training. When whole-block checkpointing is on, the
        // per-block SDPA flag goes OFF (the block recompute already covers attention).
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
        // the real dataset item. A resumed run replays the skipped prefix so the phase matches.
        let mut alternation = perceptual
            .as_ref()
            .map(|_| AuxAlternation::new(cache.len() / edges.len(), accum));
        if let Some(alt) = alternation.as_mut() {
            for s in 1..=start_step {
                alt.key(s, schedule.sample((s - 1) as usize).0);
            }
        }
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let entry = schedule.cache_index((step - 1) as usize);
            let (x0, context, mask_weight) = &cache[entry];
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
            // Epic 2123 E8: plan the step's loss terms; an aux-only step trains at the noise level
            // remapped into the loss window.
            let plan = match (perceptual.as_ref(), alternation.as_mut()) {
                (Some(path), Some(alt)) => {
                    let plan = plan_aux_step(path, alt, &schedule, step, t)?;
                    t = plan.noise_level;
                    Some(plan)
                }
                _ => None,
            };
            let aux = perceptual
                .as_ref()
                .zip(plan.as_ref())
                .map(|(path, plan)| AuxStep {
                    path,
                    plan,
                    image: entry,
                });
            let (losses, grads) = compute_step_loss_grads(
                &mut self.transformer,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                context,
                t,
                &noise,
                mae,
                mask_weight.as_ref(),
                compute_dtype,
                lora_dtype,
                checkpoint_blocks,
                aux,
            )?;
            last_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // The final update can fire with fewer than `accum` grads when `steps` isn't a multiple
                // of the accumulation; divide by the actual in-window count instead.
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

            // sc-5637 — periodic best-effort previews from the in-progress adapter. Install the current
            // factors as concrete adapters for the forward-only render; the next step's traced `loss_fn`
            // re-installs them. A render failure must NOT abort the long training run — log and continue.
            if sampling_enabled && step % cfg.sample_every == 0 {
                adapter.install_as(
                    &mut self.transformer,
                    &params,
                    alpha,
                    rank,
                    lora_dtype,
                    LOKR_DTYPE,
                )?;
                let neg = sample_neg
                    .as_ref()
                    .expect("neg context pre-encoded when sampling");
                let total = sample_caps.len() as u32;
                for (i, (prompt, ctx_pos)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match render_sample(
                        &self.transformer,
                        &self.vae,
                        ctx_pos,
                        neg,
                        sample_seed,
                        edge,
                        cfg.sample_steps.max(1) as usize,
                        cfg.sample_guidance_scale,
                        &req.cancel,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(e) => eprintln!(
                            "[sc-5637] {KREA_2_RAW_TRAINER_ID} preview sample failed at step {step} \
                             (prompt {}): {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
            }
        }

        // Cancelled before a single step completed (`steps == 0` is rejected by `validate`): the factors
        // are still the `B = 0` no-op init. Surface the cancellation rather than writing a valid-looking
        // identity adapter as a trained artifact.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save final adapter (the diffusers/PEFT format the sc-7578 apply path loads) ---
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

/// The perceptual plan of 1-based micro-`step` (epic 2123 E8): the alternation key comes from the
/// step's REAL dataset item (the bucket schedule's `(item, bucket)`), the reference from its
/// `(item, bucket)` cache entry; `raw_t` is the sampled noise level.
fn plan_aux_step(
    path: &PerceptualPath,
    alternation: &mut AuxAlternation,
    schedule: &BucketSchedule,
    step: u32,
    raw_t: f32,
) -> Result<StepPlan> {
    let k = (step - 1) as usize;
    let key = alternation.key(step, schedule.sample(k).0);
    path.plan(key, schedule.cache_index(k), raw_t)
}

/// Krea's latent family for the shared aux-loss builder (epic 2123 E8): the Qwen-Image VAE's
/// 16-channel per-channel-normalized latent (`QwenVae::encode` applies `latents_mean` /
/// `latents_std`, the space the DiT predicts in), decoded by TAEW2.1 — the TAEHV checkpoint
/// upstream lists for Qwen-Image, which takes that normalized latent with no scale/shift. A still
/// image is one `T = 1` clip.
fn krea_decoder() -> mlx_gen_perceptual::DecoderSpec {
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
            label: "krea trainer",
            decoder: krea_decoder(),
            latent_lpips: None,
        },
    )
}

/// Compute every (item, bucket) cache entry's perceptual reference once, from its clean
/// `[1, 16, h, w]` latent (already the decoder's NCHW layout).
fn prepare_perceptual_references(
    path: &mut PerceptualPath,
    cache: &[(Array, Array, Option<Array>)],
) -> Result<()> {
    for (entry, (x0, _, _)) in cache.iter().enumerate() {
        path.ensure_reference(entry, x0)?;
    }
    Ok(())
}

/// Extra training memory (GB) the enabled perceptual losses add at the largest bucket `edge`
/// (epic 2123 E7): TAEW2.1 + the losses' frozen models plus `entries` cached references. `0` when
/// nothing is enabled.
fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &krea_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// Number of caption tokens assumed by the pre-flight projection (a representative length; the unified
/// sequence is `img_len + cap_len`, and `img_len = (edge/16)²` dominates).
const PREFLIGHT_TXT_TOKENS: f64 = 64.0;

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the unified
/// token count `s = img_len + cap_len`, for the 12B Krea DiT. The structure follows the z-image/Lens
/// `weights + linear·s + quad·s²` decomposition: the constant is the resident DiT base (bf16 ~24 GB /
/// f32 ~48 GB; the 4B encoder is freed before the train loop), the linear term is the per-token
/// activations across the 28 single-stream blocks, and the quadratic term is the seq² attention
/// transient — demoted to a single block's backward transient by the always-on SDPA-segment
/// checkpointing.
///
/// **These constants are a CONSERVATIVE INITIAL ESTIMATE** (the DiT is larger per-layer than Lens, so
/// its curve is not reused) — they err toward refusing borderline runs (recommending Gradient
/// Checkpointing) rather than allowing a SIGKILL, and are to be refit from the real-weight
/// `first_step_sweep` harness in `tests/`. `projection_is_monotonic_and_conservative` pins the shape.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        PREFLIGHT_BF16.0 + PREFLIGHT_BF16.1 * s + PREFLIGHT_BF16.2 * s * s
    } else {
        PREFLIGHT_F32.0 + PREFLIGHT_F32.1 * s + PREFLIGHT_F32.2 * s * s
    }
}

/// `(weights, linear, quad)` conservative initial constants for [`projected_dense_peak_gb`]. Refit both
/// tuples from the real-weight `first_step_sweep` once measured (see `tests/`).
const PREFLIGHT_F32: (f64, f64, f64) = (48.0, 1.20e-2, 3.0e-7);
const PREFLIGHT_BF16: (f64, f64, f64) = (24.0, 6.0e-3, 1.5e-7);

/// The edge the pre-flight guard (and the preview render) sizes for: the LARGEST bucket edge (epic
/// 2123 E7) — the dense first step's working set is set by the biggest latent the run will ever train
/// on, whichever bucket the schedule happens to draw first.
fn preflight_edge(edges: &[u32]) -> u32 {
    edges.iter().copied().max().unwrap_or(0)
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. The budget is MLX's own reported memory
/// limit (≈ the device's recommended working set). Only consulted when gradient checkpointing is OFF.
///
/// Epic 2123 E7 (sc-24830): `extra_gb` is the training-time auxiliary models' footprint
/// ([`perceptual_footprint_gb`]); with it the guard also runs with gradient checkpointing on
/// (`checkpointed`), stacking on the resident DiT base ([`checkpointed_baseline_gb`]).
fn preflight_memory_guard(edge: u32, bf16: bool, extra_gb: f64, checkpointed: bool) -> Result<()> {
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    check_preflight_budget_with(edge, bf16, budget_gb, extra_gb, checkpointed)
}

/// The checkpointed DiT baseline the auxiliary-model guard stacks on: the resident-base term of
/// [`projected_dense_peak_gb`] (block checkpointing removes most of the activation terms; the
/// base weights stay). A lower bound.
fn checkpointed_baseline_gb(bf16: bool) -> f64 {
    projected_dense_peak_gb(0.0, bf16)
}

/// The preflight verdict with the auxiliary models' `extra_gb` on top of the DiT projection
/// — the dense projection, or the checkpointed baseline when `checkpointed`.
fn check_preflight_budget_with(
    edge: u32,
    bf16: bool,
    budget_gb: f64,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    if !checkpointed {
        return check_preflight_budget_extra(edge, bf16, budget_gb, extra_gb);
    }
    let projected = checkpointed_baseline_gb(bf16) + extra_gb;
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "krea trainer: a checkpointed training step at resolution {edge} with the \
             depth-anchoring models (~{extra_gb:.1} GB for TAEW2.1 and Depth-Anything-V2) needs at \
             least ~{projected:.0} GB, exceeding this machine's ~{safe:.0} GB safe budget \
             ({budget_gb:.0} GB MLX limit × 0.85). Use a smaller depth model or reduce the \
             training resolution."
        )
        .into());
    }
    Ok(())
}

/// The pure guard logic (no MLX global state, so it is unit-testable): refuse if the projected dense
/// first-step peak exceeds `budget_gb × 0.85`. `edge` is the bucketed training edge; the unified token
/// count is `(edge/16)²` (latent /8, patch 2) plus a representative caption block.
#[cfg(test)]
fn check_preflight_budget(edge: u32, bf16: bool, budget_gb: f64) -> Result<()> {
    check_preflight_budget_extra(edge, bf16, budget_gb, 0.0)
}

/// The dense guard with `extra_gb` (epic 2123 E7 auxiliary models) on top of the projection.
fn check_preflight_budget_extra(
    edge: u32,
    bf16: bool,
    budget_gb: f64,
    extra_gb: f64,
) -> Result<()> {
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + PREFLIGHT_TXT_TOKENS;
    let projected = projected_dense_peak_gb(s, bf16) + extra_gb;
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "krea trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
             would hard-kill the worker (SIGKILL) at the first step with no recoverable error. Enable \
             Gradient Checkpointing (recomputes block activations in the backward) or reduce the \
             training resolution."
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default [`DEFAULT_TARGET_MODULES`]) to full dotted
/// paths by matching them against every adapter-routable module on the DiT — the same suffix-match
/// PEFT's `LoraConfig(target_modules=…)` does. The DEFAULT set is restricted to the single-stream
/// `transformer_blocks` (the main denoiser); an explicit `lora_target_modules` matches anywhere
/// (incl. the `text_fusion` aggregator and the global projections).
fn resolve_target_paths(transformer: &Krea2Transformer, cfg: &TrainingConfig) -> Vec<String> {
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

/// Encode a center-cropped square image into a Krea training latent `[1, 16, edge/8, edge/8]` — the
/// Qwen-Image VAE encode the DiT predicts in: `preprocess_init_image` (resize + `[−1,1]` NCHW) →
/// [`QwenVae::encode`] (per-channel-normalized 16-ch latent, `[1,16,1,edge/8,edge/8]`) → drop the
/// singleton temporal axis. The encode already applies the per-channel `latents_mean`/`latents_std`
/// normalization (the DiT operates in that normalized space; `decode` de-normalizes).
fn encode_latents(vae: &QwenVae, image: &Image, edge: u32) -> Result<Array> {
    let pre = preprocess_init_image(image, edge, edge)?; // [1, 3, edge, edge] in [-1, 1]
    let lat = vae.encode(&pre)?; // [1, 16, 1, edge/8, edge/8]
    Ok(lat.squeeze_axes(&[2])?) // [1, 16, edge/8, edge/8]
}

/// Encode a caption into the DiT's `text_fusion` context `[1, n_tok, 12, 2560]` — the Qwen3-VL-4B
/// condition encoder's stacked 12 select-layers (sc-7569), the same path Turbo inference uses.
fn encode_caption(
    tokenizer: &KreaTokenizer,
    encoder: &KreaTextEncoder,
    caption: &str,
) -> Result<Array> {
    let (ids, attn) = tokenizer.encode_prompt(caption)?;
    encoder.forward(&ids, &attn)
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `t ∈ [1e-3, 1−1e-3]` — a
/// faithful port of the SceneWorks `sample_training_timestep` (identical to the Lens / Z-Image
/// trainers): `sigmoid(randn)` by default, `uniform` for linear, `(uniform + sigmoid(randn))/2` for
/// weighted; bias `high` → `√t`, `low` → `t²`. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t = match ttype.as_str() {
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
/// Krea DiT, regress the **raw** `forward()` velocity onto `noise − x0`, return `(loss, grads)`. The DiT
/// timestep is `t` (the noise fraction) directly. `dtype` is the training compute dtype; the LoRA
/// factors are cast inside the traced install (`lora_dtype`), so the DiT graph runs at `dtype`; the
/// noising math, loss, and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-single-stream-block LOCAL LoRA target paths and switches
/// the forward to the gradient-checkpointed path — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense forward.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut Krea2Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    context: &Array,
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
        context,
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

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed and the loss is the weighted perceptual term on the DiT's x0
/// estimate `x_t − t·v` (the raw velocity regresses `noise − x0`), decoded by TAEW2.1; summed
/// (`every_n == 1`) aux terms ride along on a diffusion step. The dense and block-checkpointed
/// forwards land in the same closure, so both carry it. With `aux = None` (or a plan with no aux
/// loss) the traced graph is exactly the pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    transformer: &mut Krea2Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    context: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpoint_blocks: Option<&[Vec<String>]>,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    let (x_t, target) = build_batch(x0, noise, t)?;
    let x_t_f32 = x_t.clone();
    let mask_weight = mask_weight.cloned();
    let x_t = x_t.as_dtype(dtype)?; // no-op in f32 mode
    let timestep = Array::from_slice(&[t], &[1]);
    let context = context.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets so the dense path (and any non-checkpointed text-fusion / global targets)
        // train through ordinary autograd; on the checkpointed path the single-stream block adapters
        // installed here are replaced inside each checkpoint segment by the explicit-input factors.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let v = match checkpoint_blocks {
            Some(locals) => transformer
                .forward_with_blocks_checkpointed(
                    &x_t, &timestep, &context, None, &p, locals, alpha,
                )
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward(&x_t, &timestep, &context, None)
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
                // x0 estimate in f32 from the raw velocity (`noise − x0`): x0 = x_t − t·v, already
                // the decoder's NCHW `[1, 16, h, w]` layout.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma: t }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
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

/// Render a preview image from the in-progress (already-installed) adapter on the **Raw** DiT (sc-5637):
/// the rectified-flow Euler integration with classifier-free guidance (`ctx_pos` vs the shared empty
/// `ctx_neg`), at the Raw resolution-dynamic `mu` schedule. Indicative of the LoRA's learned content
/// (the deployed `krea_2_turbo` render is CFG-free few-step; the Raw preview is the only render the
/// trainer's loaded weights can produce). A best-effort nicety — failures are logged, not fatal.
#[allow(clippy::too_many_arguments)]
fn render_sample(
    dit: &Krea2Transformer,
    vae: &QwenVae,
    ctx_pos: &Array,
    ctx_neg: &Array,
    seed: u64,
    edge: u32,
    steps: usize,
    guidance: f32,
    cancel: &CancelFlag,
) -> Result<Image> {
    let (hl, wl) = ((edge / 8) as i32, (edge / 8) as i32);
    let noise = random::normal::<f32>(&[1, 16, hl, wl], None, None, Some(&random::key(seed)?))?;
    let img_seq = (edge as f64 / 16.0).powi(2);
    let sigmas = krea_sigmas(steps, dynamic_mu(img_seq));
    // Hoist the step-invariant text fusion + host RoPE out of the per-step closure (F-079), matching the
    // inference render paths: `prepare` depends only on the context + latent geometry (never the noise
    // values or timestep), so building the prep(s) ONCE and reusing them via `forward_prepared` is
    // bit-identical to the per-step `forward` (which was `prepare` + `forward_prepared` internally).
    let prep_pos = dit.prepare(ctx_pos, None, &noise)?;
    let prep_neg = if guidance > 0.0 {
        Some(dit.prepare(ctx_neg, None, &noise)?)
    } else {
        None
    };
    // F-077: honor the training request's cancel per denoise step. Previously a fresh
    // `CancelFlag::new()` here meant the preview render could never be cancelled mid-denoise.
    let lat = run_flow_sampler(
        None,
        TimestepConvention::Sigma,
        &sigmas,
        noise,
        seed,
        cancel,
        &mut |_: Progress| {},
        |x, timestep| {
            let tt = Array::from_slice(&[timestep], &[1]);
            let v_cond = dit.forward_prepared(x, &tt, &prep_pos)?;
            // CFG via the shared [`krea_cfg_combine`] (reference `sampling.py:129`
            // `v_cond + g·(v_cond − v_uncond)`, one source of truth with the Raw inference path). Prior
            // to sc-10009 this used the standard `v_uncond + g·Δ`, which at the shared default
            // `sample_guidance_scale = 1.0` collapsed to exactly `v_cond` — zero effective CFG, the
            // washed-out previews. `guidance == 0` still collapses to the bare conditional velocity.
            let v = match &prep_neg {
                Some(neg) => {
                    let v_uncond = dit.forward_prepared(x, &tt, neg)?;
                    krea_cfg_combine(&v_cond, &v_uncond, guidance)?
                }
                None => v_cond,
            };
            Ok(v.as_dtype(Dtype::Float32)?)
        },
    )?;
    let decoded = vae.decode(&lat)?.as_dtype(Dtype::Float32)?;
    decoded_to_image(&decoded)
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
                "krea trainer",
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
            output_dir: PathBuf::from("/tmp/krea_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_raw_base_id() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "krea_2_raw");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
        // sc-24828: the one loss path (dense + block-checkpointed) reduces through the subject-mask
        // weight; the latent is unpacked `[1, 16, H, W]`, so the weight is the cached latent's shape.
        assert!(d.techniques.subject_mask_loss);
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
    fn validate_accepts_gradient_checkpointing() {
        let r = req_with(TrainingConfig {
            gradient_checkpointing: true,
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
        // The recognized spellings (incl. alias normalization) pass.
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Weighted".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_krea_velocity_with_no_sign_flip() {
        // target = noise − x0 (the RAW Krea DiT velocity; the SAME sign as Lens, OPPOSITE z-image), and
        // x_t = (1−t)·x0 + t·noise. Timestep `t` is passed straight to the DiT.
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let (x_t, target) = build_batch(&x0, &noise, 0.25).unwrap();
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]); // noise − x0
        let xt = x_t.as_slice::<f32>();
        for (got, want) in xt.iter().zip([1.75f32, 3.25, 4.75].iter()) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn sample_sigma_is_deterministic_and_in_range() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                let b = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, b, "{kind}/{bias} must be deterministic in seed");
                assert!(
                    (1e-3..=1.0 - 1e-3).contains(&a),
                    "{kind}/{bias} t={a} out of range"
                );
            }
        }
        assert!(
            sample_sigma("sigmoid", "high", 7).unwrap()
                > sample_sigma("sigmoid", "low", 7).unwrap()
        );
    }

    #[test]
    fn default_target_modules_are_block_attention() {
        // The default training surface is the standard PEFT attention projections (the real-weight
        // resolver also restricts to `transformer_blocks.*` — pinned in the #[ignore] harness).
        assert_eq!(DEFAULT_TARGET_MODULES, ["to_q", "to_k", "to_v", "to_out.0"]);
    }

    #[test]
    fn preflight_guard_fires_over_budget_and_passes_under() {
        // A 24 GB-class budget (safe ≈ 20.4 GB): the 24 GB bf16 DiT base alone exceeds it → dense 512
        // must be refused with an actionable error recommending Gradient Checkpointing.
        let err = check_preflight_budget(512, true, 24.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "got: {err}");
        assert!(
            err.contains("512"),
            "error should name the resolution: {err}"
        );
        // A 128 GB-class budget (safe ≈ 108 GB) comfortably fits dense 512 in both dtypes.
        assert!(check_preflight_budget(512, true, 128.0).is_ok());
        assert!(check_preflight_budget(512, false, 128.0).is_ok());
    }

    #[test]
    fn descriptor_declares_resolution_buckets() {
        // sc-2127: the shared technique floor only lets `resolution_buckets` through when declared.
        assert!(trainer_descriptor().techniques.resolution_buckets);
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
        // A 48 GB-class budget (safe ≈ 40.8 GB): dense bf16 fits 512 (~31 GB) but not 1024 (~52 GB) —
        // so the bucketed run is refused even though `resolution` alone would pass.
        assert!(check_preflight_budget(512, true, 48.0).is_ok());
        assert!(check_preflight_budget(preflight_edge(&edges), true, 48.0).is_err());
    }

    #[test]
    fn reachable_via_trainer_registry_by_id() {
        // The trainer registration is discoverable through the explicit family catalog.
        assert!(
            crate::provider_registry()
                .unwrap()
                .trainers()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_RAW_TRAINER_ID),
            "trainer id {KREA_2_RAW_TRAINER_ID} not registered"
        );
    }

    #[test]
    fn projection_is_monotonic_and_conservative() {
        // Monotonic increasing in token count, in both dtypes; bf16 below f32 (half the weights +
        // activations). s for edge 512/768/1024 = (edge/16)² + 64.
        for bf16 in [false, true] {
            let (s512, s768, s1024) = (1088.0, 2368.0, 4160.0);
            assert!(projected_dense_peak_gb(s512, bf16) < projected_dense_peak_gb(s768, bf16));
            assert!(projected_dense_peak_gb(s768, bf16) < projected_dense_peak_gb(s1024, bf16));
        }
        assert!(projected_dense_peak_gb(4160.0, true) < projected_dense_peak_gb(4160.0, false));
        // The bf16 base (no tokens) is ~the resident DiT weights (≥ 20 GB) — the floor the guard adds to.
        assert!(projected_dense_peak_gb(0.0, true) >= 20.0);
    }

    /// sc-24828: the subject-mask weight reaches BOTH backward paths (dense + block-checkpointed) of
    /// [`compute_loss_grads`] on the checked-in tiny `dit_golden` fixture DiT (no real weights). An
    /// all-ones map equals the unweighted loss; an all-zero map gives loss exactly 0 and all-zero
    /// adapter grads on both paths; a half map lands strictly between and agrees across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        use mlx_gen::train::loss::subject_mask_weight;
        use mlx_gen::weights::Weights;
        let w = Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dit_golden.safetensors"
        ))
        .unwrap();
        // Mirrors `tests/common::tiny_dit_config` (the fixture's dump geometry).
        let dit_cfg = crate::config::Krea2Config {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 128,
            num_attention_heads: 4,
            num_kv_heads: 2,
            attention_head_dim: 32,
            num_layers: 2,
            intermediate_size: 384,
            norm_eps: 1e-5,
            axes_dims_rope: [8, 12, 12],
            rope_theta: 1000.0,
            timestep_embed_dim: 64,
            num_text_layers: 3,
            num_layerwise_text_blocks: 2,
            num_refiner_text_blocks: 2,
            text_hidden_dim: 64,
            text_intermediate_size: 256,
            text_num_attention_heads: 2,
            text_num_kv_heads: 2,
        };
        dit_cfg.validate().unwrap();
        let mut dit = Krea2Transformer::from_weights(&w, &dit_cfg).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        assert!(!target_paths.is_empty());
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
        // Non-zero factors on both sides (the up-projection inits at zero, which would zero the
        // down-projection grads trivially).
        let scale = Array::from_slice(&[0.02f32], &[1]);
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
        let x0 = w.require("in.latent").unwrap().clone();
        let context = w.require("in.context").unwrap().clone();
        let shape = x0.shape().to_vec();
        let (gh, gw) = (shape[2] as usize, shape[3] as usize);
        let noise =
            random::normal::<f32>(&shape, None, None, Some(&random::key(2).unwrap())).unwrap();
        let map = |v: &[f32]| subject_mask_weight(v, gh, gw, &shape).unwrap();
        let mut run = |weight: Option<&Array>, ckpt: bool| {
            let (l, g) = compute_loss_grads(
                &mut dit,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &context,
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
        let n = gh * gw;
        let (plain, _) = run(None, false);
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
}

// ===========================================================================================
// sc-7577 — real-weight grad-parity harness (weight-gated, run as its own process).
//
//   cargo test -p mlx-gen-krea --release --lib real_weight_repro -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod real_weight_repro {
    use super::*;
    use std::path::PathBuf;

    /// The base `krea/Krea-2-Raw` snapshot root from the required `KREA_RAW_DIR` env var. sc-13668:
    /// there is no implicit default — the source snapshot path must be passed in explicitly.
    fn snapshot() -> Option<PathBuf> {
        std::env::var("KREA_RAW_DIR").ok().map(PathBuf::from)
    }

    #[test]
    fn source_root_requires_explicit_env_no_default() {
        let key = "KREA_RAW_DIR";
        let saved = std::env::var(key).ok();
        std::env::remove_var(key);
        assert!(
            snapshot().is_none(),
            "the source snapshot root must come from {key}: sc-13668 removed the implicit default"
        );
        std::env::set_var(key, "/sentinel/krea");
        assert_eq!(snapshot(), Some(PathBuf::from("/sentinel/krea")));
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    /// Per-block LOCAL LoRA target paths (mirrors `train_impl`), for driving the checkpointed path.
    fn block_local_targets(dit: &Krea2Transformer, target_paths: &[String]) -> Vec<Vec<String>> {
        let mut out: Vec<Vec<String>> = vec![Vec::new(); dit.num_blocks()];
        for path in target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < out.len() {
                        out[i].push(local.to_string());
                    }
                }
            }
        }
        out
    }

    fn max_rel_diff(ga: &LoraParams, gb: &LoraParams) -> f32 {
        let mut max_rel = 0f32;
        for (k, a) in ga {
            let b = gb.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        max_rel
    }

    /// The real DiT's default target surface resolves to the single-stream block attention only:
    /// 28 blocks × {to_q, to_k, to_v, to_out.0} = 112 targets, none on `text_fusion`/globals.
    #[test]
    #[ignore = "needs real krea/Krea-2-Raw weights; run as its own process"]
    fn default_targets_resolve_to_block_attention() {
        let root = snapshot().expect("set KREA_RAW_DIR to the krea/Krea-2-Raw snapshot root");
        let dit = load_transformer(&root).unwrap();
        let cfg = TrainingConfig::default();
        let paths = resolve_target_paths(&dit, &cfg);
        assert_eq!(paths.len(), dit.num_blocks() * 4, "112 attention targets");
        assert!(
            paths.iter().all(|p| p.starts_with("transformer_blocks.")),
            "default targets are single-stream blocks only"
        );
        assert!(paths.iter().any(|p| p.ends_with(".attn.to_q")));
    }

    /// Whole-block gradient checkpointing must not change the math: the checkpointed forward+grads must
    /// match the dense path within fp tolerance (it reuses the same install + block forward, recompute
    /// only). Run in f32 at a tiny resolution — the math is resolution-agnostic.
    #[test]
    #[ignore = "needs real krea/Krea-2-Raw weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let root = snapshot().expect("set KREA_RAW_DIR to the krea/Krea-2-Raw snapshot root");
        let mut dit = load_transformer(&root).unwrap();
        dit.cast_weights(Dtype::Float32).unwrap(); // f32 → clean parity, no bf16 noise
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
        let adapter = TrainAdapter::Lora { targets };
        let locals = block_local_targets(&dit, &target_paths);

        // Tiny synthetic batch (latent 16×16 → img tokens 64; the graph SIZE is irrelevant to the math
        // being equal). Context = 8 caption tokens × 12 layers × 2560.
        let x0 =
            random::normal::<f32>(&[1, 16, 16, 16], None, None, Some(&random::key(1).unwrap()))
                .unwrap();
        let noise =
            random::normal::<f32>(&[1, 16, 16, 16], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        let context = random::normal::<f32>(
            &[1, 8, 12, 2560],
            None,
            None,
            Some(&random::key(3).unwrap()),
        )
        .unwrap();

        dit.set_sdpa_checkpoint(true); // dense production path (SDPA-ckpt on)
        let (_l, g_dense) = compute_loss_grads(
            &mut dit,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &context,
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

        dit.set_sdpa_checkpoint(false); // block-checkpointed path (block recompute covers attention)
        let (_l, g_ckpt) = compute_loss_grads(
            &mut dit,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &context,
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

        let rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-7577] checkpointed-vs-dense grad max relative diff: {rel:.2e}");
        assert!(
            rel < 1e-3,
            "checkpointed grads must match dense within tolerance: max rel {rel:.2e}"
        );
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the Krea step seam on the checked-in tiny `dit_golden`
/// DiT (4-channel latent `[1, 4, 8, 8]`) with a random-init tiny-width TAEHV carrying TAEW2.1's
/// hyperparameters at the fixture's 4 latent channels, and a random-init tiny Depth-Anything-V2.
/// Drives the same [`compute_step_loss_grads`] / [`plan_aux_step`] /
/// [`prepare_perceptual_references`] `train_impl` runs. Seconds; no weights downloaded.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule};
    use mlx_gen::train::taehv::{synthetic_taehv_weights, TaehvDecoder};
    use mlx_gen::weights::Weights;
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
        dit: Krea2Transformer,
        adapter: TrainAdapter,
        params: LoraParams,
        locals: Vec<Vec<String>>,
        x0: Array,
        context: Array,
        noise: Array,
    }

    fn fixture() -> Fixture {
        let w = Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dit_golden.safetensors"
        ))
        .unwrap();
        let dit_cfg = crate::config::Krea2Config {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 128,
            num_attention_heads: 4,
            num_kv_heads: 2,
            attention_head_dim: 32,
            num_layers: 2,
            intermediate_size: 384,
            norm_eps: 1e-5,
            axes_dims_rope: [8, 12, 12],
            rope_theta: 1000.0,
            timestep_embed_dim: 64,
            num_text_layers: 3,
            num_layerwise_text_blocks: 2,
            num_refiner_text_blocks: 2,
            text_hidden_dim: 64,
            text_intermediate_size: 256,
            text_num_attention_heads: 2,
            text_num_kv_heads: 2,
        };
        let mut dit = Krea2Transformer::from_weights(&w, &dit_cfg).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
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
        let x0 = w.require("in.latent").unwrap().clone();
        let context = w.require("in.context").unwrap().clone();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(2).unwrap())).unwrap();
        eval([&x0, &context, &noise]).unwrap();
        Fixture {
            dit,
            adapter: TrainAdapter::Lora { targets },
            params,
            locals,
            x0,
            context,
            noise,
        }
    }

    fn step(
        f: &mut Fixture,
        t: f32,
        ckpt: bool,
        aux: Option<AuxStep<'_>>,
    ) -> (StepLosses, LoraParams) {
        f.dit.set_sdpa_checkpoint(!ckpt);
        let (l, g) = compute_step_loss_grads(
            &mut f.dit,
            &f.params,
            &f.adapter,
            4.0,
            4.0,
            &f.x0,
            &f.context,
            t,
            &f.noise,
            false,
            None,
            Dtype::Float32,
            None,
            ckpt.then_some(f.locals.as_slice()),
            aux,
        )
        .unwrap();
        eval(g.values()).unwrap();
        (l, g)
    }

    fn cache_of(x0: &Array, context: &Array, n: usize) -> Vec<(Array, Array, Option<Array>)> {
        (0..n)
            .map(|_| (x0.clone(), context.clone(), None))
            .collect()
    }

    fn bits(a: &Array) -> Vec<u32> {
        a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
    }

    /// AC (a)+(b), dense and block-checkpointed: a depth step (key 2) computes no diffusion term,
    /// its total IS the weighted depth term, and the zero-init LoRA-B factors get a nonzero finite
    /// gradient; a diffusion step (key 1) carries no depth term. Mutation: force
    /// `diffusion_on = true` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only_on_both_paths() {
        let mut f = fixture();
        let mut p = path();
        prepare_perceptual_references(&mut p, &cache_of(&f.x0, &f.context, 1)).unwrap();
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
            let gb: f32 = g
                .iter()
                .filter(|(k, _)| k.ends_with(".lora_b"))
                .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
                .sum();
            assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B grad {gb}");
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
        let (x_t, target) = build_batch(&f.x0, &f.noise, 0.5).unwrap();
        let timestep = Array::from_slice(&[0.5f32], &[1]);
        let context = f.context.clone();
        let dit = &mut f.dit;
        let adapter = &f.adapter;
        dit.set_sdpa_checkpoint(true);
        let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            adapter.install_as(dit, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            let v = dit
                .forward(&x_t, &timestep, &context, None)
                .map_err(|e| Exception::custom(e.to_string()))?;
            Ok(vec![reduce_loss(&subtract(&v, &target)?, None, false)?])
        };
        let (val, g_legacy) = keyed_value_and_grad(legacy)(f.params.clone(), 0).unwrap();
        eval(g_legacy.values()).unwrap();
        assert_eq!(off.total, val[0].item::<f32>());
        for (k, v) in &g_legacy {
            assert_eq!(bits(v), bits(&g_off[k]), "{k}");
        }
        let mut p = path();
        prepare_perceptual_references(&mut p, &cache_of(&f.x0, &f.context, 1)).unwrap();
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

    /// With two resolution buckets (item-major cache, seeded shuffle) every image strictly
    /// alternates diffusion / depth across its buckets in visit order — the key comes from the
    /// real item, the reference from the (item, bucket) entry — and references are built once per
    /// entry. Mutation: key on the cache entry instead of the item ⇒ no strict per-image
    /// alternation ⇒ red.
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
        let cache = cache_of(&f.x0, &f.context, items * buckets.len());
        let mut p = path();
        prepare_perceptual_references(&mut p, &cache).unwrap();
        let schedule = BucketSchedule::new(items, &buckets, 7);
        let mut alt = AuxAlternation::new(items, 1);
        let steps = 2 * schedule.epoch_len() as u32;
        let mut kinds = Vec::new();
        for s in 1..=steps {
            let plan = plan_aux_step(&p, &mut alt, &schedule, s, 0.5).unwrap();
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
        assert_eq!(p.reference_computations(), cache.len());
    }

    /// AC (d), E7: depth grows the estimate by TAEW2.1 + DA2 (more for Large) and the guard counts
    /// it on the dense and the checkpointed path (synthetic budgets). Mutations: drop
    /// `+ extra_gb` (dense) ⇒ red; drop it from the checkpointed projection ⇒ red.
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
        let dense_tokens = (512.0f64 / 16.0).ceil().powi(2) + PREFLIGHT_TXT_TOKENS;
        let dense = projected_dense_peak_gb(dense_tokens, true);
        let between = |base: f64| (base + large / 2.0) / 0.85;
        assert!(check_preflight_budget_with(512, true, between(dense), 0.0, false).is_ok());
        assert!(check_preflight_budget_with(512, true, between(dense), large, false).is_err());
        let ck = checkpointed_baseline_gb(true);
        assert!(check_preflight_budget_with(512, true, between(ck), 0.0, true).is_ok());
        let err = check_preflight_budget_with(512, true, between(ck), large, true)
            .expect_err("checkpointed depth job over budget")
            .to_string();
        assert!(err.contains("depth-anchoring"), "{err}");
    }

    /// AC (e): the descriptor declares depth anchoring; a missing TAEW2.1 checkpoint is an error
    /// naming it; a real (random-init, full-size) TAEW2.1 file loads through the shared builder
    /// (then the missing DA2 is named).
    #[test]
    fn descriptor_declares_depth_and_missing_decoder_is_named() {
        assert!(trainer_descriptor().techniques.depth_anchoring);
        let tmp = tempfile::tempdir().unwrap();
        let mut c = TrainingConfig::default();
        c.depth_anchoring.schedule = schedule();
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taehv"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c).err().unwrap().to_string();
        assert!(err.contains("TAEW2.1"), "{err}");
        let dec_dir = tmp.path().join("taehv");
        std::fs::create_dir_all(&dec_dir).unwrap();
        let w = synthetic_taehv_weights(&TaehvConfig::taew2_1(), 1).unwrap();
        let pairs: Vec<(String, Array)> = w
            .keys()
            .map(|k| (k.to_string(), w.require(k).unwrap().clone()))
            .collect();
        let refs: Vec<(&str, &Array)> = pairs.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, dec_dir.join("taew2_1.safetensors")).unwrap();
        c.perceptual_decoder_dir = Some(dec_dir);
        let err = load_perceptual_path(&c).err().unwrap().to_string();
        assert!(err.contains("Depth-Anything-V2"), "{err}");
    }
}
